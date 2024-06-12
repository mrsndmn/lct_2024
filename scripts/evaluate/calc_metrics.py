import torch
import os
from qdrant_client import QdrantClient, models

from datasets import Dataset
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from avm.search.audio import AudioIndex
from avm.fingerprint.audio import Segment

from scripts.evaluate.borrow_intervals import get_matched_segments, IntervalsConfig

@dataclass
class EvaluationConfig:
    interval_step: int
    interval_duration_in_seconds: int
    full_interval_duration_in_seconds: int
    sampling_rate: int
    # matched_threshold: float

    index_embeddings_path: str = field(default='data/music_caps/audio_embeddings')
    query_embeddings_path: str = field(default='data/music_caps/augmented_embeddings')
    queries_dataset: str = field(default='data/music_caps/augmented_audios.dataset') # датасет с разметкой, на каком моменте начались реальные данные
    file_id_field_name: str = field(default="youtube_id")

    metrics_log_path: str = field(default='data/music_caps/metrics')

    verbose: bool = field(default=True)


# вычисляет метрику хорошести эмбэддингов
def evaluate_metrics(config: EvaluationConfig, metrics_df):
    df = metrics_df

    uniq_file_ids = sorted(df['file_id'].unique())

    file_id_to_index = { ytid: i for i, ytid in enumerate(uniq_file_ids) }

    # print("uniq_file_ids", len(uniq_file_ids))

    interval_start_offset = df['interval_i'] * (config.interval_step * config.sampling_rate)
    full_interval_dutation_samples = config.sampling_rate * config.full_interval_duration_in_seconds
    ending_padding_samples = config.interval_duration_in_seconds * config.sampling_rate
    df['interval_should_match'] = (df['augmented_audio_offset'] <  interval_start_offset) & (interval_start_offset < df['augmented_audio_offset'] + full_interval_dutation_samples - ending_padding_samples)

    matched_predictions = df # df[df['interval_should_match']]
    # print("matched_predictions", len(matched_predictions))

    num_hits = (matched_predictions['hit_i'].max() + 1)
    interval_seconds = len(df) // num_hits // len(uniq_file_ids)

    num_samples = len(uniq_file_ids) * interval_seconds
    
    index_no_class = len(uniq_file_ids)

    score_table = np.zeros([ num_samples, len(uniq_file_ids) + 1 ])
    target_score_table = np.zeros([ num_samples ])
    for i, (_, row) in enumerate(matched_predictions.iterrows()):
        sample_i = i // num_hits
        row_file_id = row['file_id']
        
        file_id_index = file_id_to_index[row_file_id]

        if row['hit_i'] == 0:
            if row['interval_should_match']:
                target_score_table[sample_i] = file_id_index
            else:
                target_score_table[sample_i] = index_no_class

        hit_file_id = row['hit_file_id']
        hit_file_id_index = file_id_to_index[hit_file_id]

        score_table[sample_i, hit_file_id_index] += row['hit_score']

    normalized_score_table = score_table / (score_table.sum(axis=1, keepdims=True) + 1e-9)

    rocauc_score = roc_auc_score(target_score_table, normalized_score_table, multi_class='ovr')

    return {
        "rocauc_score": rocauc_score
    }


def evaluate_matching(config: EvaluationConfig):

    audio_index = AudioIndex(config.index_embeddings_path)

    metrics_log_path = config.metrics_log_path
    os.makedirs(metrics_log_path, exist_ok=True)

    query_embeddings_path = config.query_embeddings_path
    query_embeddings_files = set(sorted(os.listdir(query_embeddings_path)))
    queries_dataset = Dataset.load_from_disk(config.queries_dataset)

    metrics_log = []
    file_not_found = set()

    matched_segments = []
    target_segments = []

    for query_item in queries_dataset:
        file_name = query_item['file_name']
        file_name = file_name.replace('.wav', '.pt')
        if file_name not in query_embeddings_files:
            file_not_found.add(file_name)
            continue

        query_embedding = torch.load(os.path.join(query_embeddings_path, file_name))

        augmentation_name = query_item['augmentation']
        file_id = query_item[config.file_id_field_name]

        start_of_target_segment = int(query_item['augmented_audio_offset'] / 16000)
        target_segments.append([
            Segment(file_id=file_id, start_second=start_of_target_segment, end_second=start_of_target_segment + 10),
            Segment(file_id=file_id, start_second=0, end_second=10),
        ])
        
        query_hits = audio_index.search_sequential(
            query_vectors=query_embedding.numpy(),
            limit_per_vector=10
        )

        intervals_config = IntervalsConfig()
        query_matched_segments = get_matched_segments(intervals_config, file_id, query_hits)
        matched_segments.append(query_matched_segments)

        for interval_i in range(len(query_hits)):

            hits = query_hits[interval_i]

            for i, hit in enumerate(hits):
                hit_file_id = hit.payload['file_id']
                hit_interval_i = hit.payload['interval_num']

                metrics_log.append({
                    "hit_i": i,
                    "hit_score": hit.score,
                    "hit_file_id": hit_file_id,
                    "hit_interval_i": hit_interval_i,
                    "file_id": file_id,
                    "file_name": file_name,
                    "augmentation": augmentation_name,
                    "interval_i": interval_i,
                    "augmented_audio_offset": query_item['augmented_audio_offset'],
                })

    metrics_dataset = Dataset.from_list(metrics_log)

    metrics_full_path = os.path.join(metrics_log_path, "metrics.dataset")
    metrics_dataset.save_to_disk(metrics_full_path)
    metrics_dataframe = metrics_dataset.to_pandas()

    print("query_matched_segments", len(query_matched_segments))
    evaluate_iou(query_matched_segments, target_segments)

    metrics_values = evaluate_metrics(config, metrics_dataframe)
    if config.verbose:
        print("metrics_values", metrics_values)
        print("metrics_full_path", metrics_full_path)

    if len(file_not_found) > 0:
        print("not found embedding files for queries:", len(file_not_found))

    return metrics_values

if __name__ == '__main__':
    eval_config = EvaluationConfig(
        index_embeddings_path='./data/music_caps/pipeline/1718135254/index_embeddings',
        query_embeddings_path='./data/music_caps/pipeline/1718135254/validation_embeddings',
        metrics_log_path='./data/music_caps/pipeline/1718135254/metrics',
        sampling_rate=16000,
        full_interval_duration_in_seconds=10,
        interval_duration_in_seconds=5,
        interval_step=1,
        # matched_threshold=pipeline_config.matched_threshold
        verbose=True
    )
    evaluate_matching(eval_config)
    
import torch
import os
from qdrant_client import QdrantClient, models

from datasets import Dataset
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from avm.search.audio import AudioIndex

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

    metrics_log_path: str = field(default='data/music_caps/metrics')

    verbose: bool = field(default=False)


# вычисляет метрику хорошести эмбэддингов
def evaluate_metrics(config: EvaluationConfig, metrics_df):
    df = metrics_df

    uniq_youtube_ids = sorted(df['youtube_id'].unique())

    youtube_id_to_index = { ytid: i for i, ytid in enumerate(uniq_youtube_ids) }

    # print("uniq_youtube_ids", len(uniq_youtube_ids))

    interval_start_offset = df['interval_i'] * (config.interval_step * config.sampling_rate)
    full_interval_dutation_samples = config.sampling_rate * config.full_interval_duration_in_seconds
    ending_padding_samples = config.interval_duration_in_seconds * config.sampling_rate
    df['interval_should_match'] = (df['augmented_audio_offset'] <  interval_start_offset) & (interval_start_offset < df['augmented_audio_offset'] + full_interval_dutation_samples - ending_padding_samples)

    matched_predictions = df # df[df['interval_should_match']]
    # print("matched_predictions", len(matched_predictions))

    num_hits = (matched_predictions['hit_i'].max() + 1)
    interval_seconds = len(df) // num_hits // len(uniq_youtube_ids)

    num_samples = len(uniq_youtube_ids) * interval_seconds
    
    index_no_class = len(uniq_youtube_ids)

    score_table = np.zeros([ num_samples, len(uniq_youtube_ids) + 1 ])
    target_score_table = np.zeros([ num_samples ])
    for i, (_, row) in enumerate(matched_predictions.iterrows()):
        sample_i = i // num_hits
        row_youtube_id = row['youtube_id']
        
        youtube_id_index = youtube_id_to_index[row_youtube_id]

        if row['hit_i'] == 0:
            if row['interval_should_match']:
                target_score_table[sample_i] = youtube_id_index
            else:
                target_score_table[sample_i] = index_no_class

        hit_youtube_id = row['hit_youtube_id']
        hit_youtube_id_index = youtube_id_to_index[hit_youtube_id]

        score_table[sample_i, hit_youtube_id_index] += row['hit_score']

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

    for query_item in queries_dataset:
        file_name = query_item['file_name']
        file_name = file_name.replace('.wav', '.pt')
        if file_name not in query_embeddings_files:
            file_not_found.add(file_name)
            continue

        query_embedding = torch.load(os.path.join(query_embeddings_path, file_name))

        augmentation_name = query_item['augmentation']
        youtube_id = query_item['youtube_id']
        
        query_hits = audio_index.search_sequential(
            query_vectors=query_embedding.numpy(),
            limit_per_vector=10
        )

        for interval_i in range(len(query_hits)):

            hits = query_hits[interval_i]

            for i, hit in enumerate(hits):
                hit_youtube_id = hit.payload['youtube_id']
                hit_interval_i = hit.payload['interval_num']

                metrics_log.append({
                    "hit_i": i,
                    "hit_score": hit.score,
                    "hit_youtube_id": hit_youtube_id,
                    "hit_interval_i": hit_interval_i,
                    "youtube_id": youtube_id,
                    "file_name": file_name,
                    "augmentation": augmentation_name,
                    "interval_i": interval_i,
                    "augmented_audio_offset": query_item['augmented_audio_offset'],
                })


    metrics_dataset = Dataset.from_list(metrics_log)
    metrics_dataset.save_to_disk(metrics_log_path + "/metrics.dataset")
    metrics_dataframe = metrics_dataset.to_pandas()

    metrics_values = evaluate_metrics(config, metrics_dataframe)
    if config.verbose:
        print("metrics_values", metrics_values)

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
    
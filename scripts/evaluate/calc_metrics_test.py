# import pytest
from datasets import Dataset
import numpy as np

from sklearn.metrics import roc_auc_score

def test_metrics_calc():
    dataset_path = "data/music_caps/pipeline/1718132751/metrics/metrics.dataset/"
    df = Dataset.load_from_disk(dataset_path).to_pandas()

    uniq_youtube_ids = sorted(df['youtube_id'].unique())

    youtube_id_to_index = { ytid: i for i, ytid in enumerate(uniq_youtube_ids) }

    print("uniq_youtube_ids", len(uniq_youtube_ids))

    interval_start_offset = df['interval_i'] * (1 * 16000)
    full_interval_dutation_samples = 16000 * 10
    ending_padding_samples = 5 * 16000
    df['interval_should_match'] = (df['augmented_audio_offset'] <  interval_start_offset) & (interval_start_offset < df['augmented_audio_offset'] + full_interval_dutation_samples - ending_padding_samples)

    matched_predictions = df # df[df['interval_should_match']]
    print("matched_predictions", len(matched_predictions))

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

        # if score_table[sample_i, youtube_id_index] != 0:
        #     raise Exception("index is already filled [{sample_i}, {youtube_id_index}]")

        hit_youtube_id = row['hit_youtube_id']
        hit_youtube_id_index = youtube_id_to_index[hit_youtube_id]

        score_table[sample_i, hit_youtube_id_index] += row['hit_score']

    normalized_score_table = score_table / (score_table.sum(axis=1, keepdims=True) + 1e-9)

    rocauc_score = roc_auc_score(target_score_table, normalized_score_table, multi_class='ovr')

    print(rocauc_score)

    raise Exception


if __name__ == '__main__':
    test_metrics_calc()
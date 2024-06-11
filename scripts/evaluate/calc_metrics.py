import torch
import os
from qdrant_client import QdrantClient, models

from datasets import Dataset
from dataclasses import dataclass, field


@dataclass
class EvaluationConfig:
    interval_step: int
    interval_duration_in_seconds: int
    full_interval_duration_in_seconds: int
    sampling_rate: int
    # matched_threshold: float

    distance_metric: models.Distance = field(default=models.Distance.EUCLID)
    base_embeddings_path: str = field(default='data/music_caps/audio_embeddings')
    base_augmented_embeddings_path: str = field(default='data/music_caps/augmented_embeddings')
    qdrant_collection_name: str = field(default="audio_embeddings")
    metrics_log_path: str = field(default='data/music_caps/metrics')
    augmented_dataset_path: str = field(default="data/music_caps/augmented_audios.dataset")

    verbose: bool = field(default=False)



def evaluate_matching(config: EvaluationConfig):
    base_embeddings_path = config.base_embeddings_path
    base_augmented_embeddings_path = config.base_augmented_embeddings_path
    qdrant_collection_name = config.qdrant_collection_name
    metrics_log_path = config.metrics_log_path
    augmented_dataset_path = config.augmented_dataset_path

    augmented_dataset = Dataset.load_from_disk(augmented_dataset_path)

    os.makedirs(metrics_log_path, exist_ok=True)

    embeddings_files = sorted(os.listdir(base_embeddings_path))
    augmented_embeddings_files = sorted(os.listdir(base_augmented_embeddings_path))

    embeddings_by_youtube_id = {}
    index_points = []

    idx_point_i = 0
    for file_name in embeddings_files:
        embedding = torch.load(base_embeddings_path + '/' + file_name)
        youtube_id = file_name.removesuffix('.pt')

        for interval_num in range(embedding.shape[0]):
            index_point = models.PointStruct(
                id=idx_point_i,
                vector=embedding[interval_num],
                payload={"youtube_id": youtube_id, "interval_num": interval_num},
            )
            idx_point_i += 1
            index_points.append(index_point)

        embeddings_by_youtube_id[youtube_id] = embedding

    embedding_size = embedding.shape[1]

    qdrant = QdrantClient(":memory:")
    # Create collection to store books
    qdrant.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=config.distance_metric,
        )
    )

    qdrant.upload_points(
        collection_name=qdrant_collection_name,
        points=index_points,
    )

    metrics_log = []
    most_clothest_counts = 0
    total_intervals = 0
    no_embeddings_for_files = 0
    for augmented_item in augmented_dataset:
        file_name = augmented_item['file_name'].replace('.wav', '.pt')
        if file_name not in augmented_embeddings_files:
            no_embeddings_for_files += 1
            # print(f"no embedding for {file_name}")
            continue

        embedding = torch.load(base_augmented_embeddings_path + '/' + file_name)

        file_parts = file_name.removesuffix('.pt').split("_")
        augmentation_name = file_parts[-1]
        youtube_id = "_".join(file_parts[:-1])

        for interval_i in range(embedding.shape[0]):

            hits = qdrant.search(
                collection_name=qdrant_collection_name,
                query_vector=embedding[interval_i].numpy(),
                limit=10,
            )
            total_intervals += 1

            for i, hit in enumerate(hits):
                hit_youtube_id = hit.payload['youtube_id']
                hit_interval_i = hit.payload['interval_num']

                interval_matches_true_injection = augmented_item['augmented_audio_offset'] < interval_i * config.interval_step * config.sampling_rate < augmented_item['augmented_audio_offset'] + config.sampling_rate * config.full_interval_duration_in_seconds
                if i == 0 and youtube_id == hit_youtube_id and interval_matches_true_injection:
                    most_clothest_counts += 1

                metrics_log.append({
                    "hit_i": i,
                    "hit_score": hit.score,
                    "hit_youtube_id": hit_youtube_id,
                    "hit_interval_i": hit_interval_i,
                    "youtube_id": youtube_id,
                    "file_name": file_name,
                    "augmentation": augmentation_name,
                    "interval_i": interval_i,
                    "interval_matches_true_injection": interval_matches_true_injection,
                    "augmented_audio_offset": augmented_item['augmented_audio_offset'],
                })

    if config.verbose:
        print("no_embeddings_for_files", no_embeddings_for_files)
        print("augmented_embeddings_files", len(augmented_embeddings_files))
        print("found clothest:", len(metrics_log))

    metrics_dataset = Dataset.from_list(metrics_log)
    metrics_dataset.save_to_disk(metrics_log_path + "/metrics.dataset")
    metrics_dataframe = metrics_dataset.to_pandas()

    count_expected_to_match_interval = metrics_dataframe['interval_matches_true_injection'].sum()
    if config.verbose:
        print("the most nearest accuracy", round(most_clothest_counts / count_expected_to_match_interval, 2))
        print(metrics_dataframe['augmentation'].value_counts())

    metrics_log_df_filtered = metrics_dataframe[metrics_dataframe['interval_matches_true_injection']]
    metrics_log_df_filtered_first_hit = metrics_log_df_filtered[metrics_log_df_filtered['hit_i'] == 0]
    metrics_log_df_filtered_first_hit_match = metrics_log_df_filtered_first_hit[metrics_log_df_filtered_first_hit['hit_youtube_id'] == metrics_log_df_filtered_first_hit['youtube_id']]

    matched_videos_found_intervals = metrics_log_df_filtered_first_hit_match['youtube_id'].value_counts()
    if config.verbose:
        print("matched_videos", len(matched_videos_found_intervals))
        print("matched_videos_found_intervals", matched_videos_found_intervals)
        print(f"found_intervals_distribution (at most {config.full_interval_duration_in_seconds / config.interval_step})", matched_videos_found_intervals.value_counts())

    return {
        "accuracy": round(most_clothest_counts / count_expected_to_match_interval, 2),
    }

if __name__ == '__main__':
    config = EvaluationConfig()
    evaluate_matching(config)

import torch
import torch.nn as nn
import os
from qdrant_client import QdrantClient, models

from datasets import Dataset

if __name__ == '__main__':
    base_embeddings_path = 'data/music_caps/audio_embeddings'
    base_augmented_embeddings_path = 'data/music_caps/augmented_embeddings'
    qdrant_collection_name = "audio_embeddings"

    metrics_log_path = 'data/music_caps/metrics'

    os.makedirs(metrics_log_path, exist_ok=True)

    embeddings_files = sorted(os.listdir(base_embeddings_path))
    augmented_embeddings_files = sorted(os.listdir(base_augmented_embeddings_path))

    embeddings_by_youtube_id = {}
    embedding_youtube_ids = []
    for file_name in embeddings_files:
        embedding = torch.load(base_embeddings_path + '/' + file_name)
        youtube_id = file_name.removesuffix('.pt')
        embeddings_by_youtube_id[youtube_id] = embedding
        embedding_youtube_ids.append(youtube_id)

    embedding_size = embedding.shape[1]

    qdrant = QdrantClient(":memory:")
    # Create collection to store books
    qdrant.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=models.Distance.COSINE
        )
    )

    qdrant.upload_points(
        collection_name=qdrant_collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector=embeddings_by_youtube_id[youtube_id][0].numpy(),
                payload={"youtube_id": youtube_id},
            ) for idx, youtube_id in enumerate(embedding_youtube_ids)
        ]
    )

    augmented_embeddings_by_youtube_id = {}

    metrics_log = []
    most_clothest_counts = 0
    for file_name in augmented_embeddings_files:
        embedding = torch.load(base_augmented_embeddings_path + '/' + file_name)
        youtube_id = "_".join(file_name.removesuffix('.pt').split("_")[:-1])
        augmented_embeddings_by_youtube_id[youtube_id] = embedding

        hits = qdrant.search(
            collection_name=qdrant_collection_name,
            query_vector=embedding[0].numpy(),
            limit=10,
        )

        for i, hit in enumerate(hits):
            # print(f"{hit.payload['youtube_id']} == {youtube_id}")
            if hit.payload['youtube_id'] == youtube_id:
                if i == 0:
                    most_clothest_counts += 1
                metrics_log.append({
                    "position": i,
                    "score": hit.score,
                    "youtube_id": youtube_id,
                    "file_name": file_name,
                })
                break

    print("augmented_embeddings_files", len(augmented_embeddings_files))
    print("found clothest:", len(metrics_log))
    print("")
    print("accuracy                 ", round(len(metrics_log) / len(augmented_embeddings_files), 2))
    print("the most nearest accuracy", round(most_clothest_counts / len(augmented_embeddings_files), 2))

    Dataset.from_list(metrics_log).save_to_disk(metrics_log_path)


import os

from qdrant_client import QdrantClient, models
import torch
import numpy as np

class AudioIndex():

    def __init__(self,
                 index_embeddings_dir,
                 collection_name='audio_index',
                 distance_metric=models.Distance.COSINE,
                ):
        
        self.collection_name = collection_name

        embeddings_files = sorted(os.listdir(index_embeddings_dir))

        index_points = []

        idx_point_i = 0
        for file_name in embeddings_files:
            embedding = torch.load(os.path.join(index_embeddings_dir, file_name))
            file_id = file_name.removesuffix('.pt')

            for interval_num in range(embedding.shape[0]):
                index_point = models.PointStruct(
                    id=idx_point_i,
                    vector=embedding[interval_num],
                    payload={"file_id": file_id, "interval_num": interval_num},
                )
                idx_point_i += 1
                index_points.append(index_point)


        self.qdrant = QdrantClient(":memory:")

        embedding_size = embedding.shape[1]
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=embedding_size,
                distance=distance_metric,
            )
        )

        self.qdrant.upload_points(
            collection_name=self.collection_name,
            points=index_points,
        )

        return

    def search(self, query_vector: np.ndarray, limit=10):
        return self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=10,
        )

    def search_sequential(self, query_vectors: np.ndarray, limit_per_vector=10):

        query_hits = []
        for i in range(len(query_vectors)):
            query_vector = query_vectors[i]

            hits = self.search(query_vector, limit=limit_per_vector)
            query_hits.append(hits)

        return query_hits

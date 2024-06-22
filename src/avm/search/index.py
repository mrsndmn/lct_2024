from typing import Optional
from tqdm.auto import tqdm
import os
from typing import List
from qdrant_client import QdrantClient, models
from qdrant_client.conversions import common_types as types
import torch
import numpy as np
import uuid


class EmbeddingIndexFolder():

    def __init__(self,
                 index_embeddings_dir,
                 index_embeddings_files=None,
                 collection_name='audio_index',
                 distance_metric=models.Distance.DOT,
                 embedding_size=256,
                 qdrant_client=None,
                ):
        
        self.collection_name = collection_name

        if qdrant_client is not None:
            self.qdrant = qdrant_client
        else:
            self.qdrant = QdrantClient(":memory:")

        if not self.qdrant.collection_exists(collection_name=self.collection_name):
            print("creating collection")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=distance_metric,
                )
            )

            if index_embeddings_files is not None:
                embeddings_files = index_embeddings_files
            else:
                embeddings_files = sorted(os.listdir(index_embeddings_dir))

            index_points = []

            assert len(embeddings_files) > 0, "index cant be empty"

            idx_point_count = 0
            for file_name in embeddings_files:
                embedding = torch.load(os.path.join(index_embeddings_dir, file_name))
                file_id = file_name.removesuffix('.pt')
                
                idx_point_count += embedding.shape[0]
                self.load_embeddings(embedding, file_id)

            print("idx_point_count", idx_point_count)

        return

    def load_embeddings(self, embedding, file_id):
        index_points = []
        for interval_num in range(embedding.shape[0]):
            index_point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding[interval_num],
                payload={"file_id": file_id, "interval_num": interval_num},
            )
            index_points.append(index_point)

        self.qdrant.upload_points(
            collection_name=self.collection_name,
            points=index_points,
        )

        return

    def scroll(self, scroll_filter):
        return self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            with_vectors=True,
        )

    def search(
        self,
        query_vector: np.ndarray,
        query_filter: Optional[types.Filter] = None,
        limit=1,
    ) -> List[types.ScoredPoint]:

        return self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )

    def search_sequential(self, query_vectors: np.ndarray, limit_per_vector=1) -> List[List[types.ScoredPoint]]:

        query_hits = []
        for i in tqdm(range(len(query_vectors))):
            query_vector = query_vectors[i]

            hits = self.search(query_vector, limit=limit_per_vector)
            query_hits.append(hits)

        return query_hits

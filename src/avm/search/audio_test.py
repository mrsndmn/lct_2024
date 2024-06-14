import pytest
import timeit
import torch
from qdrant_client import QdrantClient, models

@pytest.mark.parametrize("distance_metric", [ models.Distance.DOT, models.Distance.COSINE ])
def test_qdrand_normalized_embeddings_diatance(distance_metric):
    embedding_size = 256
    collection_name="test"
    index_points = []

    for idx_point_i in range(100000):
        embedding = torch.rand([embedding_size])

        index_point = models.PointStruct(
            id=idx_point_i,
            vector=embedding.numpy(),
            payload={"file_id": "", "interval_num": 0},
        )
        idx_point_i += 1
        index_points.append(index_point)


    qdrant = QdrantClient(":memory:")

    qdrant.create_collection(
        collection_name="test",
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=distance_metric,
        )
    )
    # print("start index_points", len(index_points))
    qdrant.upload_points(
        collection_name=collection_name,
        points=index_points,
    )
    # print("index_points", len(index_points))

    query_vector = torch.rand([embedding_size]).numpy()

    def index_lookup():
        qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1,
        )

    # print("start timeit")
    num_repeats = 100
    print(f"metric={distance_metric}\ttime=", timeit.timeit("index_lookup()", globals=locals(), number=num_repeats) / num_repeats)


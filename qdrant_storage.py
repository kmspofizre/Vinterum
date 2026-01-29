from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch, Filter
from typing import List, Optional

class QdrantStorage:
    def __init__(self, qdrant_url: str):
        self.client = QdrantClient(qdrant_url)

    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    def upsert_data(self, collection_name: str, batch: Batch):
        self.client.upsert(collection_name=collection_name, points=batch, wait=True)

    def qdrant_search(
        self,
        collection_name: str,
        vectorized_question: List[float],
        top_k: int,
        query_filter: Optional[Filter] = None,
    ):
        hits = self.client.query_points(
            collection_name=collection_name,
            query=vectorized_question,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        return hits

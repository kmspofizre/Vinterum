from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings
from qdrant_storage import QdrantStorage

def _format_payload_to_doc(payload: dict, score: Optional[float], point_id: Optional[str]) -> Document:
    text = payload.get("page_content") or payload.get("text") or ""
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        meta = {k: v for k, v in payload.items() if k not in ("page_content", "text")}
    meta = dict(meta)
    if score is not None:
        meta["score"] = score
    if point_id is not None:
        meta["point_id"] = point_id
    return Document(page_content=text, metadata=meta)

class QdrantRetriever:
    def __init__(self, settings: Settings, storage: QdrantStorage, embeddings: Embeddings):
        self.settings = settings
        self.storage = storage
        self.embeddings = embeddings

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Document]:
        vec = self.embeddings.embed_query(question)
        resp = self.storage.qdrant_search(
            collection_name=self.settings.collection,
            vectorized_question=vec,
            top_k=top_k or self.settings.top_k,
        )
        points = getattr(resp, "points", resp) 
        docs: List[Document] = []
        for p in points:
            payload = p.payload or {}
            docs.append(_format_payload_to_doc(payload, getattr(p, "score", None), str(getattr(p, "id", "")) or None))
        return docs

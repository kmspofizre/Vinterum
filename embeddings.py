import ollama
from typing import List
from langchain_core.embeddings import Embeddings

class OllamaEmbeddingWrapper(Embeddings):
    def __init__(self, model_name: str = "bge-m3"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = ollama.embed(model=self.model_name, input=texts)
        return res["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        res = ollama.embed(model=self.model_name, input=text)
        return res["embeddings"][0]

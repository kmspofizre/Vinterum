import os
from chunker import Chunker
from langchain_core.embeddings import Embeddings

class MaterialProcessor:
    def __init__(self, materials_path: str, embedding_wrapper: Embeddings) -> None:
        self.materials_path = materials_path
        self.materials_files = []
        self.get_materials_paths()
        self.embedding_wrapper = embedding_wrapper
        self.chunker = Chunker()


    def get_materials_paths(self) -> None:
        for filename in os.listdir(self.materials_path):
            if os.path.isfile(os.path.join(self.materials_path, filename)) and (not filename.startswith(".")):
                self.materials_files.append(os.path.join(self.materials_path, filename))

    def process_materials(self, tenant: str):
        texts, metadatas = [], []
        for filename in self.materials_files:
            parts = self.chunker.chunk(filename)
            for i, part in enumerate(parts.chunks):
                texts.append(part.text)
                metadatas.append({
                    "tenant_id": tenant,
                    "doc_id": filename or "doc",
                    "chunk_id": i,
                })
        return texts, metadatas



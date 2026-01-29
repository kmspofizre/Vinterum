import os
from chonkie import Pipeline, Document


class Chunker:
    def __init__(self, chunk_size: int = 512,
                 context_size: int = 128,
                 embedding_model: str = "bge-m3",):
        self.pipe = (Pipeline()
                .chunk_with("recursive", tokenizer="gpt2", chunk_size=chunk_size, recipe="markdown")
                .refine_with("overlap", context_size=context_size))
        self.embedding_model = os.getenv("EMBED_MODEL", embedding_model)

    def chunk(self, filename: str) -> Document:
        with open(filename, encoding="utf-8") as f:
            data = f.read()
        chunked_doc = self.pipe.run(data)

        return chunked_doc

import sys

from config import Settings
from llm import build_llm
from qdrant_storage import QdrantStorage
from retrieval import QdrantRetriever
from ticketing import TicketClient
from graph import build_graph


from embeddings import OllamaEmbeddingWrapper  # bge-m3 / Ollama

def main():
    settings = Settings()
    llm = build_llm(settings)

    storage = QdrantStorage(settings.qdrant_url)
    emb = OllamaEmbeddingWrapper(model_name="bge-m3")
    retriever = QdrantRetriever(settings, storage, emb)

    tickets = TicketClient()
    app = build_graph(llm, retriever, tickets)

    question = sys.argv[1] if len(sys.argv) > 1 else "Что такое CSA?"
    out = app.invoke({"question": question, "messages": []})
    print(out["answer"])

if __name__ == "__main__":
    main()

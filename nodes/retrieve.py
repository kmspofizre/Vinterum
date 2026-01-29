from state import ChatState
from retrieval import QdrantRetriever

def node_retrieve(retriever: QdrantRetriever):
    def _run(state: ChatState) -> ChatState:
        docs = retriever.retrieve(state["question"])
        return {"docs": docs}
    return _run

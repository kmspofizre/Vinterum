from typing import List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.documents import Document


class ChatState(TypedDict, total=False):
    question: str
    needs_retrieval: Optional[bool]
    docs: List[Document]
    answer: str
    grounded: Optional[bool]
    relevant: Optional[bool]
    judge_reason: str
    ticket_id: Optional[str]
    messages: List[AnyMessage]

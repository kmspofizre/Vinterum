from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from state import ChatState
from prompts import rag_prompt, direct_prompt

def _format_context(docs, limit_chars: int = 8000) -> str:
    parts = []
    total = 0
    for i, d in enumerate(docs or []):
        meta = d.metadata or {}
        chunk = f"[doc#{i} meta={meta}]\n{d.page_content}\n"
        if total + len(chunk) > limit_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)

def node_generate_rag(llm: BaseChatModel):
    prompt = rag_prompt()
    chain = prompt | llm

    def _run(state: ChatState) -> ChatState:
        context = _format_context(state.get("docs", []))
        msg = chain.invoke({"question": state["question"], "context": context})
        answer = msg.content

        messages = list(state.get("messages", []))
        if not messages:
            messages.append(HumanMessage(content=state["question"]))
        messages.append(AIMessage(content=answer))

        return {"answer": answer, "messages": messages}

    return _run

def node_generate_direct(llm: BaseChatModel):
    prompt = direct_prompt()
    chain = prompt | llm

    def _run(state: ChatState) -> ChatState:
        msg = chain.invoke({"question": state["question"]})
        answer = msg.content

        messages = list(state.get("messages", []))
        if not messages:
            messages.append(HumanMessage(content=state["question"]))
        messages.append(AIMessage(content=answer))

        return {"answer": answer, "messages": messages}

    return _run

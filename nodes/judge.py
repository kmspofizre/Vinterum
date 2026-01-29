from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from state import ChatState
from schemas import JudgeDecision
from prompts import judge_prompt

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

def node_judge(llm: BaseChatModel):
    parser = JsonOutputParser(pydantic_object=JudgeDecision)
    prompt = judge_prompt()
    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

    def _run(state: ChatState) -> ChatState:
        context = _format_context(state.get("docs", []))
        out = chain.invoke({"question": state["question"], "context": context, "answer": state["answer"]})
        verdict = JudgeDecision(**out)
        return {"grounded": verdict.grounded, "relevant": verdict.relevant, "judge_reason": verdict.reason}

    return _run

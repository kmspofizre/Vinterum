from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from state import ChatState
from schemas import RouteDecision
from prompts import router_prompt

def node_route(llm: BaseChatModel):
    parser = JsonOutputParser(pydantic_object=RouteDecision)
    prompt = router_prompt()

    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

    def _run(state: ChatState) -> ChatState:
        out = chain.invoke({"question": state["question"]})
        decision = RouteDecision(**out)
        return {"needs_retrieval": decision.needs_retrieval}

    return _run

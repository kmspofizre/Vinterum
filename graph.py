from typing import Literal
from langgraph.graph import StateGraph, START, END

from state import ChatState

from nodes.router import node_route
from nodes.retrieve import node_retrieve
from nodes.generate import node_generate_rag, node_generate_direct
from nodes.judge import node_judge
from nodes.ticket import node_ticket
from nodes.respond import node_respond

def build_graph(llm, retriever, tickets):
    g = StateGraph(ChatState)

    g.add_node("route", node_route(llm))
    g.add_node("retrieve", node_retrieve(retriever))
    g.add_node("gen_rag", node_generate_rag(llm))
    g.add_node("gen_direct", node_generate_direct(llm))
    g.add_node("judge", node_judge(llm))
    g.add_node("ticket", node_ticket(tickets))
    g.add_node("respond", node_respond())

    def route_next(state: ChatState) -> Literal["retrieve", "gen_direct"]:
        return "retrieve" if state.get("needs_retrieval") else "gen_direct"

    def judge_next(state: ChatState) -> Literal["respond", "ticket"]:
        ok = bool(state.get("grounded")) and bool(state.get("relevant"))
        return "respond" if ok else "ticket"

    g.add_edge(START, "route")
    g.add_conditional_edges("route", route_next, {"retrieve": "retrieve", "gen_direct": "gen_direct"})

    g.add_edge("retrieve", "gen_rag")
    g.add_edge("gen_rag", "judge")
    g.add_edge("gen_direct", "judge")

    g.add_conditional_edges("judge", judge_next, {"respond": "respond", "ticket": "ticket"})
    g.add_edge("ticket", "respond")
    g.add_edge("respond", END)

    return g.compile()

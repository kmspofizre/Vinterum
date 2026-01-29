from state import ChatState
from ticketing import TicketClient

def node_ticket(tickets: TicketClient):
    def _run(state: ChatState) -> ChatState:
        body = (
            f"Question:\n{state['question']}\n\n"
            f"Answer:\n{state.get('answer','')}\n\n"
            f"Judge:\n{state.get('judge_reason','')}\n\n"
            f"Grounded={state.get('grounded')} Relevant={state.get('relevant')}\n"
        )
        ticket_id = tickets.create_ticket(title="RAG answer failed validation", body=body)
        return {"ticket_id": ticket_id}
    return _run

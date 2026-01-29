from state import ChatState

def node_respond():
    def _run(state: ChatState) -> ChatState:
        if state.get("ticket_id"):
            final = (
                f"Я не уверен в корректности ответа и создал тикет: {state['ticket_id']}.\n"
                f"Причина: {state.get('judge_reason','')}\n\n"
                f"Черновой ответ:\n{state.get('answer','')}"
            )
            return {"answer": final}
        return {"answer": state["answer"]}
    return _run

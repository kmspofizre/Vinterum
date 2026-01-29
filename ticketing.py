from dataclasses import dataclass
from typing import Optional

@dataclass
class TicketClient:
    def create_ticket(self, title: str, body: str) -> str:
        # TODO: HTTP вызов в твою тикет-систему
        return "TCK-0001"

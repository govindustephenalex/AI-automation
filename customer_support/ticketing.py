from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from customer_support.storage import JsonStore


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Ticketing:
    store: JsonStore

    def create_ticket(
        self,
        *,
        customer_id: str,
        subject: str,
        description: str,
        priority: str = "normal",
        intent: str = "general_question",
        sentiment: str = "NEUTRAL",
        entities: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        tid = f"tkt_{uuid.uuid4().hex[:10]}"
        ticket = {
            "ticket_id": tid,
            "customer_id": str(customer_id),
            "subject": subject.strip(),
            "description": description.strip(),
            "priority": priority,
            "intent": intent,
            "sentiment": sentiment,
            "entities": entities or [],
            "status": "open",
            "created_at": utcnow_iso(),
            "events": [{"at": utcnow_iso(), "type": "created"}],
        }
        self.store.put("tickets", tid, ticket)
        return ticket

    def add_event(self, ticket_id: str, event_type: str, note: str = "") -> Dict:
        ticket = self.get_ticket(ticket_id)
        ticket["events"].append({"at": utcnow_iso(), "type": event_type, "note": note})
        self.store.put("tickets", ticket_id, ticket)
        return ticket

    def set_status(self, ticket_id: str, status: str) -> Dict:
        ticket = self.get_ticket(ticket_id)
        ticket["status"] = status
        return self.add_event(ticket_id, "status_changed", f"status={status}")

    def get_ticket(self, ticket_id: str) -> Dict:
        t = self.store.get("tickets", ticket_id)
        if not t:
            raise KeyError(f"Unknown ticket_id: {ticket_id}")
        return t

    def list_recent(self, limit: int = 10) -> List[Dict]:
        all_t = list(self.store.list_bucket("tickets").values())
        all_t.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return all_t[: int(limit)]

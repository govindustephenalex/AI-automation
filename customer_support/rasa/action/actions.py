from __future__ import annotations

import uuid
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionCreateTicket(Action):
    def name(self) -> Text:
        return "action_create_ticket"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Placeholder: In production, call your ticketing service here.
        tid = f"tkt_{uuid.uuid4().hex[:10]}"
        dispatcher.utter_message(text=f"I created a support ticket: {tid}. Our team will follow up soon.")
        return []

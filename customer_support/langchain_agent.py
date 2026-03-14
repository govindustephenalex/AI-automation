r_support/langchain_agent.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from customer_support.faq import DEFAULT_FAQ, build_faq_index, search_faq
from customer_support.intent_sentiment import nlu
from customer_support.nlp_signals import build_signals
from customer_support.storage import JsonStore
from customer_support.ticketing import Ticketing


def _make_llm(model: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    Uses OpenAI via langchain-openai when OPENAI_API_KEY is present.
    Falls back to None (rule-based) if not configured.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        common = {"model": model, "temperature": temperature}
        try:
            return ChatOpenAI(**common, api_key=api_key, base_url=base_url)
        except TypeError:
            try:
                return ChatOpenAI(**common, openai_api_key=api_key, openai_api_base=base_url)
            except TypeError:
                return ChatOpenAI(**common)
    except Exception:
        return None


@dataclass
class SupportAgent:
    store: JsonStore
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self.ticketing = Ticketing(self.store)
        self.faq_index = build_faq_index(DEFAULT_FAQ)
        self.llm = _make_llm(self.llm_model, self.temperature)

        self._tools = self._build_tools()

        # Build a LangGraph ReAct agent if available; else use a deterministic router.
        self.graph = None
        try:
            from langgraph.prebuilt import create_react_agent

            if self.llm is not None:
                self.graph = create_react_agent(
                    self.llm,
                    tools=self._tools,
                    state_modifier=SystemMessage(
                        content=(
                            "You are a customer support assistant for a ride-hailing app. "
                            "Use tools to create tickets and fetch ticket status. "
                            "Be concise, ask only necessary follow-ups, and summarize actions taken."
                        )
                    ),
                )
        except Exception:
            self.graph = None

    def _build_tools(self):
        ticketing = self.ticketing
        faq_index = self.faq_index

        @tool("search_faq")
        def tool_search_faq(query: str, k: int = 3) -> str:
            """Search the FAQ and return top matches."""
            hits = search_faq(faq_index, query, k=int(k))
            return json.dumps(
                [{"key": key, "score": score, "answer": ans} for key, score, ans in hits],
                indent=2,
            )

        @tool("create_ticket")
        def tool_create_ticket(customer_id: str, subject: str, description: str, priority: str = "normal") -> str:
            """Create a support ticket."""
            sig = build_signals(description)
            n = nlu(description)
            t = ticketing.create_ticket(
                customer_id=customer_id,
                subject=subject,
                description=description,
                priority=priority,
                intent=n.intent,
                sentiment=n.sentiment,
                entities=sig.entities,
            )
            return json.dumps(t, indent=2)

        @tool("get_ticket")
        def tool_get_ticket(ticket_id: str) -> str:
            """Fetch a ticket by id."""
            t = ticketing.get_ticket(ticket_id)
            return json.dumps(t, indent=2)

        @tool("set_ticket_status")
        def tool_set_ticket_status(ticket_id: str, status: str) -> str:
            """Update ticket status (open|pending|resolved|closed)."""
            t = ticketing.set_status(ticket_id, status)
            return json.dumps(t, indent=2)

        return [tool_search_faq, tool_create_ticket, tool_get_ticket, tool_set_ticket_status]

    def respond(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Returns an updated message list with assistant response appended.
        """
        if self.graph is not None:
            out = self.graph.invoke({"messages": messages})
            return out["messages"]

        # Deterministic fallback: use NLU + FAQ + ticket creation suggestion.
        user_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_text = str(m.content)
                break
        n = nlu(user_text)
        hits = search_faq(self.faq_index, user_text, k=1)
        faq_answer = hits[0][2] if hits and hits[0][1] > 0.15 else None

        if n.intent in {"refund", "payment_issue", "driver_rider_safety", "account_access", "trip_issue"}:
            msg = (
                f"I can help with that. Intent: {n.intent} (sentiment: {n.sentiment}).\n"
                "If you want, I can create a ticket. Reply with:\n"
                "`customer_id: <id>` and a short subject."
            )
        elif faq_answer:
            msg = faq_answer
        else:
            msg = "Tell me what went wrong (trip id, time, amount, payment method). I can create a ticket and track it."

        return [*messages, AIMessage(content=msg)]

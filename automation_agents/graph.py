from __future__ import annotations

from typing import List, Literal, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from automation_agents.llm import make_llm
from automation_agents.prompts import (
    ANALYTICS_PROMPT,
    FRAUD_PROMPT,
    GUARDIANS_PROMPT,
    PRICING_PROMPT,
    ROUTING_PROMPT,
    SUPPORT_PROMPT,
    TRIAGE_PROMPT,
)
from automation_agents.tools import make_tools


Route = Literal["guardians", "support", "analytics", "pricing", "routing", "fraud"]


class AutomationState(TypedDict):
    messages: List[BaseMessage]
    route: Route


def build_automation_graph(model: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    Multi-domain automation router + ReAct agents.
    Requires OPENAI_API_KEY for LLM behavior; tools still exist but agent won't run without a model.
    """
    llm = make_llm(model=model, temperature=temperature)
    triage_llm = make_llm(model=model, temperature=0.0)
    tools = make_tools()

    try:
        from langgraph.prebuilt import create_react_agent
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("This requires langgraph.prebuilt.create_react_agent") from exc

    def _agent(system_prompt: str, toolset):
        if llm is None:
            return None
        return create_react_agent(llm, tools=toolset, state_modifier=SystemMessage(content=system_prompt))

    guardians_agent = _agent(GUARDIANS_PROMPT, tools["guardians"])
    support_agent = _agent(SUPPORT_PROMPT, tools["support"])
    analytics_agent = _agent(ANALYTICS_PROMPT, tools["analytics"])
    pricing_agent = _agent(PRICING_PROMPT, tools["pricing"])
    routing_agent = _agent(ROUTING_PROMPT, tools["routing"])
    fraud_agent = _agent(FRAUD_PROMPT, tools["fraud"])

    def triage_node(state: AutomationState) -> AutomationState:
        if triage_llm is None:
            # No LLM configured; default route.
            return {**state, "route": "support"}
        msgs = [SystemMessage(content=TRIAGE_PROMPT), *state["messages"]]
        resp = str(triage_llm.invoke(msgs).content).strip().lower()
        if resp not in {"guardians", "support", "analytics", "pricing", "routing", "fraud"}:
            resp = "support"
        return {**state, "route": resp}  # type: ignore[return-value]

    def route_edge(state: AutomationState):
        return state["route"]

    def _invoke(agent, state: AutomationState) -> AutomationState:
        if agent is None:
            # Minimal fallback (no LLM): return a guidance message.
            from langchain_core.messages import AIMessage

            msg = (
                "OPENAI_API_KEY is not set, so the automation agent cannot reason and call tools.\n"
                "Set OPENAI_API_KEY, or call the underlying CLIs directly (guardians_ai, fraud_detection, etc.)."
            )
            return {**state, "messages": [*state["messages"], AIMessage(content=msg)]}
        out = agent.invoke({"messages": state["messages"]})
        return {**state, "messages": out["messages"]}

    def guardians_node(state: AutomationState) -> AutomationState:
        return _invoke(guardians_agent, state)

    def support_node(state: AutomationState) -> AutomationState:
        return _invoke(support_agent, state)

    def analytics_node(state: AutomationState) -> AutomationState:
        return _invoke(analytics_agent, state)

    def pricing_node(state: AutomationState) -> AutomationState:
        return _invoke(pricing_agent, state)

    def routing_node(state: AutomationState) -> AutomationState:
        return _invoke(routing_agent, state)

    def fraud_node(state: AutomationState) -> AutomationState:
        return _invoke(fraud_agent, state)

    g = StateGraph(AutomationState)
    g.add_node("triage", triage_node)
    g.add_node("guardians", guardians_node)
    g.add_node("support", support_node)
    g.add_node("analytics", analytics_node)
    g.add_node("pricing", pricing_node)
    g.add_node("routing", routing_node)
    g.add_node("fraud", fraud_node)
    g.set_entry_point("triage")
    g.add_conditional_edges(
        "triage",
        route_edge,
        {
            "guardians": "guardians",
            "support": "support",
            "analytics": "analytics",
            "pricing": "pricing",
            "routing": "routing",
            "fraud": "fraud",
        },
    )
    g.add_edge("guardians", END)
    g.add_edge("support", END)
    g.add_edge("analytics", END)
    g.add_edge("pricing", END)
    g.add_edge("routing", END)
    g.add_edge("fraud", END)
    return g.compile(checkpointer=MemorySaver())

from __future__ import annotations

"""
AutoGPT-style autonomous loop using LangChain tools.

This is not the upstream AutoGPT project; it's a lightweight "AutoGPT-like" runner:
- creates a plan
- executes tool calls
- iterates with a step budget
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from automation_agents.llm import make_llm
from automation_agents.tools import make_tools


@dataclass(frozen=True)
class AutoLoopConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_steps: int = 8


def run_autogpt_like(goal: str, cfg: Optional[AutoLoopConfig] = None) -> str:
    cfg = cfg or AutoLoopConfig()
    llm = make_llm(model=cfg.model, temperature=cfg.temperature)
    if llm is None:
        raise RuntimeError("Set OPENAI_API_KEY to run the autonomous loop.")

    # Prefer LangGraph ReAct so tool calling is reliable.
    try:
        from langgraph.prebuilt import create_react_agent
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("This requires langgraph.prebuilt.create_react_agent") from exc

    tools = make_tools()["all"]
    agent = create_react_agent(
        llm,
        tools=tools,
        state_modifier=SystemMessage(
            content=(
                "You are an automation agent. Produce a short plan, then execute it by calling tools. "
                "Stop when the goal is satisfied. Keep outputs concise."
            )
        ),
    )

    messages = [HumanMessage(content=f"Goal: {goal}\nConstraints: max_steps={cfg.max_steps}")]
    out = agent.invoke({"messages": messages}, config={"recursion_limit": int(cfg.max_steps)})
    final = out["messages"][-1]
    return str(getattr(final, "content", final))


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m automation_agents.autogpt_like \"<goal>\"")
    print(run_autogpt_like(" ".join(sys.argv[1:])))

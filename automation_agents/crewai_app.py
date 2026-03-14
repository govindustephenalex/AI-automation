from __future__ import annotations

"""
Optional CrewAI integration.

This file is safe to import/run even if crewai is not installed; it will fail with a clear message.
"""

import os
from typing import Any, Dict, Optional

from automation_agents.tools import make_tools


def run_crewai(goal: str) -> str:
    try:
        from crewai import Agent, Crew, Task
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CrewAI is not installed. Install `crewai` to use this integration.") from exc

    from automation_agents.llm import make_llm

    llm = make_llm(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    if llm is None:
        raise RuntimeError("Set OPENAI_API_KEY to run CrewAI agents.")

    tools = make_tools()

    guardians = Agent(role="Guardians", goal="Handle corporate/legal tasks", backstory="Public policy + compliance", llm=llm, tools=tools["guardians"])
    support = Agent(role="Support", goal="Resolve customer support issues", backstory="Tickets + FAQ", llm=llm, tools=tools["support"])
    analytics = Agent(role="Analytics", goal="Create data reports", backstory="Pandas/sklearn reporting", llm=llm, tools=tools["analytics"])
    pricing = Agent(role="Pricing", goal="Predict demand and recommend prices", backstory="Pricing science", llm=llm, tools=tools["pricing"])
    routing = Agent(role="Routing", goal="Optimize routes", backstory="VRP/TSP/shortest path", llm=llm, tools=tools["routing"])
    fraud = Agent(role="Fraud", goal="Detect fraud", backstory="Anomaly detection and classification", llm=llm, tools=tools["fraud"])

    task = Task(
        description=goal,
        expected_output="A concise, actionable result and any tool outputs used.",
        agent=support,
    )

    crew = Crew(agents=[support, guardians, analytics, pricing, routing, fraud], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m automation_agents.crewai_app \"<goal>\"")
    print(run_crewai(" ".join(sys.argv[1:])))

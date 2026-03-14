"""
AI Agents for Automation (LangChain + LangGraph + optional CrewAI / AutoGPT-style loop).

Entry points:
- automation_agents/cli.py (interactive)
- automation_agents/crewai_app.py (optional)
- automation_agents/autogpt_like.py (optional)
"""

from .graph import build_automation_graph

__all__ = ["build_automation_graph"]

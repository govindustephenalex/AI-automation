from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from automation_agents.graph import build_automation_graph


HELP = """Commands:
  /help     Show help
  /exit     Quit
  /reset    Reset conversation
"""


def main() -> int:
    app = build_automation_graph()
    thread_id = "automation_cli"
    state = {"messages": [SystemMessage(content="You are an automation assistant.")], "route": "support"}

    print("Automation Agents (LangGraph). Type /help.\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            return 0

        if not raw:
            continue
        if raw.lower() in {"/exit", "exit", "quit"}:
            print("Bot: Goodbye!")
            return 0
        if raw.lower() in {"/help", "help"}:
            print(HELP)
            continue
        if raw.lower() == "/reset":
            thread_id = f"automation_cli_{os.urandom(4).hex()}"
            state = {"messages": [SystemMessage(content="You are an automation assistant.")], "route": "support"}
            print("Bot: Reset.\n")
            continue

        state["messages"].append(HumanMessage(content=raw))
        out = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
        state = out
        msg = state["messages"][-1]
        print(f"Bot ({state['route']}): {getattr(msg, 'content', str(msg))}\n")


if __name__ == "__main__":
    raise SystemExit(main())

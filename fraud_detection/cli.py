from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from customer_support.langchain_agent import SupportAgent
from customer_support.storage import JsonStore
from customer_support.voice import record_wav, speak, transcribe_wav


HELP = """Commands:
  /help         Show help
  /exit         Quit
  /reset        Reset conversation
  /voice [sec]  Record mic audio and transcribe
  /tickets      List recent tickets
"""


def main() -> int:
    # Store tickets/faq in a local JSON file under data/
    store = JsonStore(os.path.join("data", "support_store.json"))
    agent = SupportAgent(store)

    thread = [SystemMessage(content="You are a customer support assistant.")]

    print("Customer Support Bot. Type /help.\n")

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
            thread = [SystemMessage(content="You are a customer support assistant.")]
            print("Bot: Reset.\n")
            continue
        if raw.lower() == "/tickets":
            tickets = agent.ticketing.list_recent(limit=10)
            if not tickets:
                print("Bot: No tickets yet.\n")
                continue
            for t in tickets:
                print(f"- {t['ticket_id']} [{t['status']}] {t['subject']} (customer={t['customer_id']})")
            print("")
            continue

        if raw.lower().startswith("/voice"):
            parts = raw.split()
            sec = 6.0
            if len(parts) >= 2:
                try:
                    sec = float(parts[1])
                except ValueError:
                    print("Bot: Usage: /voice [sec]\n")
                    continue
            try:
                print(f"Bot: Listening for {sec:g}s...\n")
                wav = record_wav(seconds=sec)
                try:
                    text = transcribe_wav(wav)
                finally:
                    try:
                        os.remove(wav)
                    except OSError:
                        pass
            except Exception as exc:
                print(f"Bot: Voice failed: {exc}\n")
                continue

            print(f"You (voice): {text}\n")
            thread.append(HumanMessage(content=text))
        else:
            thread.append(HumanMessage(content=raw))

        thread = agent.respond(thread)
        msg = thread[-1]
        out = getattr(msg, "content", str(msg))
        print(f"Bot: {out}\n")
        speak(str(out))


if __name__ == "__main__":
    raise SystemExit(main())

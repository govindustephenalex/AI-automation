from __future__ import annotations

import os
from typing import Optional


def make_llm(model: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    Create a LangChain Chat model if OPENAI_API_KEY is set.
    Uses a compatibility shim for langchain-openai kwarg variants.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    common = {"model": model, "temperature": temperature}

    try:
        return ChatOpenAI(**common, api_key=api_key, base_url=base_url)
    except TypeError:
        pass
    try:
        return ChatOpenAI(**common, openai_api_key=api_key, openai_api_base=base_url)
    except TypeError:
        pass
    return ChatOpenAI(**common)

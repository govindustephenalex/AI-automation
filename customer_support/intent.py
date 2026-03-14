_support/intent_sentiment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


Intent = Literal[
    "refund",
    "payment_issue",
    "driver_rider_safety",
    "account_access",
    "trip_issue",
    "pricing_fare",
    "technical_bug",
    "general_question",
]


@dataclass(frozen=True)
class NluResult:
    intent: Intent
    intent_score: float
    sentiment: str
    sentiment_score: float


DEFAULT_LABELS: List[Intent] = [
    "refund",
    "payment_issue",
    "driver_rider_safety",
    "account_access",
    "trip_issue",
    "pricing_fare",
    "technical_bug",
    "general_question",
]


def infer_sentiment(text: str) -> Tuple[str, float]:
    """
    Transformers sentiment pipeline when available; falls back to neutral.
    """
    try:
        from transformers import pipeline

        clf = pipeline("sentiment-analysis")
        r = clf(text)[0]
        label = str(r.get("label", "NEUTRAL"))
        score = float(r.get("score", 0.0))
        return label, score
    except Exception:
        return "NEUTRAL", 0.0


def infer_intent_zero_shot(text: str, labels: Optional[List[Intent]] = None) -> Tuple[Intent, float]:
    """
    Zero-shot classification with Transformers. If unavailable, uses a small rule fallback.
    """
    labels = labels or DEFAULT_LABELS
    try:
        from transformers import pipeline

        z = pipeline("zero-shot-classification")
        r = z(text, candidate_labels=list(labels))
        best_label = str(r["labels"][0])
        best_score = float(r["scores"][0])
        return best_label, best_score  # type: ignore[return-value]
    except Exception:
        t = text.lower()
        if "refund" in t or "chargeback" in t:
            return "refund", 0.55
        if "otp" in t or "login" in t or "password" in t:
            return "account_access", 0.55
        if "crash" in t or "bug" in t or "app" in t:
            return "technical_bug", 0.5
        if "payment" in t or "card" in t or "upi" in t:
            return "payment_issue", 0.5
        if "driver" in t or "accident" in t or "unsafe" in t:
            return "driver_rider_safety", 0.5
        if "fare" in t or "price" in t or "surge" in t:
            return "pricing_fare", 0.5
        if "trip" in t or "ride" in t:
            return "trip_issue", 0.5
        return "general_question", 0.4


def nlu(text: str) -> NluResult:
    intent, intent_score = infer_intent_zero_shot(text)
    sent, sent_score = infer_sentiment(text)
    return NluResult(intent=intent, intent_score=float(intent_score), sentiment=sent, sentiment_score=float(sent_score))

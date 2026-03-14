from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_FAQ: Dict[str, str] = {
    "refund_policy": "Refunds are issued for eligible overcharges or canceled trips based on policy and review.",
    "otp_login": "If you are not receiving OTP, confirm network/SMS permissions and retry after 60 seconds.",
    "payment_failed": "If payment failed, try a different method and check bank/UPI status; do not retry too rapidly.",
    "lost_item": "Report a lost item with trip details; support will contact the driver where permitted.",
}


@dataclass
class FaqIndex:
    keys: List[str]
    answers: List[str]
    X: any
    vectorizer: any


def build_faq_index(faq: Dict[str, str]) -> FaqIndex:
    """
    Very small TF-IDF FAQ retriever (sklearn). Keeps the support bot deterministic.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    keys = list(faq.keys())
    answers = [faq[k] for k in keys]
    corpus = [k.replace("_", " ") + " " + faq[k] for k in keys]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(corpus)
    return FaqIndex(keys=keys, answers=answers, X=X, vectorizer=vec)


def search_faq(index: FaqIndex, query: str, k: int = 3) -> List[Tuple[str, float, str]]:
    q = index.vectorizer.transform([query])
    scores = (index.X @ q.T).toarray().reshape(-1)
    best = np.argsort(-scores)[: int(k)]
    out: List[Tuple[str, float, str]] = []
    for i in best:
        out.append((index.keys[int(i)], float(scores[int(i)]), index.answers[int(i)]))
    return out

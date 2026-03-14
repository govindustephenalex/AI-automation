from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentConfig:
    max_features: int = 30_000
    ngram_range: tuple[int, int] = (1, 2)


class ContentRecommender:
    """
    Content-based recommender using TF-IDF on item metadata.
    Useful as a cold-start fallback.
    """

    def __init__(self, cfg: Optional[ContentConfig] = None):
        self.cfg = cfg or ContentConfig()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_ids: List[str] = []
        self.item_to_idx: Dict[str, int] = {}
        self.X = None

    def fit(self, items: pd.DataFrame) -> "ContentRecommender":
        text = (
            items[["title", "category", "description"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )
        self.item_ids = items["item_id"].astype(str).tolist()
        self.item_to_idx = {it: i for i, it in enumerate(self.item_ids)}
        vec = TfidfVectorizer(max_features=int(self.cfg.max_features), ngram_range=self.cfg.ngram_range)
        X = vec.fit_transform(text)
        self.vectorizer = vec
        self.X = X
        return self

    def recommend_for_user(
        self,
        *,
        interactions: pd.DataFrame,
        customer_id: str,
        k: int = 10,
        min_history: int = 1,
    ) -> List[Tuple[str, float]]:
        if self.X is None:
            raise RuntimeError("ContentRecommender not fit.")

        cid = str(customer_id)
        hist = interactions[interactions["customer_id"].astype(str) == cid]
        hist_items = [str(x) for x in hist["item_id"].tolist()]
        hist_items = [it for it in hist_items if it in self.item_to_idx]

        if len(hist_items) < int(min_history):
            return []

        idxs = [self.item_to_idx[it] for it in hist_items]
        profile = self.X[idxs].mean(axis=0)
        sims = cosine_similarity(profile, self.X).reshape(-1)

        seen = set(hist_items)
        candidates = []
        for i, score in enumerate(sims):
            it = self.item_ids[int(i)]
            if it in seen:
                continue
            candidates.append((it, float(score)))
        candidates.sort(key=lambda t: t[1], reverse=True)
        return candidates[: int(k)]

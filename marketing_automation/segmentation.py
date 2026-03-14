from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Segmenter:
    n_clusters: int = 5
    random_state: int = 42
    pipeline: Pipeline | None = None

    def fit(self, features: pd.DataFrame) -> "Segmenter":
        X = features.drop(columns=["customer_id"])
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=int(self.n_clusters), random_state=int(self.random_state), n_init="auto")),
            ]
        )
        pipe.fit(X)
        self.pipeline = pipe
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("Segmenter not fit.")
        X = features.drop(columns=["customer_id"])
        labels = self.pipeline.predict(X)
        return pd.Series(labels, index=features.index, name="segment")

    def segment_summary(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        out = features.copy()
        out["segment"] = labels.values
        agg = out.groupby("segment").agg(
            customers=("customer_id", "nunique"),
            recency_days_mean=("recency_days", "mean"),
            frequency_90d_mean=("frequency_90d", "mean"),
            monetary_90d_mean=("monetary_90d", "mean"),
        )
        return agg.reset_index()


def label_segments(features: pd.DataFrame, labels: pd.Series) -> Dict[int, str]:
    """
    Assign human-friendly names based on RFM profiles.
    """
    df = features.copy()
    df["segment"] = labels.values
    g = df.groupby("segment").agg(
        recency=("recency_days", "mean"),
        freq=("frequency_90d", "mean"),
        money=("monetary_90d", "mean"),
    )
    mapping: Dict[int, str] = {}
    for seg, r in g.iterrows():
        if r["money"] > g["money"].quantile(0.75) and r["recency"] < g["recency"].quantile(0.25):
            mapping[int(seg)] = "VIP_Active"
        elif r["freq"] > g["freq"].quantile(0.75) and r["recency"] < g["recency"].quantile(0.5):
            mapping[int(seg)] = "Frequent_Active"
        elif r["recency"] > g["recency"].quantile(0.75) and r["freq"] <= g["freq"].quantile(0.5):
            mapping[int(seg)] = "Churn_Risk"
        elif r["recency"] < g["recency"].quantile(0.5) and r["freq"] <= g["freq"].quantile(0.5):
            mapping[int(seg)] = "New_or_Low_Engagement"
        else:
            mapping[int(seg)] = "Core"
    return mapping

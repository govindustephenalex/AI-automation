from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def _parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_events_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"customer_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events.csv missing columns: {sorted(missing)}")
    if "value" not in df.columns:
        df["value"] = 0.0
    df["timestamp"] = _parse_ts(df["timestamp"])
    df["customer_id"] = df["customer_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["timestamp"])
    return df


def load_interactions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"customer_id", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions.csv missing columns: {sorted(missing)}")
    if "value" not in df.columns:
        df["value"] = 1.0
    if "timestamp" in df.columns:
        df["timestamp"] = _parse_ts(df["timestamp"])
    df["customer_id"] = df["customer_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(1.0)
    return df


def load_items_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "item_id" not in df.columns:
        raise ValueError("items.csv missing column: item_id")
    df["item_id"] = df["item_id"].astype(str)
    for c in ["title", "category", "description"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    return df


def build_rfm_features(events: pd.DataFrame, *, as_of: Optional[datetime] = None) -> pd.DataFrame:
    """
    RFM features for segmentation:
    - recency_days: days since last event (lower is better)
    - frequency_90d: number of events in last 90 days
    - monetary_90d: sum(value) in last 90 days
    """
    if events.empty:
        raise ValueError("events is empty")

    as_of_dt = as_of or datetime.now(timezone.utc)
    cutoff_90d = as_of_dt - pd.Timedelta(days=90)

    last_ts = events.groupby("customer_id")["timestamp"].max()
    recency_days = (as_of_dt - last_ts).dt.total_seconds() / 86400.0

    recent = events[events["timestamp"] >= cutoff_90d]
    freq_90d = recent.groupby("customer_id")["timestamp"].size().astype(float)
    monetary_90d = recent.groupby("customer_id")["value"].sum().astype(float)

    feat = pd.DataFrame(
        {
            "customer_id": last_ts.index.astype(str),
            "recency_days": recency_days.values.astype(float),
        }
    )
    feat = feat.merge(freq_90d.rename("frequency_90d"), left_on="customer_id", right_index=True, how="left")
    feat = feat.merge(monetary_90d.rename("monetary_90d"), left_on="customer_id", right_index=True, how="left")
    feat["frequency_90d"] = feat["frequency_90d"].fillna(0.0)
    feat["monetary_90d"] = feat["monetary_90d"].fillna(0.0)
    return feat

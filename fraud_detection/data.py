from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class DataSpec:
    label_col: Optional[str] = None
    id_col: Optional[str] = "transaction_id"
    timestamp_col: Optional[str] = "timestamp"


def load_transactions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


def split_xy(df: pd.DataFrame, spec: DataSpec) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    y = None
    X = df.copy()
    if spec.label_col:
        if spec.label_col not in df.columns:
            raise KeyError(f"Label column not found: {spec.label_col}")
        y = pd.to_numeric(df[spec.label_col], errors="coerce").fillna(0).astype(int)
        X = X.drop(columns=[spec.label_col])
    return X, y


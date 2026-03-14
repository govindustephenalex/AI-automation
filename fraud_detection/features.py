from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    if timestamp_col not in out.columns:
        return out
    ts = pd.to_datetime(out[timestamp_col], errors="coerce", utc=True)
    out[f"{timestamp_col}_hour"] = ts.dt.hour
    out[f"{timestamp_col}_dow"] = ts.dt.dayofweek
    out[f"{timestamp_col}_is_weekend"] = (ts.dt.dayofweek >= 5).astype("Int64")
    return out


def add_amount_features(df: pd.DataFrame, amount_col: str = "amount") -> pd.DataFrame:
    out = df.copy()
    if amount_col not in out.columns:
        return out
    amt = pd.to_numeric(out[amount_col], errors="coerce")
    out[f"{amount_col}_log"] = np.log(np.clip(amt.astype(float), 1e-8, None))
    return out


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols: List[str] = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols: List[str] = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

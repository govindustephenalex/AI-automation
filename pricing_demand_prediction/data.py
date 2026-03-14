from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    price_col: str
    date_col: Optional[str] = None


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    out[f"{date_col}_year"] = dt.dt.year
    out[f"{date_col}_month"] = dt.dt.month
    out[f"{date_col}_day"] = dt.dt.day
    out[f"{date_col}_dow"] = dt.dt.dayofweek
    out[f"{date_col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("int64")
    return out


def split_xy(df: pd.DataFrame, spec: FeatureSpec) -> Tuple[pd.DataFrame, np.ndarray]:
    if spec.target not in df.columns:
        raise KeyError(f"Target column not found: {spec.target}")
    if spec.price_col not in df.columns:
        raise KeyError(f"Price column not found: {spec.price_col}")

    work = df.copy()
    if spec.date_col:
        if spec.date_col not in work.columns:
            raise KeyError(f"Date column not found: {spec.date_col}")
        work = add_calendar_features(work, spec.date_col)

    y = work[spec.target].to_numpy(dtype=float)

    # Keep price as a feature, but also add log(price) for stability.
    work[f"{spec.price_col}_log"] = np.log(np.clip(work[spec.price_col].astype(float), 1e-8, None))

    X = work.drop(columns=[spec.target])
    return X, y


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

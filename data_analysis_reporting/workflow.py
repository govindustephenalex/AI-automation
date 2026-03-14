from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ProblemType:
    kind: str  # "classification" | "regression"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


def infer_problem_type(y: pd.Series) -> ProblemType:
    if y.dtype == "O" or str(y.dtype).startswith("category") or y.dtype == "bool":
        return ProblemType(kind="classification")

    # Heuristic: few unique ints -> classification
    nunique = int(y.nunique(dropna=True))
    if nunique <= 20:
        if pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y):
            return ProblemType(kind="classification")
    return ProblemType(kind="regression")


def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column not found: {target}")
    X = df.drop(columns=[target])
    y = df[target]
    if X.shape[1] == 0:
        raise ValueError("No feature columns found after removing target.")
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def make_model(problem: ProblemType):
    if problem.kind == "classification":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000)
    else:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)


def evaluate(problem: ProblemType, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, Any]:
    if problem.kind == "classification":
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
        # Binary AUC when probabilities exist and exactly 2 classes
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                pass
        return metrics

    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def dataframe_profile(df: pd.DataFrame, *, max_cols: int = 200) -> Dict[str, Any]:
    # Keep it JSON-friendly (no numpy scalars) and lightweight.
    cols = list(df.columns)[:max_cols]
    missing = df[cols].isna().mean().sort_values(ascending=False)
    dtypes = df[cols].dtypes.astype(str)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": cols,
        "dtypes": {c: str(dtypes[c]) for c in cols},
        "missing_rate": {c: float(missing[c]) for c in missing.index},
    }


def build_report_md(
    *,
    data_path: Path,
    target: str,
    problem: ProblemType,
    profile: Dict[str, Any],
    metrics: Dict[str, Any],
    top_missing: pd.Series,
    sample_head: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Data Analysis & Reporting")
    lines.append("")
    lines.append(f"- Dataset: `{data_path.as_posix()}`")
    lines.append(f"- Target: `{target}`")
    lines.append(f"- Problem type: `{problem.kind}`")
    lines.append(f"- Rows: `{profile['rows']}`")
    lines.append(f"- Columns: `{profile['cols']}`")
    lines.append("")
    lines.append("## Quality Checks")
    lines.append("")
    lines.append("Top missing-rate columns:")
    lines.append("")
    for col, rate in top_missing.items():
        lines.append(f"- `{col}`: `{rate:.1%}` missing")
    lines.append("")
    lines.append("## Sample (head)")
    lines.append("")
    lines.append("```")
    lines.append(sample_head.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Model Metrics")
    lines.append("")
    for k, v in metrics.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is a baseline linear model with automatic preprocessing (impute + scale + one-hot).")
    lines.append("- For production: add data validation, leakage checks, drift monitoring, and model governance.")
    lines.append("")
    return "\n".join(lines)


def run_workflow(
    *,
    data_path: Path,
    target: str,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    df = load_csv(data_path)
    X, y = split_xy(df, target)

    problem = infer_problem_type(y)
    profile = dataframe_profile(df)

    # Split
    stratify = y if problem.kind == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pre = make_preprocessor(X_train)
    model = make_model(problem)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = None
    if problem.kind == "classification":
        # Use proba for AUC when available
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_test)
                if proba is not None and proba.shape[1] == 2:
                    y_proba = proba[:, 1]
            except Exception:
                pass

    metrics = evaluate(problem, np.asarray(y_test), np.asarray(y_pred), y_proba)

    missing = df.isna().mean().sort_values(ascending=False)
    top_missing = missing.head(10)

    report_md = build_report_md(
        data_path=data_path,
        target=target,
        problem=problem,
        profile=profile,
        metrics=metrics,
        top_missing=top_missing,
        sample_head=df.head(10),
    )

    return {"profile": profile, "metrics": metrics, "report_md": report_md}

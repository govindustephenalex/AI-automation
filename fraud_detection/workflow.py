from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.data import DataSpec, load_transactions_csv, split_xy
from fraud_detection.features import add_amount_features, add_time_features, make_preprocessor
from fraud_detection.metrics import classification_metrics, choose_threshold_by_f1
from fraud_detection.models import IsolationForestModel, RandomForestFraudModel


Mode = Literal["isolation_forest", "random_forest"]


@dataclass(frozen=True)
class FraudConfig:
    mode: Mode
    test_size: float = 0.2
    random_state: int = 42
    # When no labels exist: percentile threshold on anomaly score.
    anomaly_percentile: float = 0.99


def run_fraud_pipeline(
    *,
    data_path: Path,
    spec: DataSpec,
    cfg: FraudConfig,
) -> Dict[str, Any]:
    df = load_transactions_csv(str(data_path))
    X_raw, y = split_xy(df, spec)

    # Feature engineering (safe defaults)
    if spec.timestamp_col:
        X_raw = add_time_features(X_raw, spec.timestamp_col)
    X_raw = add_amount_features(X_raw, "amount")

    # Preserve identifiers for output
    ids = None
    if spec.id_col and spec.id_col in X_raw.columns:
        ids = X_raw[spec.id_col].astype(str)

    # Split (if labels exist, make a proper holdout)
    if y is not None and cfg.mode == "random_forest":
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
    else:
        X_train_raw, X_test_raw = train_test_split(
            X_raw, test_size=cfg.test_size, random_state=cfg.random_state
        )
        y_train = None
        y_test = y.loc[X_test_raw.index] if y is not None else None

    pre = make_preprocessor(X_train_raw)
    X_train = np.asarray(pre.fit_transform(X_train_raw), dtype=float)
    X_test = np.asarray(pre.transform(X_test_raw), dtype=float)

    metrics: Dict[str, Any] = {"mode": cfg.mode, "data_path": str(data_path)}
    predictions = pd.DataFrame(index=X_test_raw.index)
    if ids is not None:
        predictions["transaction_id"] = ids.loc[X_test_raw.index].values

    if cfg.mode == "random_forest":
        if y is None or spec.label_col is None:
            raise ValueError("random_forest mode requires label_col with 0/1 fraud labels.")

        model = RandomForestFraudModel()
        model.fit(X_train, np.asarray(y_train, dtype=int))

        score = model.predict_proba(X_test)
        thr, thr_summary = choose_threshold_by_f1(np.asarray(y_test, dtype=int), score)
        y_pred = (score >= thr).astype(int)

        metrics["threshold"] = thr
        metrics["threshold_selection"] = thr_summary
        metrics["evaluation"] = classification_metrics(np.asarray(y_test, dtype=int), y_pred, score)
        metrics["model"] = model.info()

        predictions["fraud_score"] = score
        predictions["is_fraud_pred"] = y_pred
        predictions["is_fraud_true"] = np.asarray(y_test, dtype=int)

        return {
            "metrics": metrics,
            "predictions": predictions.sort_values("fraud_score", ascending=False),
            "preprocessor": pre,
            "model": model,
        }

    # Isolation Forest (anomaly detection)
    iso = IsolationForestModel()
    # If labels exist, train on presumed-normal only (y==0) for a cleaner baseline.
    if y is not None and spec.label_col is not None:
        normal_mask = (y.loc[X_train_raw.index].astype(int) == 0).values
        X_fit = X_train[normal_mask] if normal_mask.any() else X_train
    else:
        X_fit = X_train
    iso.fit(X_fit)

    score = iso.score(X_test)

    # Threshold: supervised selection if labels exist, else percentile.
    if y_test is not None:
        thr, thr_summary = choose_threshold_by_f1(np.asarray(y_test, dtype=int), score)
        metrics["threshold"] = thr
        metrics["threshold_selection"] = thr_summary
        y_pred = (score >= thr).astype(int)
        metrics["evaluation"] = classification_metrics(np.asarray(y_test, dtype=int), y_pred, score)
        predictions["is_fraud_true"] = np.asarray(y_test, dtype=int)
        predictions["is_fraud_pred"] = y_pred
    else:
        thr = float(np.quantile(score, float(cfg.anomaly_percentile)))
        metrics["threshold"] = thr
        metrics["threshold_selection"] = {"method": "percentile", "percentile": float(cfg.anomaly_percentile)}
        predictions["is_fraud_pred"] = (score >= thr).astype(int)

    metrics["model"] = iso.info()
    predictions["fraud_score"] = score
    predictions = predictions.sort_values("fraud_score", ascending=False)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "preprocessor": pre,
        "model": iso,
    }

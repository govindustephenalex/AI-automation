from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "support": int(y_true.shape[0]),
        "positive_rate": float(np.mean(y_true)),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass
    return out


def choose_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Choose score threshold maximizing F1. scores: higher = more fraudulent/anomalous.
    Returns (threshold, summary).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if y_true.size == 0:
        return float("inf"), {"best_f1": 0.0}

    # Candidate thresholds at score quantiles to keep it fast/stable.
    qs = np.linspace(0.5, 0.999, 50)
    candidates = np.unique(np.quantile(scores, qs))

    best_thr = float(candidates[0]) if candidates.size else float("inf")
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    from sklearn.metrics import precision_recall_fscore_support

    for thr in candidates:
        y_pred = (scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
            best_prec = float(prec)
            best_rec = float(rec)

    return best_thr, {"best_f1": best_f1, "precision": best_prec, "recall": best_rec}

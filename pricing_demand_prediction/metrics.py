from __future__ import annotations

from typing import Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    # R^2
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if denom == 0.0 else float(1.0 - (float(np.sum(err**2)) / denom))

    # sMAPE
    denom_smape = np.maximum(1e-8, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    smape = float(np.mean(np.abs(err) / denom_smape))

    return {"mae": mae, "rmse": rmse, "r2": r2, "smape": smape}

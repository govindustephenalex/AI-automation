from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class XgbRegressor:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import xgboost as xgb
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: xgboost") from exc

        p = {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        if self.params:
            p.update(self.params)

        self.model = xgb.XGBRegressor(**p)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit.")
        return np.asarray(self.model.predict(X), dtype=float)

    def info(self) -> Dict[str, Any]:
        return {"backend": "xgb", "params": self.params or {}}

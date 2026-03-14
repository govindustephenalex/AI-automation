from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class IsolationForestModel:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        p = {
            "n_estimators": 400,
            "max_samples": "auto",
            "contamination": "auto",
            "random_state": 42,
            "n_jobs": -1,
        }
        if self.params:
            p.update(self.params)
        self.model = IsolationForest(**p)
        self.model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score where higher means more anomalous.
        """
        if self.model is None:
            raise RuntimeError("Model not fit.")
        # decision_function: higher means more normal -> invert
        s = -np.asarray(self.model.decision_function(X), dtype=float)
        return s

    def info(self) -> Dict[str, Any]:
        return {"backend": "isolation_forest", "params": self.params or {}}


@dataclass
class RandomForestFraudModel:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestClassifier

        p = {
            "n_estimators": 600,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        if self.params:
            p.update(self.params)
        self.model = RandomForestClassifier(**p)
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit.")
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba is not None and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=float)
        # Fallback
        return np.asarray(self.model.predict(X), dtype=float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        score = self.predict_proba(X)
        return (score >= float(threshold)).astype(int)

    def info(self) -> Dict[str, Any]:
        return {"backend": "random_forest", "params": self.params or {}}

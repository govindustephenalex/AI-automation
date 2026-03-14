from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pricing_demand_prediction.data import FeatureSpec, load_csv, make_preprocessor, split_xy
from pricing_demand_prediction.metrics import regression_metrics
from pricing_demand_prediction.models import TfRegressor, TorchRegressor, XgbRegressor
from pricing_demand_prediction.optimize import PriceGrid, recommend_prices


Backend = Literal["xgb", "tf", "torch"]


@dataclass(frozen=True)
class TrainConfig:
    backend: Backend
    test_size: float = 0.2
    random_state: int = 42
    price_min: float = 1.0
    price_max: float = 100.0
    price_step: float = 1.0
    currency: str = "USD"


def _make_model(backend: Backend):
    if backend == "xgb":
        return XgbRegressor()
    if backend == "tf":
        return TfRegressor()
    if backend == "torch":
        return TorchRegressor()
    raise ValueError(f"Unknown backend: {backend}")


def train_and_recommend(
    *,
    data_path: Path,
    spec: FeatureSpec,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    df = load_csv(str(data_path))
    X_raw, y = split_xy(df, spec)

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Preprocess
    pre = make_preprocessor(X_train_raw)
    X_train = pre.fit_transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    # Train
    model = _make_model(cfg.backend)
    model.fit(np.asarray(X_train), np.asarray(y_train))

    # Evaluate
    pred = model.predict(np.asarray(X_test))
    metrics = regression_metrics(np.asarray(y_test), np.asarray(pred))

    # Price optimization on a sample (or full test set)
    grid = PriceGrid(min_price=cfg.price_min, max_price=cfg.price_max, step=cfg.price_step)

    def predict_demand_for_raw(Xr: pd.DataFrame) -> np.ndarray:
        Xp = pre.transform(Xr)
        return model.predict(np.asarray(Xp))

    recs = recommend_prices(
        X_raw=X_test_raw.reset_index(drop=True),
        price_col=spec.price_col,
        grid=grid,
        predict_demand=predict_demand_for_raw,
    )

    artifacts = {
        "backend": cfg.backend,
        "data_path": str(data_path),
        "target": spec.target,
        "price_col": spec.price_col,
        "date_col": spec.date_col,
        "metrics": metrics,
        "grid": {"min_price": cfg.price_min, "max_price": cfg.price_max, "step": cfg.price_step},
    }

    # ColumnTransformer is not trivially JSON; store schema only.
    try:
        artifacts["preprocess_schema"] = {
            "feature_columns": list(X_raw.columns),
        }
    except Exception:
        pass

    return {
        "metrics": metrics,
        "recommendations": recs,
        "artifacts": artifacts,
    }

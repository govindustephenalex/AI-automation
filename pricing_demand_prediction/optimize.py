from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceGrid:
    min_price: float
    max_price: float
    step: float

    def values(self) -> np.ndarray:
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.max_price < self.min_price:
            raise ValueError("max_price must be >= min_price")
        n = int(np.floor((self.max_price - self.min_price) / self.step)) + 1
        return self.min_price + self.step * np.arange(n, dtype=float)


def recommend_prices(
    *,
    X_raw: pd.DataFrame,
    price_col: str,
    grid: PriceGrid,
    predict_demand: Callable[[pd.DataFrame], np.ndarray],
    clamp_demand_min: float = 0.0,
) -> pd.DataFrame:
    """
    For each row of context features, choose the price in grid that maximizes revenue:
      revenue = price * predicted_demand
    """
    prices = grid.values()
    if price_col not in X_raw.columns:
        raise KeyError(f"price_col not in features: {price_col}")

    rows: List[Dict] = []
    for idx in range(len(X_raw)):
        base = X_raw.iloc[[idx]].copy()

        best = {
            "row_index": int(idx),
            "current_price": float(base[price_col].iloc[0]),
            "recommended_price": float(base[price_col].iloc[0]),
            "pred_demand_at_current": float("nan"),
            "pred_demand_at_recommended": float("nan"),
            "pred_revenue_at_current": float("nan"),
            "pred_revenue_at_recommended": float("nan"),
        }

        # Current price metrics
        d0 = float(np.clip(predict_demand(base)[0], clamp_demand_min, None))
        p0 = float(base[price_col].iloc[0])
        best["pred_demand_at_current"] = d0
        best["pred_revenue_at_current"] = p0 * d0

        # Grid search
        best_revenue = -1.0
        best_price = p0
        best_demand = d0
        for p in prices:
            base[price_col] = p
            d = float(np.clip(predict_demand(base)[0], clamp_demand_min, None))
            revenue = float(p * d)
            if revenue > best_revenue:
                best_revenue = revenue
                best_price = float(p)
                best_demand = float(d)

        best["recommended_price"] = best_price
        best["pred_demand_at_recommended"] = best_demand
        best["pred_revenue_at_recommended"] = best_revenue
        best["pred_revenue_uplift"] = best_revenue - best["pred_revenue_at_current"]
        rows.append(best)

    return pd.DataFrame(rows)

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pricing_demand_prediction.data import FeatureSpec
from pricing_demand_prediction.workflow import TrainConfig, train_and_recommend


def main() -> int:
    parser = argparse.ArgumentParser(description="Pricing & Demand Prediction (TF / PyTorch / XGBoost).")
    parser.add_argument("--data", required=True, help="Path to CSV.")
    parser.add_argument("--target", default="demand", help="Target column (default: demand).")
    parser.add_argument("--price-col", default="price", help="Price column (default: price).")
    parser.add_argument("--date-col", default=None, help="Optional date column for calendar features.")
    parser.add_argument("--backend", choices=["xgb", "tf", "torch"], default="xgb")
    parser.add_argument("--report-dir", default="reports_pricing", help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--price-min", type=float, default=1.0)
    parser.add_argument("--price-max", type=float, default=100.0)
    parser.add_argument("--price-step", type=float, default=1.0)
    args = parser.parse_args()

    spec = FeatureSpec(target=args.target, price_col=args.price_col, date_col=args.date_col)
    cfg = TrainConfig(
        backend=args.backend,
        test_size=args.test_size,
        random_state=args.random_state,
        price_min=args.price_min,
        price_max=args.price_max,
        price_step=args.price_step,
    )

    out = train_and_recommend(data_path=Path(args.data), spec=spec, cfg=cfg)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "metrics.json").write_text(json.dumps(out["metrics"], indent=2), encoding="utf-8")
    out["recommendations"].to_csv(report_dir / "recommendations.csv", index=False)
    (report_dir / "model_artifacts.json").write_text(json.dumps(out["artifacts"], indent=2), encoding="utf-8")

    print(f"Wrote: {os.path.abspath(report_dir / 'metrics.json')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'recommendations.csv')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'model_artifacts.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fraud_detection.data import DataSpec
from fraud_detection.workflow import FraudConfig, run_fraud_pipeline


def main() -> int:
    p = argparse.ArgumentParser(description="Fraud Detection (Isolation Forest / Random Forest)")
    p.add_argument("--data", required=True, help="Path to transactions CSV.")
    p.add_argument("--mode", choices=["isolation_forest", "random_forest"], default="isolation_forest")
    p.add_argument("--label-col", default=None, help="Label column (0/1). Required for random_forest.")
    p.add_argument("--id-col", default="transaction_id", help="Optional id column for output.")
    p.add_argument("--timestamp-col", default="timestamp", help="Optional timestamp column for time features.")
    p.add_argument("--report-dir", default="reports_fraud", help="Output directory.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--anomaly-percentile", type=float, default=0.99, help="If no labels: threshold percentile.")
    p.add_argument("--save-model", action="store_true", help="Save model + preprocessor to model.joblib")
    args = p.parse_args()

    spec = DataSpec(label_col=args.label_col, id_col=args.id_col, timestamp_col=args.timestamp_col)
    cfg = FraudConfig(
        mode=args.mode,
        test_size=args.test_size,
        random_state=args.random_state,
        anomaly_percentile=args.anomaly_percentile,
    )

    out = run_fraud_pipeline(data_path=Path(args.data), spec=spec, cfg=cfg)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "metrics.json").write_text(json.dumps(out["metrics"], indent=2), encoding="utf-8")
    out["predictions"].to_csv(report_dir / "predictions.csv", index=False)

    if args.save_model:
        from joblib import dump

        dump({"preprocessor": out["preprocessor"], "model": out["model"]}, report_dir / "model.joblib")

    print(f"Wrote: {os.path.abspath(report_dir / 'metrics.json')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'predictions.csv')}")
    if args.save_model:
        print(f"Wrote: {os.path.abspath(report_dir / 'model.joblib')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

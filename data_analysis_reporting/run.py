from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from data_analysis_reporting.workflow import run_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Data Analysis & Reporting pipeline (Pandas/NumPy/sklearn).")
    parser.add_argument("--data", required=True, help="Path to CSV file.")
    parser.add_argument("--target", required=True, help="Target column name in the CSV.")
    parser.add_argument("--report-dir", default="reports", help="Output directory (default: reports).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split (default: 0.2).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    result = run_workflow(
        data_path=Path(args.data),
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    (report_dir / "report.md").write_text(result["report_md"], encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    (report_dir / "profile.json").write_text(json.dumps(result["profile"], indent=2), encoding="utf-8")

    print(f"Wrote: {os.path.abspath(report_dir / 'report.md')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'metrics.json')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'profile.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

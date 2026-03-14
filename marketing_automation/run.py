from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from marketing_automation.data import (
    build_rfm_features,
    load_events_csv,
    load_interactions_csv,
    load_items_csv,
)
from marketing_automation.metrics import marketing_metrics
from marketing_automation.orchestrator import build_campaign_actions
from marketing_automation.recommender_content import ContentRecommender
from marketing_automation.recommender_mf_tf import MatrixFactorizationRecommender
from marketing_automation.segmentation import Segmenter, label_segments


def main() -> int:
    p = argparse.ArgumentParser(description="Marketing Automation (Segmentation + Recommendations)")
    p.add_argument("--events", required=True, help="events.csv path (segmentation)")
    p.add_argument("--interactions", required=True, help="interactions.csv path (recommendations)")
    p.add_argument("--items", default=None, help="items.csv path (optional content-based cold start)")
    p.add_argument("--backend", choices=["tf_mf", "content", "hybrid"], default="hybrid")
    p.add_argument("--clusters", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--report-dir", default="reports_marketing")
    args = p.parse_args()

    events = load_events_csv(args.events)
    interactions = load_interactions_csv(args.interactions)

    # 1) Segmentation
    features = build_rfm_features(events)
    seg = Segmenter(n_clusters=int(args.clusters)).fit(features)
    labels = seg.predict(features)
    seg_names = label_segments(features, labels)

    segments_df = features[["customer_id"]].copy()
    segments_df["segment"] = labels.values.astype(int)
    segments_df["segment_name"] = segments_df["segment"].map(lambda s: seg_names.get(int(s), f"Segment_{s}"))

    # 2) Recommendations
    tf_mf = None
    content = None
    if args.backend in {"tf_mf", "hybrid"}:
        tf_mf = MatrixFactorizationRecommender().fit(interactions)
    if args.items and args.backend in {"content", "hybrid"}:
        items = load_items_csv(args.items)
        content = ContentRecommender().fit(items)

    popular = interactions.groupby("item_id")["value"].sum().sort_values(ascending=False)
    popular_items = [str(i) for i in popular.head(max(100, int(args.top_k))).index.tolist()]

    # Build per-customer recommendations
    recommendations: Dict[str, List[Tuple[str, float]]] = {}
    customer_ids = segments_df["customer_id"].astype(str).tolist()

    # Determine history counts for cold-start routing
    hist_counts = interactions.groupby("customer_id")["item_id"].size().to_dict()

    for cid in customer_ids:
        recs: List[Tuple[str, float]] = []
        if args.backend == "tf_mf" and tf_mf is not None:
            recs = tf_mf.recommend(cid, k=int(args.top_k))
        elif args.backend == "content" and content is not None:
            recs = content.recommend_for_user(interactions=interactions, customer_id=cid, k=int(args.top_k), min_history=1)
        else:
            # Hybrid: prefer TF MF for users with sufficient history; fall back to content, then popular.
            count = int(hist_counts.get(str(cid), 0))
            if tf_mf is not None and count >= 3:
                recs = tf_mf.recommend(cid, k=int(args.top_k))
            if not recs and content is not None and count >= 1:
                recs = content.recommend_for_user(
                    interactions=interactions, customer_id=cid, k=int(args.top_k), min_history=1
                )
            if not recs:
                recs = [(it, 0.0) for it in popular_items[: int(args.top_k)]]

        recommendations[str(cid)] = recs

    # 3) Campaign actions
    customers_df = segments_df.rename(columns={"segment": "segment"}).copy()
    actions = build_campaign_actions(
        customers=customers_df.rename(columns={"segment": "segment"}),
        segment_names=seg_names,
        recommendations=recommendations,
        top_k=int(args.top_k),
    )

    # 4) Outputs
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    segments_df.to_csv(report_dir / "segments.csv", index=False)

    rec_rows = []
    for cid, recs in recommendations.items():
        for rank, (item_id, score) in enumerate(recs, start=1):
            rec_rows.append({"customer_id": cid, "rank": rank, "item_id": item_id, "score": float(score)})
    pd.DataFrame(rec_rows).to_csv(report_dir / "recommendations.csv", index=False)

    actions.to_csv(report_dir / "campaign_actions.csv", index=False)

    metrics = marketing_metrics(customers=customer_ids, recommendations=recommendations)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote: {os.path.abspath(report_dir / 'segments.csv')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'recommendations.csv')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'campaign_actions.csv')}")
    print(f"Wrote: {os.path.abspath(report_dir / 'metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

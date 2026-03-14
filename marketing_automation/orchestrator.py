from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class CampaignRule:
    segment_name: str
    channel: str
    objective: str
    message_template: str


DEFAULT_RULES: List[CampaignRule] = [
    CampaignRule(
        segment_name="VIP_Active",
        channel="email",
        objective="upsell",
        message_template="Thanks for riding with us. Here are premium picks you may like: {items}",
    ),
    CampaignRule(
        segment_name="Frequent_Active",
        channel="push",
        objective="increase_frequency",
        message_template="Quick picks for your next ride: {items}",
    ),
    CampaignRule(
        segment_name="Churn_Risk",
        channel="sms",
        objective="winback",
        message_template="We miss you. Special offers available on: {items}",
    ),
    CampaignRule(
        segment_name="New_or_Low_Engagement",
        channel="push",
        objective="activation",
        message_template="Try these popular options to get started: {items}",
    ),
    CampaignRule(
        segment_name="Core",
        channel="push",
        objective="retain",
        message_template="Recommended for you: {items}",
    ),
]


def _rule_map(rules: List[CampaignRule]) -> Dict[str, CampaignRule]:
    return {r.segment_name: r for r in rules}


def build_campaign_actions(
    *,
    customers: pd.DataFrame,
    segment_names: Dict[int, str],
    recommendations: Dict[str, List[Tuple[str, float]]],
    top_k: int = 5,
    rules: Optional[List[CampaignRule]] = None,
) -> pd.DataFrame:
    """
    Build a concrete action list per customer:
    - segment
    - channel
    - objective
    - recommended items (JSON)
    - message text
    """
    rules = rules or DEFAULT_RULES
    rm = _rule_map(rules)

    rows = []
    for _, r in customers.iterrows():
        cid = str(r["customer_id"])
        seg_id = int(r["segment"])
        seg_name = segment_names.get(seg_id, f"Segment_{seg_id}")
        rule = rm.get(seg_name, rm["Core"])

        recs = recommendations.get(cid, [])[: int(top_k)]
        items = [it for it, _ in recs]
        message = rule.message_template.format(items=", ".join(items) if items else "our top picks")

        rows.append(
            {
                "customer_id": cid,
                "segment_id": seg_id,
                "segment_name": seg_name,
                "channel": rule.channel,
                "objective": rule.objective,
                "recommended_items": json.dumps(recs),
                "message": message,
            }
        )

    return pd.DataFrame(rows)

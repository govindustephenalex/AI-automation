from __future__ import annotations

from typing import Dict, List, Tuple


def marketing_metrics(
    *,
    customers: List[str],
    recommendations: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, float]:
    """
    Lightweight offline metrics for sanity checks:
    - coverage: fraction of customers with >=1 recommendation
    - avg_recs: average recommendation list size
    """
    n = float(len(customers))
    if n == 0:
        return {"coverage": 0.0, "avg_recs": 0.0}

    with_recs = 0
    total_recs = 0
    for c in customers:
        recs = recommendations.get(str(c), [])
        if recs:
            with_recs += 1
            total_recs += len(recs)

    return {"coverage": with_recs / n, "avg_recs": total_recs / n}

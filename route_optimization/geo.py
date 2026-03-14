from __future__ import annotations

import math
from typing import Tuple


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Great-circle distance between (lat, lon) points in kilometers.
    """
    lat1, lon1 = a
    lat2, lon2 = b

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    s = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(s), math.sqrt(max(0.0, 1 - s)))
    return r * c


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)

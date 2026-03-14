from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Stop:
    stop_id: str
    lat: float
    lon: float
    demand: int = 0

    @property
    def coord(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


def load_edges_csv(path: str):
    df = pd.read_csv(path)
    required = {"u", "v", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"edges.csv missing columns: {sorted(missing)}")
    return df[["u", "v", "weight"]]


def load_nodes_csv(path: str) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(path)
    required = {"node_id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"nodes.csv missing columns: {sorted(missing)}")
    return {str(r["node_id"]): (float(r["lat"]), float(r["lon"])) for _, r in df.iterrows()}


def load_stops_csv(path: str) -> List[Stop]:
    df = pd.read_csv(path)
    required = {"stop_id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stops.csv missing columns: {sorted(missing)}")

    stops: List[Stop] = []
    for _, r in df.iterrows():
        demand = int(r["demand"]) if "demand" in df.columns and pd.notna(r["demand"]) else 0
        stops.append(Stop(stop_id=str(r["stop_id"]), lat=float(r["lat"]), lon=float(r["lon"]), demand=demand))
    return stops


def index_stops(stops: List[Stop]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i, s in enumerate(stops):
        if s.stop_id in idx:
            raise ValueError(f"Duplicate stop_id: {s.stop_id}")
        idx[s.stop_id] = i
    return idx

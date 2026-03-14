from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from route_optimization.geo import haversine_km


def build_weighted_graph(edges_df) -> nx.Graph:
    """
    Build an undirected weighted graph from edges.csv.
    Use nx.DiGraph if you need directionality; keep Graph by default for ride-hailing road distance examples.
    """
    g = nx.Graph()
    for _, r in edges_df.iterrows():
        u = str(r["u"])
        v = str(r["v"])
        w = float(r["weight"])
        g.add_edge(u, v, weight=w)
    return g


@dataclass(frozen=True)
class PathResult:
    path: List[str]
    total_weight: float


def shortest_path_dijkstra(g: nx.Graph, source: str, target: str) -> PathResult:
    path = nx.shortest_path(g, source=source, target=target, weight="weight", method="dijkstra")
    dist = float(nx.shortest_path_length(g, source=source, target=target, weight="weight", method="dijkstra"))
    return PathResult(path=path, total_weight=dist)


def shortest_path_astar(g: nx.Graph, source: str, target: str, coords: Dict[str, Tuple[float, float]]) -> PathResult:
    def h(a: str, b: str) -> float:
        ca = coords.get(a)
        cb = coords.get(b)
        if not ca or not cb:
            return 0.0
        return haversine_km(ca, cb)

    path = nx.astar_path(g, source=source, target=target, heuristic=h, weight="weight")
    dist = 0.0
    for u, v in zip(path, path[1:]):
        dist += float(g[u][v].get("weight", 1.0))
    return PathResult(path=path, total_weight=float(dist))

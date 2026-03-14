from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from route_optimization.geo import haversine_km


def dist_matrix_from_coords(coords: List[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m[i, j] = haversine_km(coords[i], coords[j])
    return m


def dist_matrix_from_graph(g: nx.Graph, nodes: List[str]) -> np.ndarray:
    """
    Compute pairwise shortest-path distances over a weighted graph.
    If some pairs are disconnected, raises.
    """
    n = len(nodes)
    m = np.zeros((n, n), dtype=float)
    for i, a in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(g, a, weight="weight")
        for j, b in enumerate(nodes):
            if i == j:
                continue
            if b not in lengths:
                raise ValueError(f"Graph is disconnected for pair: {a} -> {b}")
            m[i, j] = float(lengths[b])
    return m

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class TspResult:
    route: List[int]  # indices in stops list
    total_distance: float


def _route_distance(route: List[int], dist: np.ndarray) -> float:
    d = 0.0
    for a, b in zip(route, route[1:]):
        d += float(dist[a, b])
    return d


def nearest_neighbor_tsp(dist: np.ndarray, start: int = 0, return_to_start: bool = True) -> TspResult:
    n = int(dist.shape[0])
    if n == 0:
        return TspResult(route=[], total_distance=0.0)

    unvisited = set(range(n))
    unvisited.remove(start)
    route = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: float(dist[cur, j]))
        unvisited.remove(nxt)
        route.append(nxt)
        cur = nxt
    if return_to_start:
        route.append(start)
    return TspResult(route=route, total_distance=_route_distance(route, dist))


def two_opt(route: List[int], dist: np.ndarray, max_iters: int = 2000) -> List[int]:
    """
    Classic 2-opt improvement for a cycle route.
    """
    best = route[:]
    best_d = _route_distance(best, dist)
    n = len(best)
    if n < 5:
        return best

    iters = 0
    improved = True
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                a, b = best[i - 1], best[i]
                c, d = best[k], best[k + 1]
                # If swapping reduces distance, reverse segment.
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if float(delta) < -1e-9:
                    best[i : k + 1] = reversed(best[i : k + 1])
                    best_d += float(delta)
                    improved = True
    return best


def solve_tsp_heuristic(dist: np.ndarray, start: int = 0) -> TspResult:
    init = nearest_neighbor_tsp(dist, start=start, return_to_start=True)
    improved = two_opt(init.route, dist)
    return TspResult(route=improved, total_distance=_route_distance(improved, dist))

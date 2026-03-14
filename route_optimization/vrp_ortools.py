from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VrpSolution:
    routes: List[List[int]]  # per vehicle, indices in stops list (includes depot at ends)
    total_distance: float


def solve_vrp(
    *,
    dist_km: np.ndarray,
    demands: List[int],
    depot_index: int,
    vehicle_count: int,
    vehicle_capacity: int,
    time_limit_s: int = 10,
) -> VrpSolution:
    """
    Capacitated VRP using OR-Tools Routing solver.
    dist_km: square matrix
    demands: per node demand (0 for depot)
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: ortools") from exc

    n = int(dist_km.shape[0])
    if n == 0:
        return VrpSolution(routes=[], total_distance=0.0)
    if dist_km.shape[0] != dist_km.shape[1]:
        raise ValueError("dist_km must be square")
    if len(demands) != n:
        raise ValueError("demands length must match dist_km size")

    # OR-Tools expects integer costs.
    cost = np.round(dist_km * 1000.0).astype(int)  # meters-like integer

    manager = pywrapcp.RoutingIndexManager(n, int(vehicle_count), int(depot_index))
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index: int, to_index: int) -> int:
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return int(cost[a, b])

    dist_cb_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_index)

    def demand_cb(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return int(demands[node])

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_index,
        0,
        [int(vehicle_capacity)] * int(vehicle_count),
        True,
        "Capacity",
    )

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.seconds = int(time_limit_s)

    assignment = routing.SolveWithParameters(search)
    if assignment is None:
        raise RuntimeError("No solution found (increase time limit or adjust constraints).")

    routes: List[List[int]] = []
    total_m = 0
    for v in range(int(vehicle_count)):
        idx = routing.Start(v)
        route: List[int] = [manager.IndexToNode(idx)]
        while not routing.IsEnd(idx):
            nxt = assignment.Value(routing.NextVar(idx))
            total_m += routing.GetArcCostForVehicle(idx, nxt, v)
            idx = nxt
            route.append(manager.IndexToNode(idx))
        routes.append(route)

    return VrpSolution(routes=routes, total_distance=float(total_m) / 1000.0)

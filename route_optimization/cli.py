from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np

from route_optimization.distance_matrix import dist_matrix_from_coords, dist_matrix_from_graph
from route_optimization.graph import build_weighted_graph, shortest_path_astar, shortest_path_dijkstra
from route_optimization.io import index_stops, load_edges_csv, load_nodes_csv, load_stops_csv
from route_optimization.tsp import solve_tsp_heuristic
from route_optimization.vrp_ortools import solve_vrp


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def cmd_shortest_path(args) -> int:
    edges = load_edges_csv(args.edges)
    g = build_weighted_graph(edges)

    if args.algo == "astar":
        if not args.nodes:
            raise SystemExit("--nodes is required for astar")
        coords = load_nodes_csv(args.nodes)
        res = shortest_path_astar(g, args.source, args.target, coords)
    else:
        res = shortest_path_dijkstra(g, args.source, args.target)

    _print({"path": res.path, "total_weight": res.total_weight, "algo": args.algo})
    return 0


def cmd_tsp(args) -> int:
    stops = load_stops_csv(args.stops)
    idx = index_stops(stops)
    if args.start not in idx:
        raise SystemExit(f"start stop_id not found: {args.start}")
    start_i = idx[args.start]

    coords = [s.coord for s in stops]
    dist = dist_matrix_from_coords(coords)
    res = solve_tsp_heuristic(dist, start=start_i)

    route_ids = [stops[i].stop_id for i in res.route]
    _print({"route_stop_ids": route_ids, "total_distance_km": res.total_distance})
    return 0


def cmd_vrp(args) -> int:
    stops = load_stops_csv(args.stops)
    idx = index_stops(stops)
    if args.depot not in idx:
        raise SystemExit(f"depot stop_id not found: {args.depot}")
    depot_i = idx[args.depot]

    # Distance source
    if args.edges:
        edges = load_edges_csv(args.edges)
        g = build_weighted_graph(edges)
        nodes = [s.stop_id for s in stops]
        dist = dist_matrix_from_graph(g, nodes)
    else:
        coords = [s.coord for s in stops]
        dist = dist_matrix_from_coords(coords)

    demands = [int(s.demand) for s in stops]
    demands[depot_i] = 0

    sol = solve_vrp(
        dist_km=np.asarray(dist, dtype=float),
        demands=demands,
        depot_index=depot_i,
        vehicle_count=int(args.vehicles),
        vehicle_capacity=int(args.capacity),
        time_limit_s=int(args.time_limit_s),
    )

    routes = []
    for r in sol.routes:
        routes.append([stops[i].stop_id for i in r])
    _print({"routes_stop_ids": routes, "total_distance_km": sol.total_distance})
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Route Optimization (OR-Tools + NetworkX)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("shortest-path", help="Shortest path on weighted graph")
    sp.add_argument("--edges", required=True, help="edges.csv path")
    sp.add_argument("--nodes", default=None, help="nodes.csv path (required for astar)")
    sp.add_argument("--source", required=True)
    sp.add_argument("--target", required=True)
    sp.add_argument("--algo", choices=["dijkstra", "astar"], default="dijkstra")
    sp.set_defaults(fn=cmd_shortest_path)

    tsp = sub.add_parser("tsp", help="TSP heuristic from stop coordinates")
    tsp.add_argument("--stops", required=True, help="stops.csv path")
    tsp.add_argument("--start", required=True, help="start/depot stop_id")
    tsp.set_defaults(fn=cmd_tsp)

    vrp = sub.add_parser("vrp", help="Capacitated VRP via OR-Tools")
    vrp.add_argument("--stops", required=True, help="stops.csv path")
    vrp.add_argument("--depot", required=True, help="depot stop_id")
    vrp.add_argument("--vehicles", type=int, default=3)
    vrp.add_argument("--capacity", type=int, default=40)
    vrp.add_argument("--time-limit-s", type=int, default=10)
    vrp.add_argument("--edges", default=None, help="Optional edges.csv path to use graph shortest-path distances")
    vrp.set_defaults(fn=cmd_vrp)

    args = p.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

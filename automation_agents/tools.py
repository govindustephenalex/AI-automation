from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

from automation_agents.config import AutomationSettings


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def make_tools():
    """
    Create a unified tool set that wraps all earlier modules.
    Tools return JSON strings so LLMs can reliably parse results.
    """
    settings = AutomationSettings()

    # Guardians tools (re-use existing tool factory)
    from guardians.storage import JsonStore as GuardiansStore
    from guardians_ai.tools import make_guardians_tools

    guardians_store = GuardiansStore(settings.guardians_db_path)
    guardians_tools = make_guardians_tools(guardians_store)

    # Customer support tools
    from customer_support.faq import DEFAULT_FAQ, build_faq_index, search_faq
    from customer_support.storage import JsonStore as SupportStore
    from customer_support.ticketing import Ticketing
    from customer_support.intent_sentiment import nlu
    from customer_support.nlp_signals import build_signals

    support_store = SupportStore(settings.support_db_path)
    ticketing = Ticketing(support_store)
    faq_index = build_faq_index(DEFAULT_FAQ)

    @tool("support_search_faq")
    def support_search_faq(query: str, k: int = 3) -> str:
        """Search support FAQ and return top matches."""
        hits = search_faq(faq_index, query, k=int(k))
        return _json([{"key": key, "score": score, "answer": ans} for key, score, ans in hits])

    @tool("support_create_ticket")
    def support_create_ticket(customer_id: str, subject: str, description: str, priority: str = "normal") -> str:
        """Create a customer support ticket."""
        sig = build_signals(description)
        n = nlu(description)
        t = ticketing.create_ticket(
            customer_id=customer_id,
            subject=subject,
            description=description,
            priority=priority,
            intent=n.intent,
            sentiment=n.sentiment,
            entities=sig.entities,
        )
        return _json(t)

    @tool("support_get_ticket")
    def support_get_ticket(ticket_id: str) -> str:
        """Get a ticket by id."""
        return _json(ticketing.get_ticket(ticket_id))

    @tool("support_set_ticket_status")
    def support_set_ticket_status(ticket_id: str, status: str) -> str:
        """Update ticket status (open|pending|resolved|closed)."""
        return _json(ticketing.set_status(ticket_id, status))

    @tool("support_list_tickets")
    def support_list_tickets(limit: int = 10) -> str:
        """List recent tickets."""
        return _json(ticketing.list_recent(limit=int(limit)))

    # Analytics & reporting tool
    from data_analysis_reporting.workflow import run_workflow

    @tool("analytics_run_report")
    def analytics_run_report(data_path: str, target: str, test_size: float = 0.2, random_state: int = 42) -> str:
        """Run baseline analysis/report on a CSV and return metrics + profile + markdown report."""
        out = run_workflow(
            data_path=Path(data_path),
            target=target,
            test_size=float(test_size),
            random_state=int(random_state),
        )
        return _json(out)

    # Pricing & demand prediction tool
    from pricing_demand_prediction.data import FeatureSpec
    from pricing_demand_prediction.workflow import TrainConfig, train_and_recommend

    @tool("pricing_train_and_recommend")
    def pricing_train_and_recommend(
        data_path: str,
        target: str = "demand",
        price_col: str = "price",
        backend: str = "xgb",
        test_size: float = 0.2,
        price_min: float = 1.0,
        price_max: float = 100.0,
        price_step: float = 1.0,
    ) -> str:
        """Train a demand model and recommend prices (revenue-max over grid). backend: xgb|tf|torch."""
        spec = FeatureSpec(target=target, price_col=price_col, date_col=None)
        cfg = TrainConfig(
            backend=backend,  # type: ignore[arg-type]
            test_size=float(test_size),
            price_min=float(price_min),
            price_max=float(price_max),
            price_step=float(price_step),
        )
        out = train_and_recommend(data_path=Path(data_path), spec=spec, cfg=cfg)
        # Avoid returning huge tables; return metrics + first N recs.
        recs = out["recommendations"].head(20).to_dict(orient="records")
        return _json({"metrics": out["metrics"], "artifacts": out["artifacts"], "recommendations_head": recs})

    # Route optimization tools
    from route_optimization.distance_matrix import dist_matrix_from_coords
    from route_optimization.graph import build_weighted_graph, shortest_path_astar, shortest_path_dijkstra
    from route_optimization.io import index_stops, load_edges_csv, load_nodes_csv, load_stops_csv
    from route_optimization.tsp import solve_tsp_heuristic
    from route_optimization.vrp_ortools import solve_vrp
    import numpy as np

    @tool("routing_shortest_path")
    def routing_shortest_path(edges_csv: str, source: str, target: str, algo: str = "dijkstra", nodes_csv: str = "") -> str:
        """Shortest path on edges.csv. algo: dijkstra|astar. For astar, provide nodes_csv with lat/lon."""
        edges = load_edges_csv(edges_csv)
        g = build_weighted_graph(edges)
        if algo == "astar":
            if not nodes_csv:
                raise ValueError("nodes_csv required for astar")
            coords = load_nodes_csv(nodes_csv)
            res = shortest_path_astar(g, source, target, coords)
        else:
            res = shortest_path_dijkstra(g, source, target)
        return _json({"path": res.path, "total_weight": res.total_weight, "algo": algo})

    @tool("routing_tsp")
    def routing_tsp(stops_csv: str, start_stop_id: str) -> str:
        """Solve TSP heuristically from stops.csv coords. Returns stop_id route + distance_km."""
        stops = load_stops_csv(stops_csv)
        idx = index_stops(stops)
        if start_stop_id not in idx:
            raise KeyError(f"Unknown start_stop_id: {start_stop_id}")
        coords = [s.coord for s in stops]
        dist = dist_matrix_from_coords(coords)
        res = solve_tsp_heuristic(dist, start=idx[start_stop_id])
        route_ids = [stops[i].stop_id for i in res.route]
        return _json({"route_stop_ids": route_ids, "total_distance_km": res.total_distance})

    @tool("routing_vrp")
    def routing_vrp(
        stops_csv: str,
        depot_stop_id: str,
        vehicles: int = 3,
        capacity: int = 40,
        time_limit_s: int = 10,
    ) -> str:
        """Solve capacitated VRP from stops.csv coords using OR-Tools."""
        stops = load_stops_csv(stops_csv)
        idx = index_stops(stops)
        if depot_stop_id not in idx:
            raise KeyError(f"Unknown depot_stop_id: {depot_stop_id}")
        depot_i = idx[depot_stop_id]
        coords = [s.coord for s in stops]
        dist = dist_matrix_from_coords(coords)
        demands = [int(s.demand) for s in stops]
        demands[depot_i] = 0
        sol = solve_vrp(
            dist_km=np.asarray(dist, dtype=float),
            demands=demands,
            depot_index=int(depot_i),
            vehicle_count=int(vehicles),
            vehicle_capacity=int(capacity),
            time_limit_s=int(time_limit_s),
        )
        routes = [[stops[i].stop_id for i in r] for r in sol.routes]
        return _json({"routes_stop_ids": routes, "total_distance_km": sol.total_distance})

    # Fraud detection tools
    from fraud_detection.data import DataSpec as FraudDataSpec
    from fraud_detection.workflow import FraudConfig, run_fraud_pipeline

    @tool("fraud_score_transactions")
    def fraud_score_transactions(
        data_path: str,
        mode: str = "isolation_forest",
        label_col: str = "",
        id_col: str = "transaction_id",
        timestamp_col: str = "timestamp",
        anomaly_percentile: float = 0.99,
    ) -> str:
        """Score transactions for fraud. mode: isolation_forest|random_forest. label_col required for random_forest."""
        spec = FraudDataSpec(label_col=label_col or None, id_col=id_col or None, timestamp_col=timestamp_col or None)
        cfg = FraudConfig(mode=mode, anomaly_percentile=float(anomaly_percentile))  # type: ignore[arg-type]
        out = run_fraud_pipeline(data_path=Path(data_path), spec=spec, cfg=cfg)
        preds = out["predictions"].head(30).to_dict(orient="records")
        return _json({"metrics": out["metrics"], "top_predictions": preds})

    return {
        "guardians": guardians_tools["all"],
        "support": [
            support_search_faq,
            support_create_ticket,
            support_get_ticket,
            support_set_ticket_status,
            support_list_tickets,
        ],
        "analytics": [analytics_run_report],
        "pricing": [pricing_train_and_recommend],
        "routing": [routing_shortest_path, routing_tsp, routing_vrp],
        "fraud": [fraud_score_transactions],
        "all": [
            *guardians_tools["all"],
            support_search_faq,
            support_create_ticket,
            support_get_ticket,
            support_set_ticket_status,
            support_list_tickets,
            analytics_run_report,
            pricing_train_and_recommend,
            routing_shortest_path,
            routing_tsp,
            routing_vrp,
            fraud_score_transactions,
        ],
    }

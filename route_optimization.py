import networkx as nx
from ortools.pywraplp import Solver
import matplotlib.pyplot as plt
import numpy as np

def solve_route_optimization(locations, distances, capacity):
    """
    Solves the route optimization problem using Google OR-Tools and NetworkX.

    Args:
        locations: A list of location names (e.g., ['A', 'B', 'C']).
        distances: A dictionary where keys are tuples of location pairs
                   (e.g., ('A', 'B')) and values are the distances between them.
        capacity: The maximum capacity of the route.

    Returns:
        A tuple containing:
            - best_route: A list of locations representing the optimized route.
            - total_distance: The total distance of the optimized route.
    """

    # 1. Create a NetworkX Graph
    graph = nx.Graph()
    for location in locations:
        graph.add_node(location)

    for (u, v), distance in distances.items():
        graph.add_edge(u, v, weight=distance)

    # 2. Google OR-Tools Solver
    solver = Solver('route_optimization')
    if not solver.is_running():
        solver.enableAlpha()
    
    # Decision Variables
    x = {}
    for u, v in graph.edges():
        x[u, v] = solver.IntVar(0, 1, f'x_{u}_{v}')

    # Objective Function: Minimize Total Distance
    objective = solver.Objective()
    for u, v in graph.edges():
        objective.SetCoefficient(x[u, v], graph[u][v]['weight'])
    objective.SetMinimization()

    # Constraints
    # 1. Each location must be visited exactly once
    for location in locations:
        flow_in = 0
        for u, v in graph.edges():
            if v == location:
                flow_in += x[u, v]
        solver.Add(flow_in == 1)

    # 2. Capacity constraint (simplified - assumes each edge has a capacity equal to the distance)
    for u, v in graph.edges():
        solver.Add(x[u, v] <= capacity)

    # Solve
    status = solver.Solve()

    # 3. Extract the Optimal Route from OR-Tools
    best_route = []
    if status == solver.OPTIMAL:
        for u, v in graph.edges():
            if x[u, v].solution_value() == 1:
                best_route.append(u)
                best_route.append(v)
                break  # Stop after finding the first connected edge
    else:
        print("No solution found.")
        return None, None

    # 4. Calculate Total Distance
    total_distance = 0
    for i in range(0, len(best_route) - 1, 2):
        total_distance += graph[best_route[i]][best[i+1]]['weight']

    return best_route, total_distance


if __name__ == '__main__':
    # Example Usage:
    locations = ['A', 'B', 'C', 'D']
    distances = {
        ('A', 'B'): 10,
        ('A', 'C'): 15,
        ('A', 'D'): 20,
        ('B', 'C'): 35,
        ('B', 'D'): 25,
        ('C', 'D'): 30
    }
    capacity = 10

    best_route, total_distance = solve_route_optimization(locations, distances, capacity)

    if best_route:
        print("Optimized Route:", best_route)
        print("Total Distance:", total_distance)
        
        # Visualization (optional)
        graph = nx.Graph()
        for location in locations:
            graph.add_node(location)
        for (u, v), distance in distances.items():
            graph.add_edge(u, v, weight=distance)

        pos = nx.spring_layout(graph)  # You can use different layouts
        nx.draw(graph, pos, with_labels=True, node_size=500, node_color="skyblue")
        # Highlight the edges in the optimized route
        for i in range(0, len(best_route) - 1, 2):
            u, v = best_route[i], best_route[i+1]
            nx.draw_networkx_edge(graph, u, v, edge_color='red', width=2)
        plt.show()

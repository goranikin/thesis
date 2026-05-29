"""
Greedy decoder for CVRP heatmaps.

The heatmap is a per-edge probability that the edge appears in the optimal
solution. A valid CVRP solution must satisfy:

  1. Every customer has degree exactly 2 (one in-edge, one out-edge).
  2. The depot has degree 2K where K is the number of routes (unbounded
     during decoding — we discover K).
  3. Removing the depot must leave only paths (no customer-only cycles).
  4. The sum of demands on each route does not exceed the vehicle capacity Q.

The decoder ranks edges by  (A_ij + A_ji) / dist_ij  (same as TSP) and adds
edges greedily, skipping any that would violate a constraint.

After edges are fixed, ``_extract_routes`` walks from the depot to recover
the K routes.
"""

import numpy as np
import torch

DEPOT = 0


def greedy_decode_cvrp(
    heatmap: torch.Tensor,
    edge_index: torch.Tensor,
    node_coords: torch.Tensor,
    demands: torch.Tensor,
    capacity: int,
) -> list[list[int]]:
    """
    Args:
        heatmap:     (E,) edge probabilities from the model.
        edge_index:  (2, E)
        node_coords: (N+1, 2) — node 0 is the depot.
        demands:     (N+1,)   — demands[0] = 0.
        capacity:    int.
    Returns:
        routes: list of routes, each a list of 0-indexed customer ids
                (e.g., [[3, 7], [1, 5, 12]]). Depot is implicit at start/end.
    """
    N1 = node_coords.shape[0]  # number of nodes
    E = edge_index.shape[1]  # number of edges

    edge_index_np = edge_index.detach().cpu().numpy()
    heatmap_np = heatmap.detach().cpu().numpy()
    coords_np = node_coords.detach().cpu().numpy()
    demands_np = demands.detach().cpu().numpy().astype(np.int64)

    edge_scores: dict[tuple[int, int], float] = {}
    for k in range(E):
        u = int(edge_index_np[0, k])
        v = int(edge_index_np[1, k])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        score = float(heatmap_np[k])
        edge_scores[key] = edge_scores.get(key, 0.0) + score
        # heatmap[u -> v] + heatmap[v -> u]

    for u, v in edge_scores:
        dx = coords_np[u, 0] - coords_np[v, 0]
        dy = coords_np[u, 1] - coords_np[v, 1]
        dist = float((dx * dx + dy * dy) ** 0.5)
        edge_scores[(u, v)] /= dist + 1e-8
        # divided by dist

    # sort by score
    # score = (heatmap[u -> v] + heatmap[v -> u]) / dist + regularization term
    sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)

    # adj[node] = Current selected adjacency list
    adj: dict[int, list[int]] = {i: [] for i in range(N1)}
    # Union-find restricted to customers, used to forbid customer-only cycles.
    parent = list(range(N1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # total demand in the chain.
    chain_demand: dict[int, int] = {i: int(demands_np[i]) for i in range(N1)}

    selected_customer_edges = 0  # excludes depot edges; target is N - K (unknown)
    selected_depot_edges = 0  # 2 per route

    for (u, v), _ in sorted_edges:
        # Check A - degree cap
        # A customer node has only 2 edges, and a depot node has no degree cap.
        if u != DEPOT and len(adj[u]) >= 2:
            continue
        if v != DEPOT and len(adj[v]) >= 2:
            continue

        # Check B - capacity and subtour, based on edge type
        if u == DEPOT or v == DEPOT:
            # Depot-customer edge. The customer side must not already be the
            # endpoint of a chain whose total demand exceeds capacity.
            cust = v if u == DEPOT else u
            root = find(cust)
            if chain_demand[root] > capacity:
                continue  # capacity violation — refuse
            adj[u].append(v)
            adj[v].append(u)
            selected_depot_edges += 1
        else:
            # Customer-customer edge. Reject if it merges two chains that are
            # already linked (would form a customer-only cycle), or if the
            # merged chain exceeds capacity.
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            merged_demand = chain_demand[ru] + chain_demand[rv]
            if merged_demand > capacity:
                continue
            adj[u].append(v)
            adj[v].append(u)
            union(u, v)
            new_root = find(u)
            chain_demand[new_root] = merged_demand
            selected_customer_edges += 1

        is_customer_has_degree_2: bool = all(len(adj[i]) == 2 for i in range(1, N1))
        if is_customer_has_degree_2:
            break

    # Some customers may still have degree < 2 if the heatmap was weak.
    # Close any open chain by attaching its loose end(s) to the depot.
    open_customers = [i for i in range(1, N1) if len(adj[i]) < 2]
    for u in open_customers:
        while len(adj[u]) < 2:
            # Always allowed: attach to depot.
            adj[u].append(DEPOT)
            adj[DEPOT].append(u)

    return _extract_routes(adj)


def _extract_routes(
    adj: dict[int, list[int]],
) -> list[list[int]]:
    """
    Walk away from the depot along each chosen depot edge to recover a route.
    The depot may appear in multiple routes (one per "departure" edge).

    Implementation note: we work in a multi-graph because singleton routes
    (depot -> c -> depot) use the depot-customer edge TWICE. We therefore
    track *remaining* incidences via a list per node and pop entries as we
    consume them, instead of using an "edge_key visited set" (which can't
    distinguish parallel edges).
    """
    # Mutable copy of adj — we will pop neighbours off as we walk.
    remaining: dict[int, list[int]] = {u: list(nb) for u, nb in adj.items()}

    def consume(u: int, v: int) -> None:
        """Remove exactly one (u,v) and one (v,u) incidence."""
        remaining[u].remove(v)
        remaining[v].remove(u)

    routes: list[list[int]] = []

    # As long as the depot has any unconsumed neighbours, start a new route.
    while remaining[DEPOT]:
        start = remaining[DEPOT][0]
        consume(DEPOT, start)
        route = [start]

        current = start
        while current != DEPOT:
            if not remaining[current]:
                # Open chain — close it at the depot defensively. This should
                # not happen if the repair pass ran, but handle it gracefully.
                break
            nxt = remaining[current][0]
            consume(current, nxt)
            if nxt == DEPOT:
                break
            route.append(nxt)
            current = nxt

        if route:
            routes.append(route)

    return routes


# ------------------------------------------------------------------ utilities
def compute_route_length(routes: list[list[int]], node_coords: torch.Tensor) -> float:
    """
    Total Euclidean distance over all routes, depot included at start and end
    of every route.
    """
    coords = node_coords.detach().cpu().numpy()
    total = 0.0
    for route in routes:
        path = [DEPOT, *route, DEPOT]
        for u, v in zip(path[:-1], path[1:]):
            dx = coords[u, 0] - coords[v, 0]
            dy = coords[u, 1] - coords[v, 1]
            total += float((dx * dx + dy * dy) ** 0.5)
    return total


def compute_overcapacity_violation(
    routes: list[list[int]], demands: torch.Tensor, capacity: int
) -> tuple[int, int]:
    """
    Count routes that exceed capacity and the total demand overshoot.
    Returns (num_violating_routes, total_overshoot).
    """
    d = demands.detach().cpu().numpy()
    violating = 0
    overshoot = 0
    for route in routes:
        load = int(sum(d[c] for c in route))
        if load > capacity:
            violating += 1
            overshoot += load - capacity
    return violating, overshoot


def routes_to_canonical(routes: list[list[int]]) -> str:
    """Stable text representation, useful for logging / unit tests."""
    rendered = [" ".join(str(c) for c in route) for route in routes]
    return " | ".join(rendered)


def _open_route_distance(
    route: list[int], coords: np.ndarray, depot: int = DEPOT
) -> float:
    """Euclidean length of depot -> customers -> depot (open path)."""
    if not route:
        return 0.0
    total = 0.0
    path = [depot, *route, depot]
    for u, v in zip(path[:-1], path[1:]):
        dx = coords[u, 0] - coords[v, 0]
        dy = coords[u, 1] - coords[v, 1]
        total += float((dx * dx + dy * dy) ** 0.5)
    return total


def two_opt_route(
    route: list[int],
    node_coords: torch.Tensor,
    max_iterations: int = 100,
) -> list[int]:
    """
    Intra-route 2-opt on one open CVRP route (depot fixed at both ends).

    Reversing a segment does not change which customers are served, so
    capacity is unchanged. Same move as ``difusco.tsp.decoding.two_opt`` but
    without the closing edge from the last customer back to the first.
    """
    if len(route) < 3:
        return list(route)

    coords = node_coords.detach().cpu().numpy()
    route = list(route)
    n = len(route)
    best_distance = _open_route_distance(route, coords)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                new_distance = _open_route_distance(new_route, coords)
                if new_distance < best_distance - 1e-10:
                    route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break

    return route


def two_opt_cvrp(
    routes: list[list[int]],
    node_coords: torch.Tensor,
    max_iterations: int = 100,
) -> list[list[int]]:
    """Apply intra-route 2-opt to every route independently."""
    return [
        two_opt_route(route, node_coords, max_iterations=max_iterations)
        for route in routes
    ]


def decode_cvrp(
    heatmap: torch.Tensor,
    edge_index: torch.Tensor,
    node_coords: torch.Tensor,
    demands: torch.Tensor,
    capacity: int,
    *,
    use_2opt: bool = False,
    max_2opt_iterations: int = 100,
) -> list[list[int]]:
    """
    Heatmap decode + optional per-route 2-opt (same pipeline as TSP eval).

    The heatmap greedy pass fixes topology (degree, no customer cycles, capacity).
    2-opt only shortens each route geometrically without moving customers
    between routes.
    """
    routes = greedy_decode_cvrp(
        heatmap=heatmap,
        edge_index=edge_index,
        node_coords=node_coords,
        demands=demands,
        capacity=capacity,
    )
    if use_2opt:
        routes = two_opt_cvrp(
            routes, node_coords, max_iterations=max_2opt_iterations
        )
    return routes

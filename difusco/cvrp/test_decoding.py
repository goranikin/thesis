import torch

from difusco.cvrp.decoding import (
    compute_overcapacity_violation,
    compute_route_length,
    decode_cvrp,
    greedy_decode_cvrp,
    routes_to_canonical,
    two_opt_cvrp,
    two_opt_route,
)


def _add_undirected_edge(
    edges: list[tuple[int, int]], heat: list[float], u: int, v: int, total_score: float
) -> None:
    """
    Add both directed arcs for one undirected edge.
    The decoder sums both directions, so split score equally.
    """
    half = total_score / 2.0
    edges.append((u, v))
    heat.append(half)
    edges.append((v, u))
    heat.append(half)


def test_greedy_decode_cvrp_end_to_end_example() -> None:
    """
    Reproduces the walkthrough:
      (1,2), (3,4), (0,1), (0,3), (2,0), (4,0), (5,6), (0,5), (6,0), ...
    with demands [4, 5, 6, 3, 4, 7] and capacity 12.
    """
    # 7 nodes total: depot=0, customers=1..6.
    node_coords = torch.tensor(
        [
            [0.0, 0.0],  # depot
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [3.0, 1.0],
            [4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    demands = torch.tensor([0, 4, 5, 6, 3, 4, 7], dtype=torch.int64)
    capacity = 12

    # These values represent the final edge score used by the decoder:
    # score(u,v) = (A_uv + A_vu) / dist(u,v)
    ranked_scores = [
        ((1, 2), 9.2),
        ((3, 4), 8.7),
        ((0, 1), 8.5),
        ((0, 3), 8.1),
        ((2, 0), 7.9),
        ((4, 0), 7.6),
        ((5, 6), 7.2),
        ((0, 5), 6.9),
        ((6, 0), 6.5),
        # Extra edge that should be skipped because it would merge 1-2 and 3-4
        # into load 18 > capacity.
        ((2, 3), 6.1),
    ]

    edges: list[tuple[int, int]] = []
    heat: list[float] = []

    # Convert ranked "normalized" scores into raw heatmap values by multiplying
    # by distance so the decoder computes the same ordering.
    for (u, v), normalized_score in ranked_scores:
        dist = torch.dist(node_coords[u], node_coords[v]).item()
        raw_sum = normalized_score * (dist + 1e-8)
        _add_undirected_edge(edges, heat, u, v, raw_sum)

    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
    heatmap = torch.tensor(heat, dtype=torch.float32)

    routes = greedy_decode_cvrp(
        heatmap=heatmap,
        edge_index=edge_index,
        node_coords=node_coords,
        demands=demands,
        capacity=capacity,
    )

    # Route orientation/order can vary; canonicalize each route direction.
    normalized_routes = [tuple(sorted(route)) for route in routes]
    assert sorted(normalized_routes) == [(1, 2), (3, 4), (5, 6)]

    violating, overshoot = compute_overcapacity_violation(routes, demands, capacity)
    assert violating == 0
    assert overshoot == 0

    # Optional stable print/debug helper value for breakpoints.
    canonical = routes_to_canonical(routes)
    assert isinstance(canonical, str)


def test_two_opt_route_improves_zigzag() -> None:
    """A deliberately bad order should shorten after 2-opt."""
    node_coords = torch.tensor(
        [
            [0.0, 0.0],  # depot
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    bad_route = [1, 3, 2]  # detour via customer 3 before customer 2
    improved = two_opt_route(bad_route, node_coords, max_iterations=50)
    assert improved == [1, 2, 3]
    assert compute_route_length([improved], node_coords) < compute_route_length(
        [bad_route], node_coords
    )


def test_two_opt_cvrp_preserves_customers_and_capacity() -> None:
    node_coords = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    demands = torch.tensor([0, 4, 5, 6, 3], dtype=torch.int64)
    capacity = 12
    routes = [[3, 1], [4, 2]]
    optimized = two_opt_cvrp(routes, node_coords, max_iterations=50)

    assert sorted(c for r in optimized for c in r) == sorted(
        c for r in routes for c in r
    )
    violating, overshoot = compute_overcapacity_violation(
        optimized, demands, capacity
    )
    assert violating == 0
    assert overshoot == 0


def test_decode_cvrp_with_2opt_matches_greedy_then_refine() -> None:
    node_coords = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    demands = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
    capacity = 10
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]
    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
    heatmap = torch.ones(edge_index.shape[1], dtype=torch.float32)

    greedy_routes = greedy_decode_cvrp(
        heatmap, edge_index, node_coords, demands, capacity
    )
    decoded = decode_cvrp(
        heatmap,
        edge_index,
        node_coords,
        demands,
        capacity,
        use_2opt=True,
        max_2opt_iterations=50,
    )
    refined = two_opt_cvrp(greedy_routes, node_coords, max_iterations=50)

    assert sorted(c for r in decoded for c in r) == sorted(
        c for r in refined for c in r
    )
    assert compute_route_length(decoded, node_coords) <= compute_route_length(
        greedy_routes, node_coords
    ) + 1e-6


if __name__ == "__main__":
    test_greedy_decode_cvrp_end_to_end_example()
    test_two_opt_route_improves_zigzag()
    test_two_opt_cvrp_preserves_customers_and_capacity()
    test_decode_cvrp_with_2opt_matches_greedy_then_refine()

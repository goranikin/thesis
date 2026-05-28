import torch

from difusco.cvrp.decoding import (
    compute_overcapacity_violation,
    greedy_decode_cvrp,
    routes_to_canonical,
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


if __name__ == "__main__":
    test_greedy_decode_cvrp_end_to_end_example()

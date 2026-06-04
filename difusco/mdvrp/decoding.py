"""
Greedy decoder for MDVRP assignment heatmaps.

Given the per-edge probabilities over the customer-depot bipartite edges,
recover a per-customer depot assignment that respects per-depot capacity
(approximately).

Outputs:
    assignment: list[int]   length n_customers, value in [0, n_depots).

Algorithm:
    1. Build the (n_customers, n_depots) score matrix by averaging the two
       directed probabilities for each (customer, depot) pair.
    2. Process customers in descending order of confidence margin
       (top-1 score minus median score). High-margin customers get first pick
       of their preferred depot.
    3. For each customer, pick the highest-scoring depot whose remaining
       capacity is enough for that customer's demand. If no depot has enough
       capacity, fall back to the depot with the most remaining capacity
       (records a capacity violation).
"""

import numpy as np
import torch


def greedy_decode_mdvrp_assignment(
    heatmap: torch.Tensor,
    edge_index: torch.Tensor,
    n_customers: int,
    n_depots: int,
    demands: torch.Tensor,
    capacity: int,
    num_vehicles_per_depot: int,
    node_offset: int = 0,
    edge_mask: torch.Tensor | None = None,
) -> tuple[list[int], int]:
    """
    Args:
        heatmap:      (E,) edge probabilities.
        edge_index:   (2, E)  (in *global* indexing for batched super-graphs).
        n_customers:  N
        n_depots:     K
        demands:      (N,) per-customer demand (int).
        capacity:     vehicle capacity Q.
        num_vehicles_per_depot:  upper bound on routes per depot, used to
                                 compute per-depot capacity.
        node_offset:  add this to local indices to compare against ``edge_index``
                      when the model was run on a batched super-graph.
                      Local layout: depots [0, K), customers [K, K + N).
        edge_mask:    (E,) optional 1/0 mask for which edges are bipartite.
                      Customer-customer edges have mask=0 and are ignored.
    Returns:
        (assignment, num_overcapacity_assignments)
            assignment: list[int] of length n_customers (0-indexed depot).
            num_overcapacity_assignments: count of customers that were forced
                into an over-capacity depot because no feasible one existed.
    """
    edge_index_np = edge_index.detach().cpu().numpy()
    heatmap_np = heatmap.detach().cpu().numpy()
    demands_np = demands.detach().cpu().numpy().astype(np.int64)

    if edge_mask is not None:
        mask_np = edge_mask.detach().cpu().numpy()
    else:
        mask_np = np.ones(edge_index_np.shape[1], dtype=np.float32)

    # Local node ranges in this instance.
    depot_lo, depot_hi = node_offset, node_offset + n_depots
    cust_lo, cust_hi = node_offset + n_depots, node_offset + n_depots + n_customers

    scores = np.zeros((n_customers, n_depots), dtype=np.float64)
    counts = np.zeros((n_customers, n_depots), dtype=np.float64)

    for k in range(edge_index_np.shape[1]):
        if mask_np[k] <= 0:
            continue
        u, v = int(edge_index_np[0, k]), int(edge_index_np[1, k])

        if depot_lo <= u < depot_hi and cust_lo <= v < cust_hi:
            d = u - depot_lo
            c = v - cust_lo
        elif depot_lo <= v < depot_hi and cust_lo <= u < cust_hi:
            d = v - depot_lo
            c = u - cust_lo
        else:
            continue

        scores[c, d] += float(heatmap_np[k])
        counts[c, d] += 1.0

    scores = scores / np.maximum(counts, 1.0)

    # Confidence margin per customer = top-1 score minus median score.
    sorted_scores = np.sort(scores, axis=1)            # ascending
    top1 = sorted_scores[:, -1]
    median = sorted_scores[:, sorted_scores.shape[1] // 2]
    margin = top1 - median

    customer_order = np.argsort(-margin)                # descending by margin

    # Per-depot remaining capacity = Q * num_vehicles_per_depot.
    remaining_capacity = np.full(n_depots, capacity * num_vehicles_per_depot, dtype=np.int64)

    assignment = [-1] * n_customers
    overcap = 0

    for c in customer_order:
        c = int(c)
        d_order = np.argsort(-scores[c])               # depots by score, desc
        chosen = -1
        for d in d_order:
            d = int(d)
            if remaining_capacity[d] >= demands_np[c]:
                chosen = d
                break

        if chosen == -1:
            # No feasible depot — assign to the depot with the most slack and
            # record a capacity violation.
            chosen = int(np.argmax(remaining_capacity))
            overcap += 1

        assignment[c] = chosen
        remaining_capacity[chosen] -= int(demands_np[c])

    return assignment, overcap


def assignment_accuracy(
    pred_assignment: list[int], gt_assignment: torch.Tensor
) -> float:
    """
    Fraction of customers whose predicted depot equals the ground-truth
    depot (after any depot relabeling has been canonicalized upstream).
    """
    gt = gt_assignment.detach().cpu().numpy().astype(np.int64)
    pred = np.asarray(pred_assignment, dtype=np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")
    return float((pred == gt).mean())

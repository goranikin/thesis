import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cado.cvrp.models.model import CADOCVRP, _capacity_scalar, _demands_from_features
from difusco.cvrp.decoding import (
    compute_overcapacity_violation,
    compute_route_length,
    greedy_decode_cvrp,
)


@torch.no_grad()
def evaluate(
    model: CADOCVRP,
    val_loader: DataLoader,
    device: torch.device,
    *,
    num_inference_steps: int = 50,
    schedule_type: str = "cosine",
    max_instances: int = -1,
) -> tuple[float, float, float, float]:
    """
    Validate by decoding routes per instance (batch_size=1 required).

    Returns:
        (avg_pred_length, avg_gt_length, gap_pct, overcapacity_rate)
    """
    model.eval()

    total_pred = 0.0
    total_gt = 0.0
    total_violations = 0
    num_instances = 0

    total = (
        min(max_instances, len(val_loader))
        if max_instances > 0
        else len(val_loader)
    )
    pbar = tqdm(
        val_loader,
        desc="CADO CVRP eval",
        total=total,
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        node_feat, edge_index, edge_dist, edge_label, capacities = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

        if capacities.numel() != 1:
            raise ValueError(
                "evaluate() expects batch_size=1 (per-instance decoding), "
                f"got {capacities.numel()} graphs"
            )
        capacity = _capacity_scalar(capacities)
        coords = node_feat[:, :2]
        demands = _demands_from_features(node_feat, capacity)

        heatmap = model.generate(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            num_inference_steps=num_inference_steps,
            schedule_type=schedule_type,
        )

        routes = greedy_decode_cvrp(
            heatmap=heatmap,
            edge_index=edge_index,
            node_coords=coords,
            demands=demands,
            capacity=capacity,
        )

        pred_length = compute_route_length(routes, coords)
        gt_edges = edge_label.nonzero(as_tuple=True)[0]
        gt_length = edge_dist[gt_edges].sum().item() / 2.0
        n_violating, _ = compute_overcapacity_violation(routes, demands, capacity)

        total_pred += pred_length
        total_gt += gt_length
        total_violations += n_violating
        num_instances += 1

        avg_gap = ((total_pred - total_gt) / max(total_gt, 1e-9)) * 100
        pbar.set_postfix(gap=f"{avg_gap:.2f}%", n=num_instances)

        if max_instances > 0 and num_instances >= max_instances:
            break

    avg_pred = total_pred / max(num_instances, 1)
    avg_gt = total_gt / max(num_instances, 1)
    gap = (avg_pred - avg_gt) / max(avg_gt, 1e-9) * 100
    overcapacity_rate = total_violations / max(num_instances, 1)
    return avg_pred, avg_gt, gap, overcapacity_rate

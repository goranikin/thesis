"""
Evaluation helper for CADO.

Mirrors `Trainer.validate()` from `difusco/main/trainer.py` but lives here
so the CADO trainers don't have to depend on the DIFUSCO `Trainer` class.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cado.models.model import CADOTSP
from difusco.decoding import compute_tour_length, greedy_decode_tsp, two_opt


@torch.no_grad()
def evaluate(
    model: CADOTSP,
    val_loader: DataLoader,
    device: torch.device,
    *,
    num_inference_steps: int = 50,
    schedule_type: str = "cosine",
    use_2opt: bool = True,
    max_instances: int = -1,
) -> tuple[float, float, float]:
    """
    Run greedy decoding (+ optional 2-opt) on the val loader and report the
    mean optimality gap vs. ground-truth Concorde tours.

    Args:
        max_instances: cap on instances evaluated (-1 = all).

    Returns:
        (avg_pred_length, avg_gt_length, gap_pct)
    """
    model.eval()

    total_pred_length = 0.0
    total_gt_length = 0.0
    num_instances = 0

    total = (
        min(max_instances, len(val_loader))
        if max_instances > 0
        else len(val_loader)
    )
    pbar = tqdm(
        val_loader,
        desc="CADO eval",
        total=total,
        leave=False,
        dynamic_ncols=True,
    )
    for batch in pbar:
        node_feat, edge_index, edge_dist, edge_label = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

        heatmap = model.generate(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            num_inference_steps=num_inference_steps,
            schedule_type=schedule_type,
        )

        tour = greedy_decode_tsp(heatmap, edge_index, node_feat)
        if use_2opt:
            tour = two_opt(tour, node_feat, max_iterations=100)

        pred_length = compute_tour_length(tour, node_feat)
        gt_edges = edge_label.nonzero(as_tuple=True)[0]
        gt_length = edge_dist[gt_edges].sum().item() / 2.0

        total_pred_length += pred_length
        total_gt_length += gt_length
        num_instances += 1

        avg_gap = (
            (total_pred_length / num_instances - total_gt_length / num_instances)
            / (total_gt_length / num_instances)
            * 100
        )
        pbar.set_postfix(gap=f"{avg_gap:.2f}%")

        if max_instances > 0 and num_instances >= max_instances:
            break

    avg_pred = total_pred_length / max(num_instances, 1)
    avg_gt = total_gt_length / max(num_instances, 1)
    gap = (avg_pred - avg_gt) / avg_gt * 100
    return avg_pred, avg_gt, gap

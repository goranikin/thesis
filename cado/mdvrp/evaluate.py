import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cado.mdvrp.models.model import CADOMDVRP
from difusco.mdvrp.decoding import (
    assignment_accuracy,
    greedy_decode_mdvrp_assignment,
)


@torch.no_grad()
def evaluate(
    model: CADOMDVRP,
    val_loader: DataLoader,
    device: torch.device,
    *,
    num_inference_steps: int = 50,
    schedule_type: str = "cosine",
    max_instances: int = -1,
) -> tuple[float, float]:
    """
    Validate by decoding the per-customer assignment per instance.
    Returns (avg_assignment_accuracy, avg_capacity_violation_rate).
    """
    model.eval()
    total_acc = 0.0
    total_overcap = 0.0
    num_instances = 0

    total = (
        min(max_instances, len(val_loader))
        if max_instances > 0
        else len(val_loader)
    )
    pbar = tqdm(
        val_loader,
        desc="CADO MDVRP eval",
        total=total,
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        node_feat, edge_index, edge_dist, _edge_label, edge_mask, meta_list = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_mask = edge_mask.to(device)

        if len(meta_list) != 1:
            raise ValueError(
                f"evaluate() expects batch_size=1, got {len(meta_list)} graphs"
            )
        meta = meta_list[0]

        heatmap = model.generate(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            num_inference_steps=num_inference_steps,
            schedule_type=schedule_type,
        )

        pred, overcap = greedy_decode_mdvrp_assignment(
            heatmap=heatmap,
            edge_index=edge_index,
            n_customers=meta["n_customers"],
            n_depots=meta["n_depots"],
            demands=meta["demands"],
            capacity=meta["capacity"],
            num_vehicles_per_depot=meta["num_vehicles_per_depot"],
            node_offset=meta["node_offset"],
            edge_mask=edge_mask,
        )

        acc = assignment_accuracy(pred, meta["gt_assignment"])
        overcap_rate = overcap / max(meta["n_customers"], 1)

        total_acc += acc
        total_overcap += overcap_rate
        num_instances += 1

        pbar.set_postfix(
            acc=f"{total_acc / num_instances:.3f}",
            overcap=f"{total_overcap / num_instances:.3f}",
        )

        if max_instances > 0 and num_instances >= max_instances:
            break

    avg_acc = total_acc / max(num_instances, 1)
    avg_overcap = total_overcap / max(num_instances, 1)
    return avg_acc, avg_overcap

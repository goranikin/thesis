import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cado.models.model import CADOTSP
from difusco.tsp.decoding import compute_tour_length, greedy_decode_tsp, two_opt


def _graphs_in_batch(node_feat: torch.Tensor, num_nodes: int) -> int:
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if node_feat.shape[0] % num_nodes != 0:
        raise ValueError(
            f"node_feat rows ({node_feat.shape[0]}) not divisible by "
            f"num_nodes ({num_nodes})"
        )
    return node_feat.shape[0] // num_nodes


def _split_supergraph(
    node_feat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_dist: torch.Tensor,
    edge_label: torch.Tensor,
    heatmap: torch.Tensor,
    *,
    num_nodes: int,
    max_graphs: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split a collated super-graph back into per-instance tensors."""
    full_batch = _graphs_in_batch(node_feat, num_nodes)
    edges_per_graph = edge_index.shape[1] // full_batch
    batch_size = full_batch
    if max_graphs is not None:
        batch_size = min(batch_size, max_graphs)
    graphs: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []

    for i in range(batch_size):
        n0 = i * num_nodes
        n1 = n0 + num_nodes
        e0 = i * edges_per_graph
        e1 = e0 + edges_per_graph

        local_edge_index = edge_index[:, e0:e1] - n0
        graphs.append(
            (
                node_feat[n0:n1],
                local_edge_index,
                edge_dist[e0:e1],
                edge_label[e0:e1],
                heatmap[e0:e1],
            )
        )
    return graphs


def _eval_single_graph(
    heatmap: torch.Tensor,
    edge_index: torch.Tensor,
    node_feat: torch.Tensor,
    edge_dist: torch.Tensor,
    edge_label: torch.Tensor,
    *,
    use_2opt: bool,
    max_2opt_iterations: int,
) -> tuple[float, float]:
    tour = greedy_decode_tsp(heatmap, edge_index, node_feat)
    if use_2opt:
        tour = two_opt(tour, node_feat, max_iterations=max_2opt_iterations)

    pred_length = compute_tour_length(tour, node_feat)
    gt_edges = edge_label.nonzero(as_tuple=True)[0]
    gt_length = edge_dist[gt_edges].sum().item() / 2.0
    return pred_length, gt_length


@torch.no_grad()
def evaluate(
    model: CADOTSP,
    val_loader: DataLoader,
    device: torch.device,
    *,
    num_nodes: int,
    num_inference_steps: int = 50,
    schedule_type: str = "cosine",
    use_2opt: bool = True,
    max_2opt_iterations: int = 100,
    max_instances: int = -1,
) -> tuple[float, float, float]:
    """
    Run greedy decoding (+ optional 2-opt) on the val loader and report the
    mean optimality gap vs. ground-truth Concorde tours.

    Expects `val_loader` to use `collate_tsp` with `batch_size >= 1`. Each
    batch is denoised as one disconnected super-graph, then split for decode.

    Args:
        num_nodes: nodes per instance (all graphs in a batch must share this).
        max_instances: cap on instances evaluated (-1 = all).

    Returns:
        (avg_pred_length, avg_gt_length, gap_pct)
    """
    model.eval()

    total_pred_length = 0.0
    total_gt_length = 0.0
    num_instances = 0

    dataset_size = len(val_loader.dataset)  # type: ignore
    if max_instances > 0:
        total = min(max_instances, dataset_size)
    else:
        total = dataset_size

    pbar = tqdm(
        val_loader,
        desc="CADO eval",
        total=(total + val_loader.batch_size - 1) // val_loader.batch_size,  # type: ignore
        leave=False,
        dynamic_ncols=True,
    )
    for batch in pbar:
        node_feat, edge_index, edge_dist, edge_label = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

        remaining: int | None = None
        if max_instances > 0:
            remaining = max_instances - num_instances
            if remaining <= 0:
                break

        heatmap = model.generate(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            num_inference_steps=num_inference_steps,
            schedule_type=schedule_type,
        )

        graphs = _split_supergraph(
            node_feat,
            edge_index,
            edge_dist,
            edge_label,
            heatmap,
            num_nodes=num_nodes,
            max_graphs=remaining,
        )

        for nf, ei, ed, el, hm in graphs:
            pred_length, gt_length = _eval_single_graph(
                hm,
                ei,
                nf,
                ed,
                el,
                use_2opt=use_2opt,
                max_2opt_iterations=max_2opt_iterations,
            )
            total_pred_length += pred_length
            total_gt_length += gt_length
            num_instances += 1

            avg_gap = (
                (total_pred_length / num_instances - total_gt_length / num_instances)
                / (total_gt_length / num_instances)
                * 100
            )
            pbar.set_postfix(gap=f"{avg_gap:.2f}%", n=num_instances)

            if max_instances > 0 and num_instances >= max_instances:
                break

        if max_instances > 0 and num_instances >= max_instances:
            break

    avg_pred = total_pred_length / max(num_instances, 1)
    avg_gt = total_gt_length / max(num_instances, 1)
    gap = (avg_pred - avg_gt) / avg_gt * 100
    return avg_pred, avg_gt, gap

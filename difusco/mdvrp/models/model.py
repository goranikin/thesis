"""
DifuscoMDVRP — same diffusion training/inference loop as DifuscoCVRP, with
two differences:

  1. The training loss is masked (we only count bipartite customer-depot
     edges; if customer-customer k-NN edges are present they are excluded).
  2. The batch is a 6-tuple including ``edge_mask`` and a per-instance
     ``meta_list`` carrying capacities, demands, and the ground-truth
     assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from difusco.mdvrp.models.backbone import DifuscoMDVRPBackbone
from difusco.mdvrp.models.diffusion import (
    CategoricalDiffusion,
    InferenceSchedule,
)


class DifuscoMDVRP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 12,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.T = T
        self.backbone = DifuscoMDVRPBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.diffusion = CategoricalDiffusion(T, beta_start, beta_end)

    # ------------------------------------------------------------- training
    def training_step(self, batch, device):
        """
        Per-customer K-way softmax NLL.

        For each customer c, we have K candidate depot edges. The model's edge
        head outputs (E, 2) logits over {not-in-assignment, in-assignment}; we
        convert each (c, d) pair to a single scalar score and apply softmax
        across the K depots per customer, then NLL against the ground-truth
        depot index.

        Why per-customer K-way and not per-edge BCE:
            The argmax-decoded metric (val accuracy) depends on the *rank
            ordering of K depot scores per customer*, not on the absolute
            calibration of each edge as a binary classifier. Per-edge BCE
            on a 1-of-K target is dominated by the K−1 negative edges and
            can lower its loss by becoming more confident on negatives at
            the expense of separability — exactly the failure mode we saw
            in the first MDVRP run (train loss ↓ while val acc dropped
            82% → 56%).

        Layout assumption (set up in MDVRPDataset._build_graph):
            For each instance, the first ``2*N*K`` edges of its edge_index
            block are bipartite edges. The first ``N*K`` are the cust→depot
            direction, the next ``N*K`` are the depot→cust direction. Both
            blocks share the same logical (c, d) pair ordering:

                pair (c_i, d_j)  is at in-block index  i*K + j.

            So reshaping any per-pair quantity to (N, K) row-major gives
            customer i in row i and depot j in column j.

        ``meta_list`` carries ``n_bipartite_edges`` and ``n_total_edges``
        per instance so we can slice the super-graph correctly when
        ``customer_knn > 0`` adds context edges after the bipartite block.
        """
        node_feat, edge_index, edge_dist, edge_label, edge_mask, meta_list = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)
        edge_mask = edge_mask.to(device)

        t = torch.randint(0, self.T, (1,), device=device).long()

        x_t = self.diffusion.q_sample(edge_label, t)
        logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t.float())
        # log_softmax over the 2-class edge head, then take the "positive" log-prob
        # per directed edge. We work in log-prob space throughout for stability.
        edge_log_p1 = F.log_softmax(logits, dim=-1)[:, 1]  # (E_total,)

        per_instance_losses: list[torch.Tensor] = []
        edge_offset = 0
        for meta in meta_list:
            N = int(meta["n_customers"])
            K = int(meta["n_depots"])
            n_bipartite = int(meta["n_bipartite_edges"])
            n_total = int(meta["n_total_edges"])
            assert n_bipartite == 2 * N * K, (
                f"layout assumption broken: n_bipartite={n_bipartite}, 2*N*K={2*N*K}"
            )

            bipartite_logp = edge_log_p1[edge_offset : edge_offset + n_bipartite]
            # Average the two directional log-probs per logical (c, d) pair.
            # (Averaging in log-space ≡ geometric mean of probabilities;
            # for softmax-input purposes this is just as valid as arithmetic
            # averaging and keeps the gradient simple.)
            cust_to_depot = bipartite_logp[: N * K]            # (N*K,)
            depot_to_cust = bipartite_logp[N * K : 2 * N * K]  # (N*K,)
            pair_score = 0.5 * (cust_to_depot + depot_to_cust)  # (N*K,)

            # Per-customer rank scores over K depots. row i = customer i.
            scores_nk = pair_score.view(N, K)
            # Softmax across the K depots gives the per-customer distribution.
            log_p_depot = F.log_softmax(scores_nk, dim=-1)  # (N, K)

            gt = meta["gt_assignment"].to(device)  # (N,) in [0, K)
            nll = -log_p_depot.gather(1, gt.unsqueeze(1)).squeeze(1)  # (N,)
            per_instance_losses.append(nll.mean())

            edge_offset += n_total

        # Sanity check: we should have consumed every edge in the super-graph.
        assert edge_offset == edge_log_p1.shape[0], (
            f"edge_offset={edge_offset} != E_total={edge_log_p1.shape[0]}"
        )

        # ``edge_mask`` is unused by this loss (context edges are excluded by
        # the bipartite slicing). The raw ``edge_label`` was already consumed
        # by ``q_sample`` to produce ``x_t``. Both are intentionally not
        # consulted further.
        del edge_mask

        return torch.stack(per_instance_losses).mean()

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(
        self,
        device: torch.device,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        num_inference_steps: int = 50,
        schedule_type: str = "cosine",
    ) -> torch.Tensor:
        """
        Return a (E,) heatmap over the edge set. Customer-customer context
        edges (if present) will also receive probabilities, but the decoder
        will ignore them.
        """
        self.eval()

        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        E = edge_index.shape[1]

        timesteps = InferenceSchedule.get_schedule(
            schedule_type, num_inference_steps, self.T
        )

        x_t = torch.bernoulli(torch.ones(E, device=device) * 0.5)
        x_0_pred = x_t

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
            logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t_tensor)
            probs = F.softmax(logits, dim=-1)
            x_0_pred = probs[:, 1]

            if i == len(timesteps) - 1:
                return x_0_pred
            x_t = self.diffusion.q_posterior(x_t, x_0_pred, t)

        return x_0_pred

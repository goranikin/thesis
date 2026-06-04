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
        Masked cross-entropy on the bipartite customer-depot edges.

        batch: (node_feat, edge_index, edge_dist, edge_label, edge_mask, meta_list)
        """
        node_feat, edge_index, edge_dist, edge_label, edge_mask, _meta = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)
        edge_mask = edge_mask.to(device)

        t = torch.randint(0, self.T, (1,), device=device).long()

        x_t = self.diffusion.q_sample(edge_label, t)
        logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t.float())

        targets = edge_label.long()
        # Per-edge cross-entropy with the mask weighting (1.0 for bipartite, 0.0 for
        # customer-customer context edges).
        ce = F.cross_entropy(logits, targets, reduction="none")           # (E,)
        denom = edge_mask.sum().clamp(min=1.0)
        return (ce * edge_mask).sum() / denom

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

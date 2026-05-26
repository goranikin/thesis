"""
DifuscoCVRP — same diffusion training/inference loop as DifuscoTSP,
just wired to the CVRP backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from difusco.cvrp.models.backbone import DifuscoCVRPBackbone
from difusco.cvrp.models.diffusion import (
    CategoricalDiffusion,
    InferenceSchedule,
)


class DifuscoCVRP(nn.Module):
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
        self.backbone = DifuscoCVRPBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.diffusion = CategoricalDiffusion(T, beta_start, beta_end)

    # ------------------------------------------------------------- training
    def training_step(self, batch, device):
        """
        One training step over a batched super-graph from collate_cvrp.

        The batch is a 5-tuple
        ``(node_feat, edge_index, edge_dist, edge_label, capacities)``;
        per-instance capacities are not used during training and are simply
        ignored here. They are consumed by the trainer's validation loop
        (the decoder requires them for capacity-aware decoding).
        """
        node_feat, edge_index, edge_dist, edge_label, _capacities = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

        t = torch.randint(0, self.T, (1,), device=device).long()

        x_t = self.diffusion.q_sample(edge_label, t.item())
        logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t.float())

        targets = edge_label.long()
        return F.cross_entropy(logits, targets)

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
        Run iterative denoising and return a (E,) heatmap of edge probabilities.
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
        x_0_pred = x_t  # initialize so the variable is bound if timesteps is empty

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
            logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t_tensor)
            probs = F.softmax(logits, dim=-1)
            x_0_pred = probs[:, 1]

            if i == len(timesteps) - 1:
                return x_0_pred
            x_t = self.diffusion.q_posterior(x_t, x_0_pred, t)

        return x_0_pred

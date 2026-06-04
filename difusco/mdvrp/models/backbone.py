"""
MDVRP backbone — same AGNN layer as TSP/CVRP, just a different node encoder.

Node features: (x, y, demand/Q, role) where role flags customer vs depot.
Edge features: distance + noisy assignment label (Bernoulli diffusion).

The edge head is binary (2 logits per edge) because the prediction target
is still an edge label in {0, 1}: 1 iff the customer-depot pair is in the
ground-truth assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from difusco.mdvrp.models.embeddings import (
    MDVRPNodeEncoder,
    ScalarEmbeddingSine,
    timestep_embedding,
)
from difusco.tsp.models.backbone import AGNNLayer  # reused verbatim


class DifuscoMDVRPBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Node: 4 features -> hidden_dim
        # [x, y, demand/Q, role]
        # role is embedded by an embedding layer (0=depot, 1=customer)
        self.node_encoder = MDVRPNodeEncoder(hidden_dim)

        # Edge: distance (hidden_dim // 2) || noisy label (hidden_dim // 2) -> hidden_dim
        self.edge_dist_embed = ScalarEmbeddingSine(hidden_dim // 2)
        self.edge_noise_embed = ScalarEmbeddingSine(hidden_dim // 2)

        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers: nn.ModuleList = nn.ModuleList(
            [AGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Args:
            node_feat:      (N_total, 4)
            edge_index:     (2, E)
            edge_distances: (E,)
            x_t:            (E,) noisy edge labels in {0, 1}
            t:              (1,) scalar timestep
        Returns:
            (E, 2) logits
        """
        h = self.node_encoder(node_feat)                                  # (N_total, hidden)

        e_dist = self.edge_dist_embed(edge_distances)                     # (E, hidden/2)
        e_noise = self.edge_noise_embed(x_t)                              # (E, hidden/2)
        e = torch.cat([e_dist, e_noise], dim=-1)                          # (E, hidden)

        t_emb = timestep_embedding(t, self.hidden_dim)                    # (hidden,)
        t_emb = self.time_proj(t_emb.unsqueeze(0) if t_emb.dim() == 1 else t_emb)

        for layer in self.layers:
            h, e = layer(h, e, edge_index, t_emb)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
                e = F.dropout(e, p=self.dropout, training=True)

        return self.edge_head(e)                                          # (E, 2)

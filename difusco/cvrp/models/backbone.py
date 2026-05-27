"""
CVRP backbone.

Identical AGNN layer to the TSP backbone; the only differences are:
  - Node encoding: 4D (x, y, demand/Q, is_depot) -> hidden_dim via CVRPNodeEncoder
  - Edge encoding: same (edge distance + noisy label), unchanged

The edge head is unchanged because the prediction target is still binary
(edge-in-solution vs not).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from difusco.cvrp.models.embeddings import (
    CVRPNodeEncoder,
    ScalarEmbeddingSine,
    timestep_embedding,
)
from difusco.tsp.models.backbone import AGNNLayer  # reused verbatim


class DifuscoCVRPBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Node: 4 features -> hidden_dim
        # Here's the big difference from TSP!
        # [x, y, demand/Q, is_depot]
        # is_depot is embedded by an embedding layer
        self.node_encoder = CVRPNodeEncoder(hidden_dim)

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
            node_feat:      (N+1, 4)   per-node features [x, y, demand/Q, is_depot]
            edge_index:     (2, E)
            edge_distances: (E,)       Euclidean distance per edge
            x_t:            (E,)       current noisy edge labels in {0, 1}
            t:              (1,)       scalar timestep
        Returns:
            (E, 2) logits
        """
        h = self.node_encoder(node_feat)  # (N+1, hidden)

        e_dist = self.edge_dist_embed(edge_distances)  # (E, hidden/2)
        e_noise = self.edge_noise_embed(x_t)  # (E, hidden/2)
        e = torch.cat([e_dist, e_noise], dim=-1)  # (E, hidden)

        t_emb = timestep_embedding(t, self.hidden_dim)  # (hidden,)
        t_emb = self.time_proj(t_emb.unsqueeze(0) if t_emb.dim() == 1 else t_emb)

        for layer in self.layers:
            h, e = layer(h, e, edge_index, t_emb)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
                e = F.dropout(e, p=self.dropout, training=True)

        return self.edge_head(e)  # (E, 2)


# How do we know which edges belong to which vehicle's tour? I mean, can we separate the edges into different sets for each vehicle?
# 1. Nerual networks do not know how to assign edges to vehicles. It only outputs a binary label for each edge.
# 2. The recover process should partition the edges into different sets for each vehicle.
# Customer nodes:  degree = 2  (one in-edge, one out-edge, like TSP)
# Depot node:      degree = 2K (where K is the number of vehicles used)

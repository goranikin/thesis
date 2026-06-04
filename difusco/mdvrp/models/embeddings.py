"""
Per-node encoder for MDVRP.

Fuses position, demand, and a role flag (customer vs depot) into a single
``hidden_dim``-d vector. Mirrors CVRPNodeEncoder; only the role-flag
semantics change.
"""

import torch
import torch.nn as nn

from difusco.tsp.models.embeddings import (
    PositionEmbeddingSine2D,
    ScalarEmbeddingSine,
    sinusoidal_embedding,
    timestep_embedding,
)

__all__ = [
    "sinusoidal_embedding",
    "timestep_embedding",
    "PositionEmbeddingSine2D",
    "ScalarEmbeddingSine",
    "MDVRPNodeEncoder",
]


class MDVRPNodeEncoder(nn.Module):
    """
    Encode MDVRP node features (x, y, demand/Q, role) into a hidden vector.

    role: 0 = depot, 1 = customer (learned 2-vocab embedding).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.pos_embed = PositionEmbeddingSine2D(hidden_dim)
        self.demand_embed = ScalarEmbeddingSine(hidden_dim)
        self.role_embed = nn.Embedding(2, hidden_dim)

        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (N_total, 4) with columns [x, y, demand/Q, role].
        Returns:
            (N_total, hidden_dim) per-node embedding.
        """
        coords = node_feat[:, :2]
        demand = node_feat[:, 2]
        role = node_feat[:, 3].long()

        pos = self.pos_embed(coords)
        dem = self.demand_embed(demand)
        rol = self.role_embed(role)

        h = torch.cat([pos, dem, rol], dim=-1)
        return self.fuse(h)

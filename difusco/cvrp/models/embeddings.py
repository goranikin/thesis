"""
Embedding modules for CVRP.

We reuse the sinusoidal scalar / 2D positional embeddings from the TSP
package and add ``CVRPNodeEncoder``, which fuses position, demand, and the
depot indicator into a single hidden-dimensional vector per node.
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
    "CVRPNodeEncoder",
]


class CVRPNodeEncoder(nn.Module):
    """
    Encode CVRP node features (x, y, demand/Q, is_depot) into a single
    ``hidden_dim``-dimensional vector.

    Components:
        - Position:   PositionEmbeddingSine2D(hidden_dim)          -> hidden_dim
        - Demand:     ScalarEmbeddingSine(hidden_dim)              -> hidden_dim
        - Depot flag: nn.Embedding(2, hidden_dim) on ``is_depot``  -> hidden_dim

    These three vectors are concatenated and projected back down to hidden_dim:
        fuse: Linear(3 * hidden_dim, hidden_dim) -> SiLU -> Linear(hidden_dim, hidden_dim)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.pos_embed = PositionEmbeddingSine2D(hidden_dim)
        self.demand_embed = ScalarEmbeddingSine(hidden_dim)
        self.depot_embed = nn.Embedding(2, hidden_dim)

        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: (N+1, 4) with columns [x, y, demand/Q, is_depot].
        Returns:
            (N+1, hidden_dim) per-node embedding.
        """
        coords = node_feat[:, :2]                       # (N+1, 2)
        demand = node_feat[:, 2]                        # (N+1,)
        is_depot = node_feat[:, 3].long()               # (N+1,) in {0, 1}

        pos = self.pos_embed(coords)                    # (N+1, hidden_dim)
        dem = self.demand_embed(demand)                 # (N+1, hidden_dim)
        dep = self.depot_embed(is_depot)                # (N+1, hidden_dim)

        h = torch.cat([pos, dem, dep], dim=-1)          # (N+1, 3 * hidden_dim)
        return self.fuse(h)

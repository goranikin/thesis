import torch
import torch.nn.functional as F

from cado.models.diffusion import CADOCategoricalDiffusion
from difusco.cvrp.decoding import compute_route_length, greedy_decode_cvrp
from difusco.cvrp.models.diffusion import InferenceSchedule
from difusco.cvrp.models.model import DifuscoCVRP


def _capacity_scalar(capacity: torch.Tensor) -> int:
    return int(capacity.view(-1)[0].item())


def _demands_from_features(node_feat: torch.Tensor, capacity: int) -> torch.Tensor:
    return torch.round(node_feat[:, 2] * capacity).long()


class CADOCVRP(DifuscoCVRP):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 12,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim, num_layers, T, beta_start, beta_end, dropout)
        self.diffusion: CADOCategoricalDiffusion = CADOCategoricalDiffusion(
            T, beta_start, beta_end
        )

    def rollout(
        self,
        device: torch.device,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        num_inference_steps: int = 10,
        schedule_type: str = "cosine",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E = edge_index.shape[1]
        timesteps = InferenceSchedule.get_schedule(
            schedule_type, num_inference_steps, self.T
        )

        x_t = torch.bernoulli(torch.ones(E, device=device) * 0.5)
        log_probs: list[torch.Tensor] = []

        for i, t in enumerate(timesteps):
            logits = self.backbone(
                node_feat,
                edge_index,
                edge_dist,
                x_t,
                torch.tensor([t], device=device, dtype=torch.float32),
            )
            x_0_prob = F.softmax(logits, dim=-1)[:, 1]
            if i == len(timesteps) - 1:
                x_0 = (x_0_prob > 0.5).float()
                break
            x_t, logp = self.diffusion.q_posterior_with_logprob(x_t, x_0_prob, t)
            log_probs.append(logp)

        return x_0, torch.stack(log_probs)

    def compute_reward(
        self,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        node_feat: torch.Tensor,
        gt_length: float,
        capacity: int,
        mode: str,
    ) -> float:
        coords = node_feat[:, :2]
        demands = _demands_from_features(node_feat, capacity)
        routes = greedy_decode_cvrp(
            heatmap=x_0,
            edge_index=edge_index,
            node_coords=coords,
            demands=demands,
            capacity=capacity,
        )
        pred_length = compute_route_length(routes, coords)
        if mode == "SR":
            return -pred_length
        if mode == "LCR":
            return -(pred_length - gt_length)
        raise ValueError(f"Invalid reward mode: {mode}")

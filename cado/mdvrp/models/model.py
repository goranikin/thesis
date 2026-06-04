"""
CADOMDVRP — REINFORCE/PPO fine-tuning wrapper around DifuscoMDVRP.

Reward: assignment quality, expressed as the negative miss rate of the
greedily-decoded per-customer depot assignment vs. the ground-truth
assignment carried in the batch metadata.

The signature of ``compute_reward`` is kept open so a future cost-based
reward (decompose into K CVRPs, solve each with PyVRP, return -total-cost)
can be plugged in by overriding this method.
"""

import torch
import torch.nn.functional as F

from cado.models.diffusion import CADOCategoricalDiffusion
from difusco.mdvrp.decoding import (
    assignment_accuracy,
    greedy_decode_mdvrp_assignment,
)
from difusco.mdvrp.models.diffusion import InferenceSchedule
from difusco.mdvrp.models.model import DifuscoMDVRP


class CADOMDVRP(DifuscoMDVRP):
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

    # ------------------------------------------------------------- rollout
    def rollout(
        self,
        device: torch.device,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_mask: torch.Tensor,
        num_inference_steps: int = 10,
        schedule_type: str = "cosine",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a stochastic denoising trajectory and return:
            x_0       : (E,) final decoded edge labels in {0, 1}
            log_probs : (M-1,) per-step mean log-prob over bipartite edges
                         (customer-customer context edges are excluded so the
                         gradient signal is not diluted).

        Note: log-prob is averaged ONLY over bipartite edges where
        ``edge_mask > 0``. This keeps the per-step magnitude comparable to
        CADO-CVRP (which averages over all edges).
        """
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
            x_t, logp = self._masked_q_posterior_with_logprob(
                x_t, x_0_prob, t, edge_mask
            )
            log_probs.append(logp)

        return x_0, torch.stack(log_probs)

    def _masked_q_posterior_with_logprob(
        self,
        x_t: torch.Tensor,
        x_0_prob: torch.Tensor,
        t: int,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Wrap ``CADOCategoricalDiffusion.q_posterior_with_logprob`` but
        average the per-step log-prob only over the bipartite edges
        (``edge_mask > 0``) — customer-customer context edges contribute
        nothing to the policy-gradient signal.
        """
        sample, _ = self.diffusion.q_posterior_with_logprob(x_t, x_0_prob, t)
        # Re-derive per-edge log-prob with the masked-mean reduction.
        if t == 0:
            return sample, x_t.new_zeros((1,))
        beta_t = self.diffusion.betas[t].float().to(x_t.device)
        alpha_bar_tm1 = self.diffusion.alphas_cumprod[t - 1].float().to(x_t.device)

        p_xt_g_1 = x_t * (1 - beta_t) + (1 - x_t) * beta_t
        p_xt_g_0 = x_t * beta_t + (1 - x_t) * (1 - beta_t)
        p_xtm1_1 = (
            x_0_prob * (1 + alpha_bar_tm1) / 2
            + (1 - x_0_prob) * (1 - alpha_bar_tm1) / 2
        )
        p_xtm1_0 = 1.0 - p_xtm1_1
        prob_1 = p_xt_g_1 * p_xtm1_1 / (p_xt_g_1 * p_xtm1_1 + p_xt_g_0 * p_xtm1_0 + 1e-8)
        prob_1 = prob_1.clamp(1e-6, 1 - 1e-6)

        per_edge_logp = sample * torch.log(prob_1) + (1.0 - sample) * torch.log(1.0 - prob_1)
        masked_sum = (per_edge_logp * edge_mask).sum()
        denom = edge_mask.sum().clamp(min=1.0)
        return sample, masked_sum / denom

    # -------------------------------------------------------- reward
    def compute_reward(
        self,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        n_customers: int,
        n_depots: int,
        demands: torch.Tensor,
        capacity: int,
        num_vehicles_per_depot: int,
        gt_assignment: torch.Tensor,
        node_offset: int,
        mode: str,
    ) -> float:
        """
        Reward = -(miss rate) of the decoded assignment vs. ground truth.
        Higher is better (0.0 = perfect, -1.0 = worst).

        ``mode``:
            - "LCR" — return raw negative miss rate.
            - "SR"  — return the same raw value; batch-normalization is
                      handled by the outer trainer.
        """
        pred_assignment, _overcap = greedy_decode_mdvrp_assignment(
            heatmap=x_0,
            edge_index=edge_index,
            n_customers=n_customers,
            n_depots=n_depots,
            demands=demands,
            capacity=capacity,
            num_vehicles_per_depot=num_vehicles_per_depot,
            node_offset=node_offset,
            edge_mask=edge_mask,
        )
        accuracy = assignment_accuracy(pred_assignment, gt_assignment)
        miss_rate = 1.0 - accuracy

        if mode in ("LCR", "SR"):
            return -miss_rate
        raise ValueError(f"Invalid reward mode: {mode}")

import torch
import torch.nn.functional as F

from difusco.tsp.decoding import compute_tour_length, greedy_decode_tsp, two_opt
from difusco.tsp.models.diffusion import CategoricalDiffusion, InferenceSchedule
from difusco.tsp.models.model import DifuscoTSP


class CADOCategoricalDiffusion(CategoricalDiffusion):
    def __init__(self, T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__(T, beta_start, beta_end)

    def q_posterior_with_logprob(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: int,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same math as q_posterior, but returns (sample, log_prob) so REINFORCE/PPO
        can backprop through the policy.

        Args:
            x_t:      (E,) current noisy state
            x_0_pred: (E,) predicted p(x_0 = 1 | x_t) — KEEPS GRADIENTS
            t:        current timestep
            action:   if provided, use this as the sample (used by PPO to
                      recompute log_prob of a previously-collected action under
                      the updated policy). Otherwise, draw a fresh sample.

        Returns:
            sample:   (E,) {0, 1}, DETACHED (state transitions are not
                      differentiable; only log_prob carries gradients).
            log_prob: scalar, mean Bernoulli log-prob across edges. The mean
                      (not sum) is used so the magnitude does not scale with E.
        """
        if t == 0:
            # No stochasticity at the boundary
            return (x_0_pred > 0.5).float(), x_t.new_zeros((1,))

        beta_t = self.betas[t].float().to(x_t.device)
        alpha_bar_t_minus_1 = self.alphas_cumprod[t - 1].float().to(x_t.device)

        # Likelihood and prior — identical to q_posterior
        p_xt_given_xtm1_is_1 = x_t * (1 - beta_t) + (1 - x_t) * beta_t
        p_xt_given_xtm1_is_0 = x_t * beta_t + (1 - x_t) * (1 - beta_t)
        p_xtm1_is_1 = (
            x_0_pred * (1 + alpha_bar_t_minus_1) / 2
            + (1 - x_0_pred) * (1 - alpha_bar_t_minus_1) / 2
        )
        p_xtm1_is_0 = 1.0 - p_xtm1_is_1

        unnorm_1 = p_xt_given_xtm1_is_1 * p_xtm1_is_1
        unnorm_0 = p_xt_given_xtm1_is_0 * p_xtm1_is_0
        prob_1 = unnorm_1 / (unnorm_1 + unnorm_0 + 1e-8)

        # Clamp to keep log finite at boundary timesteps
        prob_1 = prob_1.clamp(1e-6, 1 - 1e-6)

        # Sample (or accept the supplied action). torch.bernoulli is already
        # non-differentiable, but .detach() makes the intent explicit.
        if action is None:
            sample = torch.bernoulli(prob_1).detach()
        else:
            sample = action.detach()

        # log p(sample) = sample * log(p1) + (1 - sample) * log(1 - p1)
        log_prob = (
            sample * torch.log(prob_1) + (1.0 - sample) * torch.log(1.0 - prob_1)
        ).mean()
        return sample, log_prob


class CADOTSP(DifuscoTSP):
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

    def _categorical_inference(
        self, node_feat, edge_index, edge_dist, E, timesteps, device
    ):
        """
        Reverse process for categorical diffusion.

        Start from x_T ~ Uniform({0,1}), iteratively denoise.
        """
        # Start from pure noise: x_T ~ Bernoulli(0.5) = Uniform({0,1})
        x_t = torch.bernoulli(torch.ones(E, device=device) * 0.5)

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)

            # Predict p(x_0 | x_t)
            logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t_tensor)
            probs = F.softmax(logits, dim=-1)  # (E, 2)
            x_0_pred = probs[:, 1]  # probability of class 1

            if i == len(timesteps) - 1:
                # Last step: return heatmap (don't sample)
                return x_0_pred
            else:
                # Sample x_{t-1} from posterior
                x_t = self.diffusion.q_posterior(x_t, x_0_pred, t)

        return x_0_pred

    def rollout(
        self,
        device: torch.device,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        num_inference_steps: int = 10,
        schedule_type: str = "cosine",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout the model for a given number of inference steps.
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
                torch.tensor([t], device=device, dtype=torch.float32).long(),
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
        mode: str,
        use_2opt: bool = True,
    ) -> float:
        """
        Decode x_0 -> tour -> length, then return a scalar reward.

        Modes (paper Section 4.2):
            "SR"  (Standard Reward): R = -length. Batch-normalize externally.
            "LCR" (Label-Centered Reward): R = -(length - gt_length).
                  Uses the ground-truth length as an unbiased baseline; do NOT
                  batch-normalize on top.
        """
        tour = greedy_decode_tsp(x_0, edge_index, node_feat)
        if use_2opt:
            tour = two_opt(tour, node_feat, max_iterations=100)

        pred_length = compute_tour_length(tour, node_feat)
        if mode == "SR":
            return -pred_length
        elif mode == "LCR":
            return -(pred_length - gt_length)
        else:
            raise ValueError(f"Invalid reward mode: {mode}")

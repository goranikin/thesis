"""
Categorical diffusion with log-prob for CADO policy-gradient training.

Shared by TSP and CVRP CADO models; the diffusion math is identical to
``difusco.*.models.diffusion.CategoricalDiffusion``.
"""

import torch

from difusco.tsp.models.diffusion import CategoricalDiffusion


class CADOCategoricalDiffusion(CategoricalDiffusion):
    def q_posterior_with_logprob(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: int,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same math as q_posterior, but returns (sample, log_prob) for REINFORCE/PPO.

        Returns:
            sample:   (E,) {0, 1}, detached
            log_prob: scalar mean Bernoulli log-prob across edges
        """
        if t == 0:
            return (x_0_pred > 0.5).float(), x_t.new_zeros((1,))

        beta_t = self.betas[t].float().to(x_t.device)
        alpha_bar_t_minus_1 = self.alphas_cumprod[t - 1].float().to(x_t.device)

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
        prob_1 = prob_1.clamp(1e-6, 1 - 1e-6)

        if action is None:
            sample = torch.bernoulli(prob_1).detach()
        else:
            sample = action.detach()

        log_prob = (
            sample * torch.log(prob_1) + (1.0 - sample) * torch.log(1.0 - prob_1)
        ).mean()
        return sample, log_prob

"""
Bernoulli (binary categorical) diffusion. Identical to the TSP version —
the diffusion machinery does not depend on what the edge label represents,
only that it is in {0, 1}. Re-exported here so the CVRP package is self-
contained and the import surface mirrors ``difusco.tsp.models.diffusion``.
"""

import math

import numpy as np
import torch


class CategoricalDiffusion:
    def __init__(self, T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T

        betas = np.linspace(beta_start, beta_end, T)
        self.betas = torch.tensor(betas, dtype=torch.float64)

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.tensor(np.cumprod(alphas), dtype=torch.float64)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[t.to("cpu")].float().to(x_0.device)
        prob_one = x_0 * (1.0 + alpha_bar) / 2.0 + (1.0 - x_0) * (1.0 - alpha_bar) / 2.0
        return torch.bernoulli(prob_one)

    def q_posterior(
        self, x_t: torch.Tensor, x_0_pred: torch.Tensor, t: int
    ) -> torch.Tensor:
        if t == 0:
            return (x_0_pred > 0.5).float()

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
        return torch.bernoulli(prob_1)


class InferenceSchedule:
    @staticmethod
    def get_schedule(schedule_type: str, num_inference_steps: int, T: int):
        if schedule_type == "linear":
            c = T / num_inference_steps
            timesteps = [int(c * i) for i in range(num_inference_steps, 0, -1)]
        elif schedule_type == "cosine":
            timesteps = []
            for i in range(num_inference_steps, 0, -1):
                c = i / num_inference_steps
                t = int(math.cos((1 - c) / 2 * math.pi) * T)
                t = max(0, min(T - 1, t))
                timesteps.append(t)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        return timesteps

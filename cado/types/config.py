"""
Shared CADO RL hyperparameters (TSP and CVRP).
"""

from typing import Literal

from difusco.tsp.types.base import Schema


class CADOConfig(Schema):
    """RL fine-tuning hyperparameters; used by both problem packages."""

    pretrained_ckpt: str = "checkpoints/best_model.pt"
    algorithm: Literal["reinforce", "ppo"] = "reinforce"
    reward_mode: Literal["LCR", "SR"] = "LCR"
    use_2opt_in_reward: bool = False

    lora_rank: int = 2
    selective_layers: int = 1

    lr: float = 1.0e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    M_train: int = 10
    M_eval: int = 50
    schedule_type: Literal["linear", "cosine"] = "cosine"
    init_noise: Literal["bernoulli_half"] = "bernoulli_half"

    epochs: int = 1000
    samples_per_epoch: int = 512
    batch_size: int = 32

    ppo_inner_epochs: int = 4
    ppo_clip_epsilon: float = 0.2

    log_interval: int = 4
    eval_every: int = 25
    eval_subset: int = 200
    eval_batch_size: int = 16

    save_best_only: bool = True
    ckpt_dir: str = "checkpoints/cado"

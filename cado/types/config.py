"""
Config schemas for CADO RL fine-tuning.

`CADORunConfig` mirrors `difusco.types.config.RunConfig` but adds a `cado`
section validated by `CADOConfig`. It re-uses the DIFUSCO `data`, `model`,
`diffusion`, and `wandb` sub-schemas so the same YAML composition rules
apply.
"""

from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from difusco.types.base import Schema
from difusco.types.config import (
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    WandbConfig,
)


class CADOConfig(Schema):
    """RL fine-tuning hyperparameters; matches `configs/cado_config.yaml`."""

    # Path to the SL-trained DIFUSCO checkpoint.
    pretrained_ckpt: str = "checkpoints/best_model.pt"

    # Algorithm choice.
    algorithm: Literal["reinforce", "ppo"] = "reinforce"

    # Reward.
    reward_mode: Literal["LCR", "SR"] = "LCR"
    use_2opt_in_reward: bool = False

    # Hybrid-FT (Section 4.3 of the paper).
    lora_rank: int = 2
    selective_layers: int = 1

    # Optimization.
    lr: float = 1.0e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Denoising rollout / schedule.
    M_train: int = 10
    M_eval: int = 50
    schedule_type: Literal["linear", "cosine"] = "cosine"
    init_noise: Literal["bernoulli_half"] = "bernoulli_half"

    # Outer training loop.
    epochs: int = 1000
    samples_per_epoch: int = 512
    batch_size: int = 32

    # PPO-only (ignored when algorithm == "reinforce").
    ppo_inner_epochs: int = 4
    ppo_clip_epsilon: float = 0.2

    # Logging / evaluation.
    log_interval: int = 4
    eval_every: int = 25
    eval_subset: int = 200

    # Checkpointing.
    save_best_only: bool = True
    ckpt_dir: str = "checkpoints/cado"


class CADORunConfig(Schema):
    """Top-level config for `cado/main/run_train.py`."""

    data_path: str = "./data/tsp50_128000_concorde.txt"
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    cado: CADOConfig = Field(default_factory=CADOConfig)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "CADORunConfig":
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict from Hydra config, got {type(raw)}")
        return cls.model_validate(raw)

    def wandb_run_name(self) -> str:
        if self.wandb.run_name:
            return self.wandb.run_name
        return (
            f"cado_{self.cado.algorithm}_tsp{self.data.num_nodes}"
            f"_h{self.model.hidden_dim}_L{self.model.num_layers}"
            f"_{self.cado.reward_mode}"
        )

    def wandb_config(self) -> dict:
        return self.model_dump()

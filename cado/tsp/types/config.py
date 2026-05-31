from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from cado.types.config import CADOConfig
from difusco.tsp.types.base import Schema
from difusco.tsp.types.config import (
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    WandbConfig,
)


class CADOTSPRunConfig(Schema):
    """Top-level config for ``cado.tsp.main.run_train``."""

    data_path: str = "./data/tsp50_128000_concorde.txt"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    cado: CADOConfig = Field(default_factory=CADOConfig)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "CADOTSPRunConfig":
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

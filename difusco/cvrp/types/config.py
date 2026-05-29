from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from difusco.cvrp.types.base import Schema


class ModelConfig(Schema):
    hidden_dim: int
    num_layers: int


class DiffusionConfig(Schema):
    T: int
    beta_start: float
    beta_end: float


class TrainingConfig(Schema):
    lr: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.0
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4
    train_val_split: float = 0.9
    log_interval: int = 100
    eval_every: int = 10


class DataConfig(Schema):
    """
    For CVRP, ``num_customers`` is the number of customers per instance.
    Total nodes in the graph is ``num_customers + 1`` (depot is index 0).

    ``sparse_factor`` is the k for the customer-k-NN graph. Each customer
    connects to its k nearest customers AND to the depot. The depot itself
    connects to every customer. Use ``-1`` for a dense (fully-connected) graph.
    """

    num_customers: int = 50
    sparse_factor: int = -1


class InferenceConfig(Schema):
    inference_steps: int = 50
    schedule: str = "cosine"
    use_2opt: bool = True
    max_2opt_iterations: int = 100
    eval_subset: int = 500


class WandbConfig(Schema):
    project: str = "difusco-cvrp"
    run_name: str | None = None
    mode: Literal["online", "offline", "disabled", "shared"] = "online"


class RunConfig(Schema):
    data_path: str = "./data/cvrp50-50_128000_pyvrp.txt"
    checkpoint_dir: str = "./checkpoints"
    model: ModelConfig = Field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "RunConfig":
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict from Hydra config, got {type(raw)}")
        return cls.model_validate(raw)

    def wandb_run_name(self) -> str:
        if self.wandb.run_name:
            return self.wandb.run_name
        return (
            f"cvrp{self.data.num_customers}_h{self.model.hidden_dim}"
            f"_L{self.model.num_layers}"
        )

    def wandb_config(self) -> dict:
        return self.model_dump()

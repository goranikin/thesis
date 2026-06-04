from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from difusco.mdvrp.types.base import Schema


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
    eval_every: int = 5


class DataConfig(Schema):
    """
    MDVRP graph configuration.

    Each instance has ``num_customers`` customers and a *variable* number of
    depots in ``[min_depots, max_depots]``. The total node count of the graph
    is therefore ``num_customers + K`` where K is per-instance. The edge set
    is the full customer-depot bipartite graph plus (optionally) a
    customer-customer k-NN augmentation purely for message-passing context.
    """

    num_customers: int = 50
    min_depots: int = 2
    max_depots: int = 5
    # If > 0, add customer-customer k-NN edges (label=0, type-flagged) so
    # nearby customers can directly exchange messages during diffusion. The
    # heatmap is still only decoded from customer-depot edges.
    customer_knn: int = 0


class InferenceConfig(Schema):
    inference_steps: int = 50
    schedule: str = "cosine"
    # Sub-sample of val instances during in-training validation (full eval
    # in run_eval.py).
    eval_subset: int = 500


class WandbConfig(Schema):
    project: str = "difusco-mdvrp"
    run_name: str | None = None
    mode: Literal["online", "offline", "disabled", "shared"] = "online"


class RunConfig(Schema):
    data_path: str = "./data/mdvrp50_pyvrp.txt"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_tag: str = "difusco_mdvrp"
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
            f"mdvrp{self.data.num_customers}"
            f"_D{self.data.min_depots}-{self.data.max_depots}"
            f"_h{self.model.hidden_dim}_L{self.model.num_layers}"
        )

    def wandb_config(self) -> dict:
        return self.model_dump()

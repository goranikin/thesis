from typing import Any, Literal, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, model_validator

from data_generation.types.base import Schema
from data_generation.types.constants import (
    DEFAULT_CVRP_GENERATION,
    DEFAULT_MDVRP_GENERATION,
    DEFAULT_TSP_GENERATION,
)

DepotPositioning = Literal["center", "eccentric", "random", "quadrant"]
CustomerPositioning = Literal["random", "clustered", "random-clustered"]
DemandDistribution = Literal["U", "CV", "Q", "SL"]
TspSolverName = Literal["concorde", "lkh"]


def _merge_hydra_config(cfg: DictConfig, defaults: dict[str, Any]) -> dict[str, Any]:
    OmegaConf.set_struct(cfg, False)
    merged = OmegaConf.merge(OmegaConf.create(defaults), cfg)
    raw = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(raw, dict):
        msg = f"Expected dict from Hydra config, got {type(raw)}"
        raise TypeError(msg)
    return cast(dict[str, Any], raw)


class BatchGenerationConfig(Schema):
    num_samples: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    seed: int
    filename: str | None

    @model_validator(mode="after")
    def batch_size_divides_num_samples(self) -> "BatchGenerationConfig":
        if self.num_samples % self.batch_size != 0:
            msg = "num_samples must be divisible by batch_size"
            raise ValueError(msg)
        return self

    @property
    def num_batches(self) -> int:
        return self.num_samples // self.batch_size


class NodeRangeConfig(Schema):
    min_nodes: int = Field(ge=1)
    max_nodes: int = Field(ge=1)

    @model_validator(mode="after")
    def min_nodes_le_max_nodes(self) -> "NodeRangeConfig":
        if self.min_nodes > self.max_nodes:
            raise ValueError("min_nodes must be <= max_nodes")
        return self

    def sample_num_nodes(self) -> int:
        return int(np.random.randint(self.min_nodes, self.max_nodes + 1))


class XInstanceGeneratorConfig(Schema):
    depot_positioning: DepotPositioning
    customer_positioning: CustomerPositioning
    demand_distribution: DemandDistribution


class TspSolverConfig(Schema):
    solver: TspSolverName
    lkh_trials: int = Field(ge=1)


class TspGenerationConfig(BatchGenerationConfig, NodeRangeConfig, TspSolverConfig):
    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "TspGenerationConfig":
        return cls.model_validate(_merge_hydra_config(cfg, DEFAULT_TSP_GENERATION))

    @property
    def output_path(self) -> str:
        if self.filename is not None:
            return self.filename
        return (
            f"tsp{self.min_nodes}-{self.max_nodes}_{self.num_samples}_{self.solver}.txt"
        )


class VrpSolverConfig(Schema):
    solver_runtime: float = Field(gt=0)


class CvrpGenerationConfig(
    BatchGenerationConfig,
    NodeRangeConfig,
    XInstanceGeneratorConfig,
    VrpSolverConfig,
):
    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "CvrpGenerationConfig":
        return cls.model_validate(_merge_hydra_config(cfg, DEFAULT_CVRP_GENERATION))

    @property
    def output_path(self) -> str:
        if self.filename is not None:
            return self.filename
        return f"cvrp{self.min_nodes}-{self.max_nodes}_{self.num_samples}_pyvrp.txt"


class MdvrpDepotRangeConfig(Schema):
    min_depots: int = Field(ge=2)
    max_depots: int = Field(ge=2)

    @model_validator(mode="after")
    def min_depots_le_max_depots(self) -> "MdvrpDepotRangeConfig":
        if self.min_depots > self.max_depots:
            raise ValueError("min_depots must be <= max_depots")
        return self


class MdvrpGenerationConfig(
    BatchGenerationConfig,
    MdvrpDepotRangeConfig,
    VrpSolverConfig,
):
    min_customers_per_depot: int = Field(ge=1)
    max_customers_per_depot: int = Field(ge=1)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "MdvrpGenerationConfig":
        return cls.model_validate(_merge_hydra_config(cfg, DEFAULT_MDVRP_GENERATION))

    @model_validator(mode="after")
    def customer_range_valid(self) -> "MdvrpGenerationConfig":
        if self.min_customers_per_depot > self.max_customers_per_depot:
            raise ValueError(
                "min_customers_per_depot must be <= max_customers_per_depot"
            )
        return self

    @property
    def customer_range(self) -> tuple[int, int]:
        return (self.min_customers_per_depot, self.max_customers_per_depot)

    @property
    def output_path(self) -> str:
        if self.filename is not None:
            return self.filename
        return (
            f"mdvrp{self.min_customers_per_depot}-"
            f"{self.max_customers_per_depot}x"
            f"{self.min_depots}-{self.max_depots}_{self.num_samples}_pyvrp.txt"
        )

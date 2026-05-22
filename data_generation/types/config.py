from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from data_generation.types.base import Schema

DepotPositioning = Literal["center", "eccentric", "random", "quadrant"]
CustomerPositioning = Literal["random", "clustered", "random-clustered"]
DemandDistribution = Literal["U", "CV", "Q", "SL"]
TspSolverName = Literal["concorde", "lkh"]


class BatchGenerationConfig(Schema):
    num_samples: Annotated[int, Field(ge=1)] = 128_000
    batch_size: Annotated[int, Field(ge=1)] = 128
    seed: int = 1234
    filename: str | None = None

    @model_validator(mode="after")
    def batch_size_divides_num_samples(self) -> BatchGenerationConfig:
        if self.num_samples % self.batch_size != 0:
            msg = "num_samples must be divisible by batch_size"
            raise ValueError(msg)
        return self

    @property
    def num_batches(self) -> int:
        return self.num_samples // self.batch_size


class NodeRangeConfig(Schema):
    min_nodes: Annotated[int, Field(ge=1)] = 50
    max_nodes: Annotated[int, Field(ge=1)] = 50

    @model_validator(mode="after")
    def min_nodes_le_max_nodes(self) -> NodeRangeConfig:
        if self.min_nodes > self.max_nodes:
            raise ValueError("min_nodes must be <= max_nodes")
        return self

    def sample_num_nodes(self) -> int:
        return int(np.random.randint(self.min_nodes, self.max_nodes + 1))


class XInstanceGeneratorConfig(Schema):
    depot_positioning: DepotPositioning = "random"
    customer_positioning: CustomerPositioning = "random"
    demand_distribution: DemandDistribution = "CV"

    def to_generator_kwargs(self) -> dict[str, str]:
        return {
            "depot_positioning": self.depot_positioning,
            "customer_positioning": self.customer_positioning,
            "demand_distribution": self.demand_distribution,
        }


class TspSolverConfig(Schema):
    solver: TspSolverName = "concorde"
    lkh_trials: Annotated[int, Field(ge=1)] = 1000


class TspGenerationConfig(BatchGenerationConfig, NodeRangeConfig, TspSolverConfig):
    @property
    def output_path(self) -> str:
        if self.filename is not None:
            return self.filename
        return f"tsp{self.min_nodes}-{self.max_nodes}_{self.solver}.txt"


class VrpSolverConfig(Schema):
    solver_runtime: Annotated[float, Field(gt=0)] = 5.0


class CvrpGenerationConfig(
    BatchGenerationConfig,
    NodeRangeConfig,
    XInstanceGeneratorConfig,
    VrpSolverConfig,
):
    batch_size: Annotated[int, Field(ge=1)] = 128

    @property
    def output_path(self) -> str:
        if self.filename is not None:
            return self.filename
        return f"cvrp{self.min_nodes}-{self.max_nodes}_pyvrp.txt"


class MdvrpDepotRangeConfig(Schema):
    min_depots: Annotated[int, Field(ge=2)] = 2
    max_depots: Annotated[int, Field(ge=2)] = 10

    @model_validator(mode="after")
    def min_depots_le_max_depots(self) -> MdvrpDepotRangeConfig:
        if self.min_depots > self.max_depots:
            raise ValueError("min_depots must be <= max_depots")
        return self


class MdvrpGenerationConfig(
    BatchGenerationConfig,
    MdvrpDepotRangeConfig,
    VrpSolverConfig,
):
    min_customers_per_depot: Annotated[int, Field(ge=1)] = 10
    max_customers_per_depot: Annotated[int, Field(ge=1)] = 20
    batch_size: Annotated[int, Field(ge=1)] = 64

    @model_validator(mode="after")
    def customer_range_valid(self) -> MdvrpGenerationConfig:
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
            f"{self.min_depots}-{self.max_depots}_pyvrp.txt"
        )

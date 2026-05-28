"""Shared fixtures for CVRP training smoke tests."""

from pathlib import Path

import pytest

import wandb
from data_generation.types.dataset import (
    Coordinate,
    CvrpInstance,
    CvrpSample,
    VrpRoutes,
)
from difusco.cvrp.types import (
    DataConfig,
    DiffusionConfig,
    InferenceConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
    WandbConfig,
)


def _make_cvrp_line(
    coords: list[tuple[float, float]],
    demands: list[int],
    capacity: int,
    routes: list[list[int]],
    num_vehicles: int | None = None,
) -> str:
    """Build one dataset line (depot + len(coords)-1 customers). Routes are 1-indexed customer ids."""
    nodes = [Coordinate.from_pair(xy) for xy in coords]
    sample = CvrpSample(
        instance=CvrpInstance(
            nodes=nodes,
            demands=demands,
            vehicle_capacity=capacity,
            num_vehicles=num_vehicles or max(len(routes), 1),
        ),
        routes=VrpRoutes(routes=routes),
    )
    return sample.to_line().strip()


def _tiny_instances() -> list[str]:
    """Eight 3-customer instances — small enough for a fast local pilot run."""
    base_coords = [
        (100.0, 100.0),  # depot
        (300.0, 100.0),
        (100.0, 400.0),
        (400.0, 400.0),
    ]
    variants = [
        ([0, 4, 5, 3], 12, [[2, 3], [4]]),
        ([0, 3, 4, 5], 12, [[2], [3, 4]]),
        ([0, 5, 3, 4], 12, [[2, 3, 4]]),
        ([0, 2, 6, 4], 12, [[2, 4], [3]]),
        ([0, 4, 4, 4], 12, [[2], [3], [4]]),
        ([0, 3, 3, 6], 12, [[2, 3], [4]]),
        ([0, 5, 5, 2], 12, [[2], [3, 4]]),
        ([0, 2, 3, 7], 12, [[2, 3], [4]]),
    ]
    lines: list[str] = []
    for i, (demands, capacity, routes) in enumerate(variants):
        offset = float(i * 25)
        coords = [(x + offset, y + offset) for x, y in base_coords]
        lines.append(_make_cvrp_line(coords, demands, capacity, routes))
    return lines


@pytest.fixture(autouse=True)
def _wandb_disabled():
    wandb.init(mode="disabled", reinit=True)
    yield  # type: ignore[misc]
    wandb.finish(quiet=True)


@pytest.fixture
def tiny_cvrp_data_path(tmp_path: Path) -> Path:
    path = tmp_path / "cvrp_pilot.txt"
    path.write_text("\n".join(_tiny_instances()) + "\n")
    return path


@pytest.fixture
def pilot_config(tiny_cvrp_data_path: Path) -> RunConfig:
    """Minimal config mirroring run_train.py but sized for a laptop smoke test."""
    return RunConfig(
        data_path=str(tiny_cvrp_data_path),
        model=ModelConfig(hidden_dim=32, num_layers=2),
        diffusion=DiffusionConfig(T=50, beta_start=1e-4, beta_end=0.02),
        training=TrainingConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            train_val_split=0.75,
            log_interval=1,
            eval_every=1,
        ),
        data=DataConfig(num_customers=3, sparse_factor=-1),
        inference=InferenceConfig(
            inference_steps=3,
            schedule="cosine",
            eval_subset=2,
        ),
        wandb=WandbConfig(mode="disabled"),
    )

"""
Smoke tests for CVRP training (pilot before a remote run).

Run all tests:
    uv run pytest difusco/cvrp/test_train.py -v

Run only the end-to-end pilot (closest to run_train.py):
    uv run pytest difusco/cvrp/test_train.py::test_pilot_fit -v
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, random_split

from difusco.cvrp.conftest import _tiny_instances
from difusco.cvrp.dataset import CVRPDataset, collate_cvrp
from difusco.cvrp.main.trainer import Trainer
from difusco.cvrp.models.model import DifuscoCVRP
from difusco.cvrp.types import RunConfig


def _build_loaders(cfg: RunConfig) -> tuple[DataLoader, DataLoader]:
    dataset = CVRPDataset(
        file_path=cfg.data_path,
        num_customers=cfg.data.num_customers,
        sparse_factor=cfg.data.sparse_factor,
    )
    n_train = int(cfg.training.train_val_split * len(dataset))
    n_val = len(dataset) - n_train
    torch.manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_cvrp,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_cvrp,
        num_workers=0,
    )
    return train_loader, val_loader


def _build_trainer(cfg: RunConfig, device: torch.device) -> Trainer:
    model = DifuscoCVRP(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        T=cfg.diffusion.T,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        dropout=cfg.training.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )
    return Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        log_interval=cfg.training.log_interval,
    )


def test_tiny_dataset_loads(tiny_cvrp_data_path: Path) -> None:
    dataset = CVRPDataset(
        file_path=str(tiny_cvrp_data_path),
        num_customers=3,
        sparse_factor=-1,
    )
    assert len(dataset) == len(_tiny_instances())
    node_feat, edge_index, edge_dist, edge_label, capacity = dataset[0]
    assert node_feat.shape == (4, 4)
    assert edge_index.shape[0] == 2
    assert edge_dist.shape == edge_label.shape
    assert capacity.numel() == 1


def test_training_step_finite(pilot_config: RunConfig) -> None:
    device = torch.device("cpu")
    train_loader, _ = _build_loaders(pilot_config)
    trainer = _build_trainer(pilot_config, device)

    batch = next(iter(train_loader))
    trainer.model.train()
    loss = trainer.model.training_step(batch, device)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_train_one_epoch(pilot_config: RunConfig) -> None:
    device = torch.device("cpu")
    train_loader, _ = _build_loaders(pilot_config)
    trainer = _build_trainer(pilot_config, device)

    avg_loss = trainer.train_one_epoch(train_loader, epoch=1)

    assert avg_loss > 0


def test_validate_runs(pilot_config: RunConfig) -> None:
    device = torch.device("cpu")
    _, val_loader = _build_loaders(pilot_config)
    trainer = _build_trainer(pilot_config, device)

    pred_len, gt_len, gap, avg_routes, overcap = trainer.validate(
        val_loader,
        num_inference_steps=pilot_config.inference.inference_steps,
        schedule_type=pilot_config.inference.schedule,
        max_instances=pilot_config.inference.eval_subset,
    )

    assert pred_len >= 0
    assert gt_len > 0
    assert gap == pytest.approx(gap)
    assert avg_routes >= 1
    assert 0 <= overcap <= 1


def test_pilot_fit(pilot_config: RunConfig, tmp_path: Path) -> None:
    """End-to-end pilot: same wiring as run_train.py, tiny data and 2 epochs."""
    device = torch.device("cpu")
    train_loader, val_loader = _build_loaders(pilot_config)
    trainer = _build_trainer(pilot_config, device)

    best_path = tmp_path / "best_model.pt"
    last_path = tmp_path / "last_model.pt"

    result = trainer.fit(
        config=pilot_config,
        train_loader=train_loader,
        val_loader=val_loader,
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
    )

    assert best_path.is_file()
    assert last_path.is_file()
    assert len(result.history) == pilot_config.training.epochs
    assert all(r.gap is not None for r in result.history)
    assert result.best_gap == pytest.approx(result.best_gap)

    ckpt = torch.load(best_path, weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt["config"]["data"]["num_customers"] == 3

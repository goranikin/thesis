import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

import wandb
from difusco.dataset import TSPDataset, collate_tsp
from difusco.main.trainer import Trainer
from difusco.models.model import DifuscoTSP
from difusco.types import RunConfig
from difusco.types.training import FitResult
from utils import select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@hydra.main(
    version_base=None,
    config_path=str(_CONFIG_DIR),
    config_name="config",
)
def main(hydra_cfg: DictConfig) -> None:
    cfg: RunConfig = RunConfig.from_hydra(hydra_cfg)

    device: torch.device = select_device()
    logger.info(f"Device: {device}")

    dataset: TSPDataset = TSPDataset(
        file_path=cfg.data_path,
        num_nodes=cfg.data.num_nodes,
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
        collate_fn=collate_tsp,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_tsp,
        num_workers=0,
    )

    model = DifuscoTSP(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        T=cfg.diffusion.T,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        dropout=cfg.training.dropout,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: DIFUSCO")
    logger.info(f"  Layers: {cfg.model.num_layers}, Hidden: {cfg.model.hidden_dim}")
    logger.info(f"  Parameters: {num_params:,}")
    logger.info(
        f"  T={cfg.diffusion.T}, β=[{cfg.diffusion.beta_start}, {cfg.diffusion.beta_end}]"
    )
    logger.info(
        f"  Inference: {cfg.inference.inference_steps} steps, {cfg.inference.schedule} schedule"
    )

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb_run_name(),
        mode=cfg.wandb.mode,
        config=cfg.wandb_config(),
    )
    wandb.watch(model, log="gradients", log_freq=500)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        log_interval=cfg.training.log_interval,
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Training for {cfg.training.epochs} epochs")
    logger.info("=" * 60)

    ckpt_dir = (
        Path(os.getcwd()) / "checkpoints" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Checkpoints: {ckpt_dir}")

    result: FitResult = trainer.fit(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        best_checkpoint_path=ckpt_dir / "best_model.pt",
        last_checkpoint_path=ckpt_dir / "last_model.pt",
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Best Gap: {result.best_gap:.2f}%")
    logger.info(f"  Checkpoints saved to: {ckpt_dir}")
    logger.info("=" * 60)

    wandb.log({"val/final_best_gap_pct": result.best_gap})
    wandb.finish()


# uv run python -m difusco.main.run_train data_path=data/tsp50_128000_concorde.txt model=small
if __name__ == "__main__":
    main()

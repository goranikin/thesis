"""
MDVRP training entrypoint. Mirrors difusco.cvrp.main.run_train.

uv run python -m difusco.mdvrp.main.run_train \\
    data_path=data/mdvrp50_pyvrp.txt model=small
"""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

import wandb
from difusco.mdvrp.dataset import MDVRPDataset, collate_mdvrp
from difusco.mdvrp.main.trainer import Trainer
from difusco.mdvrp.models.model import DifuscoMDVRP
from difusco.mdvrp.types import RunConfig
from difusco.mdvrp.types.training import FitResult
from utils import best_model_path, last_model_path, run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@hydra.main(
    version_base=None,
    config_path=str(_CONFIG_DIR),
    config_name="difusco_mdvrp_config",
)
def main(hydra_cfg: DictConfig) -> None:
    cfg: RunConfig = RunConfig.from_hydra(hydra_cfg)

    device: torch.device = select_device()
    logger.info(f"Device: {device}")

    data_path = Path(get_original_cwd()) / cfg.data_path
    dataset = MDVRPDataset(
        file_path=str(data_path),
        num_customers=cfg.data.num_customers,
        min_depots=cfg.data.min_depots,
        max_depots=cfg.data.max_depots,
        customer_knn=cfg.data.customer_knn,
    )
    n_train = int(cfg.training.train_val_split * len(dataset))
    n_val = len(dataset) - n_train

    torch.manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_mdvrp,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,  # decoder runs per instance
        shuffle=False,
        collate_fn=collate_mdvrp,
        num_workers=0,
    )

    model = DifuscoMDVRP(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        T=cfg.diffusion.T,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        dropout=cfg.training.dropout,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: DIFUSCO-MDVRP")
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

    wandb.define_metric("train/global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss_step", step_metric="train/global_step")
    wandb.define_metric("train/grad_norm", step_metric="train/global_step")
    wandb.define_metric("train/loss_epoch", step_metric="epoch")
    wandb.define_metric("train/lr", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

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

    project_root = Path(get_original_cwd())
    ckpt_dir = run_dir(
        cfg.checkpoint_dir, cfg.checkpoint_tag, cwd=project_root, mkdir=True
    )
    wandb.config.update(
        {
            "checkpoint_run": ckpt_dir.name,
            "checkpoint_run_dir": str(ckpt_dir),
            "checkpoint_tag": cfg.checkpoint_tag,
        }
    )
    logger.info(f"  Checkpoints: {ckpt_dir}")

    result: FitResult = trainer.fit(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        best_checkpoint_path=best_model_path(ckpt_dir),
        last_checkpoint_path=last_model_path(ckpt_dir),
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Best Accuracy: {result.best_accuracy:.3f}")
    logger.info(f"  Checkpoints saved to: {ckpt_dir}")
    logger.info("=" * 60)

    wandb.log(
        {
            "val/final_best_accuracy": result.best_accuracy,
            "epoch": cfg.training.epochs,
        }
    )
    wandb.summary["val/final_best_accuracy"] = result.best_accuracy
    wandb.summary["checkpoint_run"] = ckpt_dir.name
    wandb.summary["checkpoint_best_path"] = str(best_model_path(ckpt_dir))
    wandb.finish()


if __name__ == "__main__":
    main()

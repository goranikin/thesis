"""
CADO training entrypoint.

Loads a pretrained DIFUSCO checkpoint, applies Hybrid Fine-Tuning (LoRA on
early layers + full retraining on the last layers), and dispatches to either
the REINFORCE trainer (paper-faithful, Eq. 9) or the PPO trainer (more
sample-efficient, DDPO-style).

Run:
    uv run python -m cado.main.run_train \
        cado.pretrained_ckpt=checkpoints/best_model.pt \
        cado.algorithm=reinforce
"""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

import wandb
from cado.main.train_ppo import train_ppo
from cado.main.train_rf import train_reinforce
from cado.models.lora import apply_hybrid_ft, trainable_parameter_summary
from cado.models.model import CADOTSP
from cado.types import CADORunConfig
from difusco.tsp.dataset import TSPDataset, collate_tsp
from utils import select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loaders(
    cfg: CADORunConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val loaders.

    For RL, every gradient update consumes one *instance* (not a graph batch);
    we set batch_size=1 here and collect `cado.batch_size` instances inside
    the trainer. This keeps the per-instance reward computation simple.
    """
    dataset = TSPDataset(
        file_path=cfg.data_path,
        num_nodes=cfg.data.num_nodes,
        sparse_factor=cfg.data.sparse_factor,
    )
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train

    torch.manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=1,  # RL trainer accumulates `cado.batch_size` instances itself
        shuffle=True,
        collate_fn=collate_tsp,
        num_workers=0,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.cado.eval_batch_size,
        shuffle=False,
        collate_fn=collate_tsp,
        num_workers=0,
    )
    return train_loader, val_loader


def _load_pretrained(model: CADOTSP, ckpt_path: str | Path) -> dict:
    """Load the SL-trained DIFUSCO checkpoint into a fresh CADOTSP."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {ckpt_path}. "
            "Train a DIFUSCO baseline first with `python -m difusco.main.run_train`."
        )
    logger.info(f"Loading SL checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning(
            f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        logger.warning(
            f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    return ckpt


def _setup_wandb_metrics() -> None:
    """
    Declare custom x-axes BEFORE any wandb.log call. This is the same fix
    we applied to difusco/main/run_train.py — without it, train and val
    metrics would silently collide on the internal step counter.
    """
    wandb.define_metric("train/global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("val/*", step_metric="epoch")


@hydra.main(
    version_base=None,
    config_path=str(_CONFIG_DIR),
    config_name="cado_config",
)
def main(hydra_cfg: DictConfig) -> None:
    cfg: CADORunConfig = CADORunConfig.from_hydra(hydra_cfg)
    _set_seed(cfg.seed)

    device = select_device()
    logger.info(f"Device: {device}")
    logger.info(f"Algorithm: {cfg.cado.algorithm}, Reward: {cfg.cado.reward_mode}")

    # ---------------------------- Data ---------------------------- #
    train_loader, val_loader = _build_loaders(cfg)

    # ---------------------------- Model --------------------------- #
    model = CADOTSP(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        T=cfg.diffusion.T,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        dropout=0.0,  # dropout off during RL fine-tuning
    )
    _load_pretrained(
        model=model,
        ckpt_path=cfg.cado.pretrained_ckpt,
    )
    apply_hybrid_ft(
        model=model,
        lora_rank=cfg.cado.lora_rank,
        num_selective=cfg.cado.selective_layers,
    )
    model = model.to(device)

    summary = trainable_parameter_summary(model)
    logger.info(
        "Hybrid-FT — trainable: %s / total: %s (%.2f%%)",
        f"{summary['trainable']:,}",
        f"{summary['total']:,}",
        100 * summary["trainable"] / summary["total"],
    )

    # -------------------------- Optimizer ------------------------- #
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.cado.lr,
        weight_decay=cfg.cado.weight_decay,
    )

    # ---------------------------- Wandb --------------------------- #
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb_run_name(),
        mode=cfg.wandb.mode,
        config=cfg.wandb_config(),
    )
    _setup_wandb_metrics()

    # -------------------------- Checkpoints ----------------------- #
    ckpt_dir = (
        Path(os.getcwd()) / cfg.cado.ckpt_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"
    logger.info(f"Checkpoints: {ckpt_dir}")

    # --------------------------- Train ---------------------------- #
    logger.info("=" * 60)
    logger.info(f"  CADO {cfg.cado.algorithm.upper()} fine-tuning")
    logger.info(
        f"  Epochs: {cfg.cado.epochs}, samples/epoch: {cfg.cado.samples_per_epoch}"
    )
    logger.info(f"  Batch: {cfg.cado.batch_size}, M_train: {cfg.cado.M_train}")
    logger.info(
        f"  Eval: subset={cfg.cado.eval_subset}, batch={cfg.cado.eval_batch_size}, "
        f"M_eval={cfg.cado.M_eval}"
    )
    logger.info("=" * 60)

    if cfg.cado.algorithm == "reinforce":
        best_gap = train_reinforce(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            ckpt_path=ckpt_path,
        )
    elif cfg.cado.algorithm == "ppo":
        best_gap = train_ppo(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            ckpt_path=ckpt_path,
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.cado.algorithm}")

    logger.info("=" * 60)
    logger.info(f"  Best Gap: {best_gap:.2f}%")
    logger.info(f"  Best checkpoint: {ckpt_path}")
    logger.info("=" * 60)

    wandb.summary["val/best_gap_pct"] = best_gap
    wandb.finish()


# uv run python -m cado.main.run_train
if __name__ == "__main__":
    main()

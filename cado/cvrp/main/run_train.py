"""
CADO-CVRP training entrypoint.

Run:
    uv run python -m cado.cvrp.main.run_train \\
        cado.algorithm=reinforce

Checkpoints are saved under
``{checkpoint_dir}/cado_cvrp_YYYYMMDD_HHMMSS/best_model.pt``.
"""

import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

import wandb
from cado.cvrp.main.train_ppo import train_ppo
from cado.cvrp.main.train_rf import train_reinforce
from cado.cvrp.models.model import CADOCVRP
from cado.cvrp.types import CADOCVRPRunConfig
from cado.models.lora import apply_hybrid_ft, trainable_parameter_summary
from difusco.cvrp.dataset import CVRPDataset, collate_cvrp
from utils import CADO_CVRP, best_model_path, resolve_pretrained, run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loaders(cfg: CADOCVRPRunConfig) -> tuple[DataLoader, DataLoader]:
    data_path = Path(get_original_cwd()) / cfg.data_path
    dataset = CVRPDataset(
        file_path=str(data_path),
        num_customers=cfg.data.num_customers,
        sparse_factor=cfg.data.sparse_factor,
    )
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train

    torch.manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_cvrp,
        num_workers=0,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_cvrp,
        num_workers=0,
    )
    return train_loader, val_loader


def _load_pretrained(model: CADOCVRP, ckpt_path: Path) -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {ckpt_path}. "
            "Train a DIFUSCO-CVRP baseline first with "
            "`python -m difusco.cvrp.main.run_train`."
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
    wandb.define_metric("train/global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("val/*", step_metric="epoch")


@hydra.main(
    version_base=None,
    config_path=str(_CONFIG_DIR),
    config_name="cado_cvrp_config",
)
def main(hydra_cfg: DictConfig) -> None:
    cfg: CADOCVRPRunConfig = CADOCVRPRunConfig.from_hydra(hydra_cfg)
    _set_seed(cfg.seed)

    device = select_device()
    logger.info(f"Device: {device}")
    logger.info(f"Algorithm: {cfg.cado.algorithm}, Reward: {cfg.cado.reward_mode}")

    train_loader, val_loader = _build_loaders(cfg)

    model = CADOCVRP(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        T=cfg.diffusion.T,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        dropout=0.0,
    )
    project_root = Path(get_original_cwd())
    pretrained_path = resolve_pretrained(
        cfg.checkpoint_dir,
        cfg.cado.pretrained_tag,
        cfg.cado.pretrained_ckpt,
        cwd=project_root,
    )
    _load_pretrained(model=model, ckpt_path=pretrained_path)
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.cado.lr,
        weight_decay=cfg.cado.weight_decay,
    )

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb_run_name(),
        mode=cfg.wandb.mode,
        config=cfg.wandb_config(),
    )
    _setup_wandb_metrics()

    ckpt_dir = run_dir(cfg.checkpoint_dir, CADO_CVRP, cwd=project_root, mkdir=True)
    ckpt_path = best_model_path(ckpt_dir)
    wandb.config.update(
        {
            "checkpoint_run": ckpt_dir.name,
            "checkpoint_run_dir": str(ckpt_dir),
            "pretrained_ckpt": str(pretrained_path),
        }
    )
    logger.info(f"Checkpoints: {ckpt_dir}")

    logger.info("=" * 60)
    logger.info(f"  CADO-CVRP {cfg.cado.algorithm.upper()} fine-tuning")
    logger.info(
        f"  Epochs: {cfg.cado.epochs}, samples/epoch: {cfg.cado.samples_per_epoch}"
    )
    logger.info(f"  Batch: {cfg.cado.batch_size}, M_train: {cfg.cado.M_train}")
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
    wandb.summary["checkpoint_run"] = ckpt_dir.name
    wandb.summary["checkpoint_best_path"] = str(ckpt_path)
    wandb.finish()


if __name__ == "__main__":
    main()

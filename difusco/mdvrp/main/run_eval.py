"""
Evaluate a trained DIFUSCO MDVRP model on a test file.

uv run python -m difusco.mdvrp.main.run_eval \\
    --checkpoint best_model.pt \\
    --data data/mdvrp50_pyvrp.txt
"""

import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader

from difusco.mdvrp.dataset import MDVRPDataset, collate_mdvrp
from difusco.mdvrp.main.trainer import Trainer
from difusco.mdvrp.models.model import DifuscoMDVRP
from difusco.mdvrp.types import RunConfig
from utils import select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_run_config(ckpt: dict) -> RunConfig:
    return RunConfig.model_validate(ckpt.get("config") or {})


def _build_model_from_ckpt(ckpt: dict, overrides: dict) -> DifuscoMDVRP:
    kwargs = _load_run_config(ckpt).model_dump(exclude_none=True)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    logger.info("Model config:")
    for k, v in kwargs.items():
        logger.info(f"  {k}: {v}")
    return DifuscoMDVRP(
        hidden_dim=kwargs["hidden_dim"],
        num_layers=kwargs["num_layers"],
        T=kwargs["T"],
        beta_start=kwargs["beta_start"],
        beta_end=kwargs["beta_end"],
        dropout=kwargs["dropout"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DIFUSCO MDVRP model on a test set."
    )
    parser.add_argument("--checkpoint", "-c", default="best_model.pt")
    parser.add_argument("--data", "-d", default="data/mdvrp50_pyvrp.txt")
    parser.add_argument("--num-customers", type=int, default=None)
    parser.add_argument("--min-depots", type=int, default=2)
    parser.add_argument("--max-depots", type=int, default=5)
    parser.add_argument("--customer-knn", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--max-instances", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = select_device()
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg = _load_run_config(ckpt)
    num_customers = args.num_customers or cfg.data.num_customers
    logger.info(
        f"Test set: {args.data}  (num_customers={num_customers}, "
        f"depots in [{args.min_depots}, {args.max_depots}])"
    )

    dataset = MDVRPDataset(
        args.data,
        num_customers=num_customers,
        min_depots=args.min_depots,
        max_depots=args.max_depots,
        customer_knn=args.customer_knn,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_mdvrp,
        num_workers=args.num_workers,
    )

    model = _build_model_from_ckpt(ckpt, overrides={})
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"  missing keys: {missing}")
    if unexpected:
        logger.warning(f"  unexpected keys: {unexpected}")
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {num_params:,}")
    if "epoch" in ckpt:
        logger.info(f"  Trained epoch: {ckpt['epoch']}")
    if "best_accuracy" in ckpt:
        logger.info(f"  Reported best accuracy (val): {ckpt['best_accuracy']:.4f}")

    logger.info(
        f"\nInference: steps={args.inference_steps}, schedule={args.schedule}"
    )

    trainer = Trainer(model, device=device)
    t0 = time.time()
    acc, overcap, _ = trainer.validate(
        loader,
        num_inference_steps=args.inference_steps,
        schedule_type=args.schedule,
        max_instances=args.max_instances,
    )
    elapsed = time.time() - t0

    n_eval = (
        min(args.max_instances, len(dataset))
        if args.max_instances > 0
        else len(dataset)
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("  Test Results")
    logger.info("=" * 60)
    logger.info(f"  Instances evaluated      : {n_eval}")
    logger.info(f"  Assignment accuracy      : {acc:.4f}")
    logger.info(f"  Capacity-violation rate  : {overcap:.4f}")
    logger.info(
        f"  Wall time                : {elapsed:.1f}s "
        f"({elapsed / max(n_eval, 1):.3f}s/inst)"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

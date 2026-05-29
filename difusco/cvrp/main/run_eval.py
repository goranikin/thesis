"""
Evaluate a trained DIFUSCO CVRP model on a test file.

uv run python -m difusco.cvrp.main.run_eval \\
    --checkpoint best_model.pt \\
    --data data/cvrp50-50_128000_pyvrp.txt

uv run python -m difusco.cvrp.main.run_eval \\
    --checkpoint best_model.pt \\
    --data data/cvrp50-50_128000_pyvrp.txt \\
    --inference-steps 50 --schedule cosine --no-2opt --max-instances 128
"""

import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader

from difusco.cvrp.dataset import CVRPDataset, collate_cvrp
from difusco.cvrp.main.trainer import Trainer
from difusco.cvrp.models.model import DifuscoCVRP
from difusco.cvrp.types import RunConfig
from utils import select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_run_config(ckpt: dict) -> RunConfig:
    return RunConfig.model_validate(ckpt.get("config") or {})


def _build_model_from_ckpt(ckpt: dict, overrides: dict) -> DifuscoCVRP:
    kwargs = _load_run_config(ckpt).model_dump(exclude_none=True)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    logger.info("Model config:")
    for k, v in kwargs.items():
        logger.info(f"  {k}: {v}")
    return DifuscoCVRP(
        hidden_dim=kwargs["hidden_dim"],
        num_layers=kwargs["num_layers"],
        T=kwargs["T"],
        beta_start=kwargs["beta_start"],
        beta_end=kwargs["beta_end"],
        dropout=kwargs["dropout"],
    )


def _infer_num_customers(ckpt: dict, cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    return _load_run_config(ckpt).data.num_customers or 50


def _infer_sparse_factor(ckpt: dict, cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    return _load_run_config(ckpt).data.sparse_factor or -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DIFUSCO CVRP model on a test set."
    )
    parser.add_argument("--checkpoint", "-c", default="best_model.pt")
    parser.add_argument("--data", "-d", default="data/cvrp50-50_128000_pyvrp.txt")
    parser.add_argument("--num-customers", type=int, default=None)
    parser.add_argument("--sparse-factor", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument(
        "--no-2opt",
        dest="use_2opt",
        action="store_false",
        help="Disable intra-route 2-opt after greedy decoding.",
    )
    parser.set_defaults(use_2opt=True)
    parser.add_argument(
        "--max-2opt-iterations",
        type=int,
        default=100,
        help="Max 2-opt passes per route when 2-opt is enabled.",
    )
    parser.add_argument("--max-instances", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = select_device()
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_customers = _infer_num_customers(ckpt, args.num_customers)
    sparse_factor = _infer_sparse_factor(ckpt, args.sparse_factor)
    logger.info(
        f"Test set: {args.data}  (num_customers={num_customers}, "
        f"sparse_factor={sparse_factor})"
    )

    dataset = CVRPDataset(
        args.data, num_customers=num_customers, sparse_factor=sparse_factor
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_cvrp,
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
    if "best_gap" in ckpt:
        logger.info(f"  Reported best gap (val): {ckpt['best_gap']:.4f}%")

    logger.info(
        f"\nInference: steps={args.inference_steps}, schedule={args.schedule}, "
        f"2-opt={args.use_2opt} (max_iter={args.max_2opt_iterations})"
    )

    trainer = Trainer(model, device=device)
    t0 = time.time()
    (
        pred_len,
        gt_len,
        gap,
        avg_routes,
        overcap_rate,
    ) = trainer.validate(
        loader,
        num_inference_steps=args.inference_steps,
        schedule_type=args.schedule,
        use_2opt=args.use_2opt,
        max_2opt_iterations=args.max_2opt_iterations,
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
    logger.info(f"  Instances evaluated : {n_eval}")
    logger.info(f"  Predicted tour len  : {pred_len:.4f}")
    logger.info(f"  Ground-truth len    : {gt_len:.4f}")
    logger.info(f"  Optimality gap      : {gap:.4f}%")
    logger.info(f"  Avg routes (pred)   : {avg_routes:.2f}")
    logger.info(f"  Overcapacity rate   : {overcap_rate:.2%}")
    logger.info(
        f"  Wall time           : {elapsed:.1f}s ({elapsed / max(n_eval, 1):.3f}s/inst)"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

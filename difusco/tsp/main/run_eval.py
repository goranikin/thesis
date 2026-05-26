"""
uv run python -m difusco.main.run_eval \
    --checkpoint best_model.pt \
    --data data/tsp-50-1280-concorde.txt

uv run python -m difusco.main.run_eval \
    --checkpoint best_model.pt \
    --data data/tsp-50-1280-concorde.txt \
    --inference-steps 50 --schedule cosine --no-2opt --max-instances 128
"""

import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader

from difusco.tsp.dataset import TSPDataset, collate_tsp
from difusco.tsp.main.trainer import Trainer
from difusco.tsp.models.model import DifuscoTSP
from difusco.tsp.types import RunConfig
from utils import select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_run_config(ckpt: dict) -> RunConfig:
    return RunConfig.model_validate(ckpt.get("config") or {})


def _build_model_from_ckpt(ckpt: dict, overrides: dict) -> DifuscoTSP:
    """
    Reconstruct DifuscoTSP using the config saved inside the checkpoint, with
    optional manual overrides. Falls back to paper defaults if no config was
    saved (older checkpoints).
    """
    kwargs = _load_run_config(ckpt).model_dump(exclude_none=True)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    logger.info("Model config:")
    for k, v in kwargs.items():
        logger.info(f"  {k}: {v}")
    return DifuscoTSP(
        hidden_dim=kwargs["hidden_dim"],
        num_layers=kwargs["num_layers"],
        T=kwargs["T"],
        beta_start=kwargs["beta_start"],
        beta_end=kwargs["beta_end"],
        dropout=kwargs["dropout"],
    )


def _infer_num_nodes(ckpt: dict, cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    return _load_run_config(ckpt).data.num_nodes or 50


def _infer_sparse_factor(ckpt: dict, cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    return _load_run_config(ckpt).data.sparse_factor or -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DIFUSCO TSP model on a test set."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="best_model.pt",
        help="Path to model checkpoint (.pt). Defaults to ./best_model.pt.",
    )
    parser.add_argument(
        "--data",
        "-d",
        default="data/tsp-50-1280-concorde.txt",
        help="Path to TSP test file (Concorde format).",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="TSP problem size. Inferred from checkpoint config if omitted.",
    )
    parser.add_argument(
        "--sparse-factor",
        type=int,
        default=None,
        help="KNN sparse factor (-1 = dense). Inferred from checkpoint if omitted.",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps at inference time.",
    )
    parser.add_argument(
        "--schedule",
        choices=["linear", "cosine"],
        default="cosine",
        help="Inference timestep schedule.",
    )
    parser.add_argument(
        "--no-2opt",
        dest="use_2opt",
        action="store_false",
        help="Disable 2-opt local search post-processing.",
    )
    parser.set_defaults(use_2opt=True)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=-1,
        help="Evaluate on at most this many instances (-1 = all).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible diffusion sampling.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = select_device()
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_nodes = _infer_num_nodes(ckpt, args.num_nodes)
    sparse_factor = _infer_sparse_factor(ckpt, args.sparse_factor)
    logger.info(
        f"Test set: {args.data}  (num_nodes={num_nodes}, sparse_factor={sparse_factor})"
    )

    dataset = TSPDataset(args.data, num_nodes=100, sparse_factor=sparse_factor)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_tsp,
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
        f"\nInference: steps={args.inference_steps}, "
        f"schedule={args.schedule}, 2-opt={args.use_2opt}"
    )

    trainer = Trainer(model, device=device)
    t0 = time.time()
    pred_len, gt_len, gap = trainer.validate(
        loader,
        num_inference_steps=args.inference_steps,
        schedule_type=args.schedule,
        use_2opt=args.use_2opt,
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
    logger.info(
        f"  Wall time           : {elapsed:.1f}s ({elapsed / max(n_eval, 1):.3f}s/instance)"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

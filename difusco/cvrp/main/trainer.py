"""
Training and validation loops for DifuscoCVRP.

The training loop is identical to DIFUSCO TSP — cross-entropy on the
denoising target. Validation differs because we have to:
  - decode K routes (not a single Hamiltonian cycle)
  - compute total tour length across all routes
  - report capacity-violation rate

The validation loader yields BATCH_SIZE=1 graphs, exactly like the TSP
trainer, so we can recover ``demands`` and ``capacity`` directly from the
per-instance node features (column 2 is demand/Q, so demand = round(col2 * Q)
— but we don't actually need to round, because the heatmap decoder only
needs the raw demand values; we instead pass them through the dataloader by
attaching them to the batch tuple).
"""

import logging
import time
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from difusco.cvrp.decoding import (
    compute_overcapacity_violation,
    compute_route_length,
    greedy_decode_cvrp,
)
from difusco.cvrp.models.model import DifuscoCVRP
from difusco.cvrp.types import EpochRecord, FitResult, RunConfig
from difusco.cvrp.types.config import InferenceConfig, TrainingConfig
from utils import select_device

logger = logging.getLogger(__name__)


class Trainer:
    """Training and validation loops for DifuscoCVRP."""

    def __init__(
        self,
        model: DifuscoCVRP,
        device: torch.device | None = None,
        optimizer: Optimizer | None = None,
        lr_scheduler: LRScheduler | None = None,
        log_interval: int = 100,
        grad_clip_max_norm: float = 1.0,
    ):
        self.device = device or select_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_interval = log_interval
        self.grad_clip_max_norm = grad_clip_max_norm

    # ----------------------------------------------------------------- train
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        if self.optimizer is None:
            raise ValueError("optimizer is required for training")

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d} [train]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            loss = self.model.training_step(batch, self.device)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_max_norm
            )

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}", grad=f"{grad_norm:.3f}")

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            if batch_idx % self.log_interval == 0:
                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/global_step": global_step,
                    }
                )

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------- validate
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        num_inference_steps: int = 10,
        schedule_type: str = "cosine",
        max_instances: int = -1,
    ) -> tuple[float, float, float, float, float]:
        """
        Validate by decoding K routes per instance and comparing total length
        against the ground-truth multi-route solution.

        ``val_loader`` MUST use ``batch_size=1`` (and ``collate_cvrp``); the
        decoder runs per-instance because capacity varies per instance.

        Returns:
            (avg_pred_length, avg_gt_length, gap_pct,
             avg_num_routes_pred, overcapacity_rate)
        """
        self.model.eval()
        total_pred = 0.0
        total_gt = 0.0
        total_routes_pred = 0
        total_routes_gt = 0
        total_violations = 0
        num_instances = 0

        total = (
            min(max_instances, len(val_loader))
            if max_instances > 0
            else len(val_loader)
        )
        pbar = tqdm(
            val_loader,
            desc="Validating",
            total=total,
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            node_feat, edge_index, edge_dist, edge_label, capacities = batch
            node_feat = node_feat.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_dist = edge_dist.to(self.device)
            edge_label = edge_label.to(self.device)

            # capacities is a 1-D LongTensor of length B; we require B=1.
            if capacities.numel() != 1:
                raise ValueError(
                    f"validate() expects batch_size=1 (per-instance decoding), "
                    f"got {capacities.numel()} graphs in batch"
                )
            capacity = int(capacities[0].item())

            # Recover demands from the dataset features.
            # node_feat[:, 2] = demand / capacity; multiply back and round.
            coords = node_feat[:, :2]
            demands = torch.round(node_feat[:, 2] * capacity).long()

            heatmap = self.model.generate(
                device=self.device,
                node_feat=node_feat,
                edge_index=edge_index,
                edge_dist=edge_dist,
                num_inference_steps=num_inference_steps,
                schedule_type=schedule_type,
            )

            routes = greedy_decode_cvrp(
                heatmap=heatmap,
                edge_index=edge_index,
                node_coords=coords,
                demands=demands,
                capacity=capacity,
            )

            pred_length = compute_route_length(routes, coords)

            # Ground-truth length from edge labels — each tour edge is counted
            # in both directions in the (directed) edge_index, so divide by 2.
            gt_tour_edges = edge_label.nonzero(as_tuple=True)[0]
            gt_length = edge_dist[gt_tour_edges].sum().item() / 2.0

            n_violating, _ = compute_overcapacity_violation(
                routes, demands, capacity
            )

            total_pred += pred_length
            total_gt += gt_length
            total_routes_pred += len(routes)
            # GT route count: every positive depot edge ends a route, /2 because
            # depot edges are symmetric in the directed edge_index.
            depot_mask = (edge_index[0] == 0) | (edge_index[1] == 0)
            total_routes_gt += int(edge_label[depot_mask].sum().item() // 2)
            total_violations += n_violating
            num_instances += 1

            avg_gap = ((total_pred - total_gt) / max(total_gt, 1e-9)) * 100
            pbar.set_postfix(
                gap=f"{avg_gap:.2f}%",
                routes=f"{total_routes_pred / max(num_instances, 1):.1f}",
            )

            if max_instances > 0 and num_instances >= max_instances:
                break

        avg_pred = total_pred / max(num_instances, 1)
        avg_gt = total_gt / max(num_instances, 1)
        gap = (avg_pred - avg_gt) / max(avg_gt, 1e-9) * 100
        avg_routes_pred = total_routes_pred / max(num_instances, 1)
        overcapacity_rate = total_violations / max(num_instances, 1)

        return avg_pred, avg_gt, gap, avg_routes_pred, overcapacity_rate

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        config: RunConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        best_checkpoint_path: str | Path,
        last_checkpoint_path: str | Path | None = None,
    ) -> FitResult:
        if self.optimizer is None:
            raise ValueError("optimizer is required for fit()")

        training: TrainingConfig = config.training
        inference: InferenceConfig = config.inference
        best_checkpoint_path = Path(best_checkpoint_path)
        epochs = training.epochs
        best_gap = float("inf")
        history: list[EpochRecord] = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = self.train_one_epoch(train_loader, epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                lr = float(self.lr_scheduler.get_last_lr()[0])
            else:
                lr = float(self.optimizer.param_groups[0]["lr"])

            wandb.log(
                {
                    "train/loss_epoch": loss,
                    "train/lr": lr,
                    "epoch": epoch,
                }
            )

            run_validation = (
                epoch == 1 or epoch % training.eval_every == 0 or epoch == epochs
            )

            if run_validation:
                (
                    pred_len,
                    gt_len,
                    gap,
                    avg_routes,
                    overcapacity_rate,
                ) = self.validate(
                    val_loader,
                    num_inference_steps=inference.inference_steps,
                    schedule_type=inference.schedule,
                    max_instances=inference.eval_subset,
                )
                elapsed = time.time() - t0

                saved_best = gap < best_gap
                if saved_best:
                    best_gap = gap
                    self.save_checkpoint(
                        best_checkpoint_path,
                        epoch=epoch,
                        best_gap=best_gap,
                        config=config,
                        scheduler=self.lr_scheduler,
                    )

                wandb.log(
                    {
                        "val/pred_tour_length": pred_len,
                        "val/gt_tour_length": gt_len,
                        "val/gap_pct": gap,
                        "val/best_gap_pct": best_gap,
                        "val/num_routes_pred": avg_routes,
                        "val/overcapacity_rate": overcapacity_rate,
                        "epoch": epoch,
                    }
                )
                history.append(
                    EpochRecord(
                        epoch=epoch,
                        loss=loss,
                        lr=lr,
                        time_s=elapsed,
                        pred_length=pred_len,
                        gt_length=gt_len,
                        gap=gap,
                        num_routes_pred=avg_routes,
                        overcapacity_rate=overcapacity_rate,
                        saved_best=saved_best,
                    )
                )

                tag = "  [saved best]" if saved_best else ""
                logger.info(
                    f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Tour: {pred_len:.3f} "
                    f"(GT: {gt_len:.3f}, Gap: {gap:.2f}%) | "
                    f"Routes: {avg_routes:.1f} | Overcap: {overcapacity_rate:.2%} | "
                    f"{elapsed:.1f}s{tag}"
                )
            else:
                elapsed = time.time() - t0
                history.append(
                    EpochRecord(
                        epoch=epoch,
                        loss=loss,
                        lr=lr,
                        time_s=elapsed,
                    )
                )
                logger.info(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | {elapsed:.1f}s")

        if last_checkpoint_path is not None:
            self.save_checkpoint(
                last_checkpoint_path,
                epoch=epochs,
                best_gap=best_gap,
                config=config,
                scheduler=self.lr_scheduler,
            )

        return FitResult(best_gap=best_gap, history=history)

    # -------------------------------------------------------- checkpointing
    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        best_gap: float,
        config: RunConfig,
        scheduler: LRScheduler | None = None,
    ) -> None:
        path = Path(path)
        config_payload = config.model_dump(exclude_none=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "scheduler_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "epoch": epoch,
                "best_gap": best_gap,
                "config": config_payload,
            },
            path,
        )

from difusco.types.config import InferenceConfig, TrainingConfig
import logging
import time
from pathlib import Path

import torch
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from difusco.decoding import compute_tour_length, greedy_decode_tsp, two_opt
from difusco.models.model import DifuscoTSP
from difusco.types import EpochRecord, FitResult, RunConfig
from utils import select_device

logger = logging.getLogger(__name__)


class Trainer:
    """Training and validation loops for DifuscoTSP."""

    def __init__(
        self,
        model: DifuscoTSP,
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
                    },
                    step=global_step,
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        num_inference_steps: int = 10,
        schedule_type: str = "cosine",
        use_2opt: bool = True,
        max_instances: int = -1,
    ) -> tuple[float, float, float]:
        """
        Run inference on a loader and report tour-length optimality gap.

        Args:
            max_instances: cap instances evaluated (-1 = all).
                           200–500 is enough for convergence monitoring during training.
        """
        self.model.eval()
        total_pred_length = 0.0
        total_gt_length = 0.0
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
            node_feat, edge_index, edge_dist, edge_label = batch
            node_feat = node_feat.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_dist = edge_dist.to(self.device)
            edge_label = edge_label.to(self.device)

            heatmap = self.model.generate(
                device=self.device,
                node_feat=node_feat,
                edge_index=edge_index,
                edge_dist=edge_dist,
                num_inference_steps=num_inference_steps,
                schedule_type=schedule_type,
            )

            tour = greedy_decode_tsp(heatmap, edge_index, node_feat)

            if use_2opt:
                tour = two_opt(tour, node_feat, max_iterations=100)

            pred_length = compute_tour_length(tour, node_feat)

            gt_tour_edges = edge_label.nonzero(as_tuple=True)[0]
            gt_length = edge_dist[gt_tour_edges].sum().item() / 2.0

            total_pred_length += pred_length
            total_gt_length += gt_length
            num_instances += 1

            avg_gap = (
                (total_pred_length / num_instances - total_gt_length / num_instances)
                / (total_gt_length / num_instances)
                * 100
            )
            pbar.set_postfix(gap=f"{avg_gap:.2f}%")

            if max_instances > 0 and num_instances >= max_instances:
                break

        avg_pred = total_pred_length / max(num_instances, 1)
        avg_gt = total_gt_length / max(num_instances, 1)
        gap = (avg_pred - avg_gt) / avg_gt * 100

        return avg_pred, avg_gt, gap

    def fit(
        self,
        config: RunConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        best_checkpoint_path: str | Path,
        last_checkpoint_path: str | Path | None = None,
    ) -> FitResult:
        """
        Full training run driven by ``config``: train each epoch, validate
        periodically, save the best checkpoint, optionally save a final checkpoint.
        """
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

            if self.lr_scheduler is not None:
                lr = float(self.lr_scheduler.get_last_lr()[0])
            else:
                lr = float(self.optimizer.param_groups[0]["lr"])

            wandb.log(
                {
                    "train/loss_epoch": loss,
                    "train/lr": lr,
                    "epoch": epoch,
                },
                step=epoch,
            )

            run_validation = (
                epoch == 1 or epoch % training.eval_every == 0 or epoch == epochs
            )

            if run_validation:
                pred_len, gt_len, gap = self.validate(
                    val_loader,
                    num_inference_steps=inference.inference_steps,
                    schedule_type=inference.schedule,
                    use_2opt=inference.use_2opt,
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
                        "epoch": epoch,
                    },
                    step=epoch,
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
                        saved_best=saved_best,
                    )
                )

                if saved_best:
                    logger.info(
                        "  Epoch %3d | Loss: %.4f | Tour: %.3f (GT: %.3f, Gap: %.2f%%) | %.1fs  [saved best]",
                        epoch,
                        loss,
                        pred_len,
                        gt_len,
                        gap,
                        elapsed,
                    )
                else:
                    logger.info(
                        "  Epoch %3d | Loss: %.4f | Tour: %.3f (GT: %.3f, Gap: %.2f%%) | %.1fs",
                        epoch,
                        loss,
                        pred_len,
                        gt_len,
                        gap,
                        elapsed,
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
                logger.info(
                    "  Epoch %3d | Loss: %.4f | %.1fs",
                    epoch,
                    loss,
                    elapsed,
                )

        if last_checkpoint_path is not None:
            self.save_checkpoint(
                last_checkpoint_path,
                epoch=epochs,
                best_gap=best_gap,
                config=config,
                scheduler=self.lr_scheduler,
            )

        return FitResult(best_gap=best_gap, history=history)

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

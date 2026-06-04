"""
Training and validation loops for DifuscoMDVRP.

Training: masked cross-entropy on customer-depot edges (binary).
Validation: assignment accuracy + capacity-violation rate.

Routing-cost evaluation (decompose into K CVRPs, solve each, compare to GT
total cost) is intentionally NOT part of the training loop because it
requires running PyVRP per instance. See run_eval.py for that.
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
from difusco.mdvrp.decoding import (
    assignment_accuracy,
    greedy_decode_mdvrp_assignment,
)
from difusco.mdvrp.models.model import DifuscoMDVRP
from difusco.mdvrp.types import EpochRecord, FitResult, RunConfig
from difusco.mdvrp.types.config import InferenceConfig, TrainingConfig
from utils import select_device

logger = logging.getLogger(__name__)


class Trainer:
    """Training and validation loops for DifuscoMDVRP."""

    def __init__(
        self,
        model: DifuscoMDVRP,
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
    ) -> tuple[float, float, float]:
        """
        Validate by decoding per-instance assignment from the heatmap.

        Returns:
            (avg_assignment_accuracy, avg_overcapacity_rate, avg_loss_proxy)
        """
        self.model.eval()
        total_acc = 0.0
        total_overcap = 0.0
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
            node_feat, edge_index, edge_dist, _edge_label, edge_mask, meta_list = batch
            node_feat = node_feat.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_dist = edge_dist.to(self.device)
            edge_mask = edge_mask.to(self.device)

            heatmap = self.model.generate(
                device=self.device,
                node_feat=node_feat,
                edge_index=edge_index,
                edge_dist=edge_dist,
                num_inference_steps=num_inference_steps,
                schedule_type=schedule_type,
            )

            # Per-instance decoding (validation uses batch_size=1).
            if len(meta_list) != 1:
                raise ValueError(
                    f"validate() expects batch_size=1, got {len(meta_list)} graphs"
                )
            meta = meta_list[0]
            assignment, overcap = greedy_decode_mdvrp_assignment(
                heatmap=heatmap,
                edge_index=edge_index,
                n_customers=meta["n_customers"],
                n_depots=meta["n_depots"],
                demands=meta["demands"],
                capacity=meta["capacity"],
                num_vehicles_per_depot=meta["num_vehicles_per_depot"],
                node_offset=meta["node_offset"],
                edge_mask=edge_mask,
            )

            acc = assignment_accuracy(assignment, meta["gt_assignment"])
            overcap_rate = overcap / max(meta["n_customers"], 1)

            total_acc += acc
            total_overcap += overcap_rate
            num_instances += 1

            pbar.set_postfix(
                acc=f"{total_acc / num_instances:.3f}",
                overcap=f"{total_overcap / num_instances:.3f}",
            )

            if max_instances > 0 and num_instances >= max_instances:
                break

        avg_acc = total_acc / max(num_instances, 1)
        avg_overcap = total_overcap / max(num_instances, 1)
        return avg_acc, avg_overcap, 0.0

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
        best_acc = -1.0
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
                acc, overcap, _ = self.validate(
                    val_loader,
                    num_inference_steps=inference.inference_steps,
                    schedule_type=inference.schedule,
                    max_instances=inference.eval_subset,
                )
                elapsed = time.time() - t0

                saved_best = acc > best_acc
                if saved_best:
                    best_acc = acc
                    self.save_checkpoint(
                        best_checkpoint_path,
                        epoch=epoch,
                        best_accuracy=best_acc,
                        config=config,
                        scheduler=self.lr_scheduler,
                    )

                wandb.log(
                    {
                        "val/assignment_accuracy": acc,
                        "val/capacity_violation_rate": overcap,
                        "val/best_accuracy": best_acc,
                        "epoch": epoch,
                    }
                )
                history.append(
                    EpochRecord(
                        epoch=epoch,
                        loss=loss,
                        lr=lr,
                        time_s=elapsed,
                        assignment_accuracy=acc,
                        capacity_violation_rate=overcap,
                        saved_best=saved_best,
                    )
                )

                tag = "  [saved best]" if saved_best else ""
                logger.info(
                    f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.3f} | "
                    f"Overcap: {overcap:.3f} | {elapsed:.1f}s{tag}"
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
                best_accuracy=best_acc,
                config=config,
                scheduler=self.lr_scheduler,
            )

        return FitResult(best_accuracy=best_acc, history=history)

    # -------------------------------------------------------- checkpointing
    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        best_accuracy: float,
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
                "best_accuracy": best_accuracy,
                "config": config_payload,
            },
            path,
        )

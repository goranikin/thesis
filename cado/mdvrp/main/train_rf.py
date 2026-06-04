"""
Vanilla REINFORCE fine-tuning for CADO-MDVRP.

Reward: assignment quality (negative miss rate against ground-truth depot
labels). See ``CADOMDVRP.compute_reward``.

Best-checkpoint metric: highest assignment accuracy on the val subset.
"""

from pathlib import Path

import torch
from tqdm import tqdm

import wandb
from cado.mdvrp.evaluate import evaluate
from cado.mdvrp.models.model import CADOMDVRP

# 6-tuple from MDVRPDataset.collate_mdvrp:
#   (node_feat, edge_index, edge_dist, edge_label, edge_mask, meta_list)
MDVRPInstance = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict],
]


def _move_meta(meta: dict, device: torch.device) -> dict:
    """Move tensor fields of a per-instance meta dict to device."""
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in meta.items()
    }


def reinforce_step(
    model: CADOMDVRP,
    batch: list[MDVRPInstance],
    optimizer: torch.optim.Optimizer,
    reward_mode: str = "LCR",
    M_train: int = 10,
    schedule_type: str = "cosine",
    grad_clip: float = 1.0,
) -> dict:
    device = next(model.parameters()).device

    sum_log_probs_per_inst = []
    rewards_per_inst = []

    for node_feat, edge_index, edge_dist, _, edge_mask, meta_list in batch:
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_mask = edge_mask.to(device)
        assert len(meta_list) == 1, "Use batch_size=1 in the DataLoader."
        meta = _move_meta(meta_list[0], device)

        x_0, log_probs = model.rollout(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            edge_mask=edge_mask,
            num_inference_steps=M_train,
            schedule_type=schedule_type,
        )
        sum_log_probs_per_inst.append(log_probs.sum())

        with torch.no_grad():
            reward = model.compute_reward(
                x_0=x_0,
                edge_index=edge_index,
                edge_mask=edge_mask,
                n_customers=meta["n_customers"],
                n_depots=meta["n_depots"],
                demands=meta["demands"],
                capacity=meta["capacity"],
                num_vehicles_per_depot=meta["num_vehicles_per_depot"],
                gt_assignment=meta["gt_assignment"],
                node_offset=meta["node_offset"],
                mode=reward_mode,
            )
        rewards_per_inst.append(reward)

    log_probs_batch = torch.stack(sum_log_probs_per_inst)
    rewards_batch = torch.tensor(
        rewards_per_inst, device=device, dtype=log_probs_batch.dtype
    )

    if reward_mode == "SR":
        advantage = (rewards_batch - rewards_batch.mean()) / (
            rewards_batch.std() + 1e-6
        )
    else:
        advantage = rewards_batch

    entropy = -log_probs_batch.mean()
    loss = -(advantage.detach() * log_probs_batch).mean() - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
    )
    optimizer.step()

    return {
        "loss": loss.item(),
        "mean_reward": rewards_batch.mean().item(),
        "mean_log_prob": log_probs_batch.mean().item(),
        "grad_norm": grad_norm.item(),
    }


def train_reinforce(
    model: CADOMDVRP,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device,
    cfg,
    *,
    ckpt_path: Path | None = None,
) -> float:
    """
    Returns the best (highest) assignment accuracy seen on the val subset.
    """
    model.train()
    accum_batch = cfg.cado.batch_size
    samples_per_epoch = cfg.cado.samples_per_epoch

    train_iter = iter(train_loader)
    global_step = 0
    best_accuracy = -1.0

    for epoch in range(1, cfg.cado.epochs + 1):
        n_updates = samples_per_epoch // accum_batch
        pbar = tqdm(range(n_updates), desc=f"Epoch {epoch}", dynamic_ncols=True)

        for _ in pbar:
            batch: list[MDVRPInstance] = []
            for _ in range(accum_batch):
                try:
                    inst = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inst = next(train_iter)
                batch.append(inst)

            metrics = reinforce_step(
                model,
                batch,
                optimizer,
                reward_mode=cfg.cado.reward_mode,
                M_train=cfg.cado.M_train,
                schedule_type=cfg.cado.schedule_type,
                grad_clip=cfg.cado.grad_clip,
            )

            if global_step % cfg.cado.log_interval == 0:
                wandb.log(
                    {
                        **{f"train/{k}": v for k, v in metrics.items()},
                        "train/global_step": global_step,
                    }
                )
            global_step += 1
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                reward=f"{metrics['mean_reward']:.4f}",
            )

        if epoch % cfg.cado.eval_every == 0 or epoch == cfg.cado.epochs:
            model.eval()
            acc, overcap = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                num_inference_steps=cfg.cado.M_eval,
                schedule_type=cfg.cado.schedule_type,
                max_instances=cfg.cado.eval_subset,
            )
            wandb.log(
                {
                    "val/assignment_accuracy": acc,
                    "val/capacity_violation_rate": overcap,
                    "epoch": epoch,
                }
            )

            if acc > best_accuracy:
                best_accuracy = acc
                if ckpt_path is not None:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_accuracy": best_accuracy,
                            "global_step": global_step,
                        },
                        ckpt_path,
                    )
                    print(
                        f"  Epoch {epoch}: acc = {acc:.3f}  "
                        f"overcap = {overcap:.3f}  [saved to {ckpt_path}]"
                    )
                else:
                    print(f"  Epoch {epoch}: acc = {acc:.3f}  [new best]")
            else:
                print(f"  Epoch {epoch}: acc = {acc:.3f}")
            model.train()

    return best_accuracy

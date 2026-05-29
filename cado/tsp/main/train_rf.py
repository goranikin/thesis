"""
Vanilla REINFORCE fine-tuning for CADO-TSP (paper Eq. 9).
"""

from pathlib import Path

import torch
from tqdm import tqdm

import wandb
from cado.tsp.evaluate import evaluate
from cado.tsp.models.model import CADOTSP


def _ground_truth_length(edge_dist: torch.Tensor, edge_label: torch.Tensor) -> float:
    return edge_dist[edge_label.nonzero(as_tuple=True)[0]].sum().item() / 2.0


def reinforce_step(
    model: CADOTSP,
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    reward_mode: str = "LCR",
    use_2opt_in_reward: bool = False,
    M_train: int = 10,
    schedule_type: str = "cosine",
    grad_clip: float = 1.0,
) -> dict:
    device = next(model.parameters()).device

    sum_log_probs_per_inst = []
    rewards_per_inst = []

    for node_feat, edge_index, edge_dist, edge_label in batch:
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

        x_0, log_probs = model.rollout(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            num_inference_steps=M_train,
            schedule_type=schedule_type,
        )
        sum_log_probs_per_inst.append(log_probs.sum())

        gt_length = _ground_truth_length(edge_dist, edge_label)
        with torch.no_grad():
            reward = model.compute_reward(
                x_0=x_0,
                edge_index=edge_index,
                node_feat=node_feat,
                gt_length=gt_length,
                mode=reward_mode,
                use_2opt=use_2opt_in_reward,
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
    model: CADOTSP,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device,
    cfg,
    *,
    ckpt_path: Path | None = None,
) -> float:
    model.train()
    accum_batch = cfg.cado.batch_size
    samples_per_epoch = cfg.cado.samples_per_epoch

    train_iter = iter(train_loader)
    global_step = 0
    best_gap = float("inf")

    for epoch in range(1, cfg.cado.epochs + 1):
        epoch_metrics = {"loss": 0.0, "mean_reward": 0.0}
        n_updates = samples_per_epoch // accum_batch

        pbar = tqdm(range(n_updates), desc=f"Epoch {epoch}", dynamic_ncols=True)
        for _ in pbar:
            batch = []
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
                use_2opt_in_reward=cfg.cado.use_2opt_in_reward,
                M_train=cfg.cado.M_train,
                schedule_type=cfg.cado.schedule_type,
                grad_clip=cfg.cado.grad_clip,
            )
            epoch_metrics["loss"] += metrics["loss"]
            epoch_metrics["mean_reward"] += metrics["mean_reward"]

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
            pred_len, gt_len, gap = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                num_nodes=cfg.data.num_nodes,
                num_inference_steps=cfg.cado.M_eval,
                schedule_type=cfg.cado.schedule_type,
                use_2opt=True,
                max_instances=cfg.cado.eval_subset,
            )
            wandb.log(
                {
                    "val/pred_len": pred_len,
                    "val/gt_len": gt_len,
                    "val/gap_pct": gap,
                    "epoch": epoch,
                }
            )

            if gap < best_gap:
                best_gap = gap
                if ckpt_path is not None:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_gap": best_gap,
                            "global_step": global_step,
                        },
                        ckpt_path,
                    )
                    print(
                        f"  Epoch {epoch}: gap = {gap:.2f}%  [saved best to {ckpt_path}]"
                    )
                else:
                    print(f"  Epoch {epoch}: gap = {gap:.2f}%  [new best]")
            else:
                print(f"  Epoch {epoch}: gap = {gap:.2f}%")
            model.train()

    return best_gap

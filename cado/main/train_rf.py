"""
Vanilla REINFORCE fine-tuning for CADO (paper Eq. 9).

The objective:
    ∇_θ J(θ) = E_τ [ R(τ) * Σ_t ∇_θ log π_θ(x_{t-1} | x_t, g) ]

Implementation outline per epoch:
    1. For each instance in the batch:
        - Rollout an M-step trajectory, collecting sum_log_prob
        - Decode x_0 → tour → reward
    2. Convert per-instance rewards to advantages:
        - SR: batch-normalize (R - mean) / std
        - LCR: use as-is (gt_length already provides the baseline)
    3. Loss = -mean(advantage * sum_log_prob)
    4. Backprop + clip + step.

Why we iterate over instances inside the batch instead of batching them as a
super-graph:
    - The decoder (greedy_decode_tsp + 2-opt) is per-instance Python code
      that runs on CPU; it cannot trivially digest a batched super-graph.
    - Iterating gives us per-instance rewards needed for advantage computation.
    - We accumulate gradients via autograd; the final loss is the average over
      the batch, so it is equivalent to a batched update in expectation.
"""

from pathlib import Path

import torch
from tqdm import tqdm

import wandb
from cado.evaluate import evaluate
from cado.models.model import CADOTSP


def _ground_truth_length(edge_dist: torch.Tensor, edge_label: torch.Tensor) -> float:
    """
    Recover the GT tour length from the dataset's edge labels.

    Each undirected tour edge appears twice in edge_label (once as (u,v) and
    once as (v,u)), so we divide by 2.
    """
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
    """
    One REINFORCE update over one batch of TSP instances.

    Args:
        model: a DifuscoTSP with apply_hybrid_ft applied. Must expose:
               - model.rollout(node_feat, edge_index, edge_dist, num_inference_steps)
                 returning (x_0, log_probs_per_step)
               - model.compute_reward(x_0, edge_index, node_feat, gt_length, mode, use_2opt)
        batch: a list of single-instance tuples (node_feat, edge_index, edge_dist, edge_label).
               IMPORTANT: pass a list of per-instance tuples here, NOT the
               output of collate_tsp(). See the trainer below for how this is
               constructed.
        optimizer: optimizer over only the trainable parameters.
        reward_mode: "SR" (batch-normalized) or "LCR" (gt_length baseline).
        use_2opt_in_reward: whether to run 2-opt inside the reward. Paper's
                            CADO-L variant trains WITHOUT 2-opt for speed
                            (Table 13). We default to False for the same reason.
        M_train: number of denoising steps during the RL rollout. Paper uses 10.
        schedule_type: "linear" or "cosine".
        grad_clip: gradient norm clip. Paper uses 1.0.

    Returns:
        Dict of scalar metrics for logging.
    """
    device = next(model.parameters()).device

    sum_log_probs_per_inst = []  # one tensor (scalar with grad) per instance
    rewards_per_inst = []  # one Python float per instance

    for node_feat, edge_index, edge_dist, edge_label in batch:
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

        # Rollout — gradients flow through log_probs but NOT through x_t.
        x_0, log_probs = model.rollout(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            num_inference_steps=M_train,
            schedule_type=schedule_type,
        )
        # log_probs: shape (M-1,) — one scalar log-prob per stochastic step.
        sum_log_prob = log_probs.sum()
        sum_log_probs_per_inst.append(sum_log_prob)

        # Reward — purely a Python float, no gradient.
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

    # Stack into batch tensors. log_probs_batch is differentiable;
    # rewards_batch is not.
    log_probs_batch = torch.stack(sum_log_probs_per_inst)  # (B,)
    rewards_batch = torch.tensor(
        rewards_per_inst, device=device, dtype=log_probs_batch.dtype
    )  # (B,)

    # Compute the advantage.
    #   For SR, normalize across the batch — this is the standard REINFORCE
    #   baseline trick that reduces variance.
    #   For LCR, the reward is already an unbiased baseline so we use as-is.
    if reward_mode == "SR":
        advantage = (rewards_batch - rewards_batch.mean()) / (
            rewards_batch.std() + 1e-6
        )
    else:  # LCR
        advantage = rewards_batch

    # REINFORCE loss. The negative sign is because we MAXIMIZE reward.
    # `advantage` is detached (it was built from Python floats with no grad).
    loss = -(advantage.detach() * log_probs_batch).mean()

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
    """
    Full REINFORCE training loop.

    The data loader is expected to use batch_size=1 with the existing
    collate_tsp; we re-collect per-instance tuples here. If you prefer
    batch_size > 1 from the loader, you can instead bypass collate_tsp by
    setting collate_fn to `lambda x: x`.

    Returns:
        best_gap: the lowest validation gap observed during training.
    """

    model.train()
    accum_batch = cfg.cado.batch_size  # paper: 64
    samples_per_epoch = cfg.cado.samples_per_epoch  # paper: 512

    train_iter = iter(train_loader)
    global_step = 0
    best_gap = float("inf")

    for epoch in range(1, cfg.cado.epochs + 1):
        epoch_metrics = {"loss": 0.0, "mean_reward": 0.0}
        n_updates = samples_per_epoch // accum_batch

        pbar = tqdm(range(n_updates), desc=f"Epoch {epoch}", dynamic_ncols=True)
        for _ in pbar:
            # Collect `accum_batch` single-instance tuples.
            batch = []
            for _ in range(accum_batch):
                try:
                    inst = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inst = next(train_iter)
                # If loader uses batch_size=1 + collate_tsp, `inst` is already a
                # 4-tuple of per-instance tensors. Unwrap the batch dim if needed.
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
                # No explicit step=; the wandb x-axis is bound to
                # train/global_step via define_metric in run_train.py.
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

        # Periodic validation
        if epoch % cfg.cado.eval_every == 0 or epoch == cfg.cado.epochs:
            model.eval()
            pred_len, gt_len, gap = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
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

            # Track + save best checkpoint.
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

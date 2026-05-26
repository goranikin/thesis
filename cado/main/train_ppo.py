"""
PPO fine-tuning for CADO (the DDPO-style implementation that the paper's
supplementary code actually uses).

The paper itself writes the REINFORCE objective (Eq. 9), but the supplementary
code's `train_co.py` (lines 1125-1145) implements PPO with importance-ratio
clipping. The two key ideas:

  (a) Reuse each collected trajectory for K inner-epoch updates. This is much
      more sample-efficient than vanilla REINFORCE.
  (b) Clip the importance ratio   ρ = exp(log π_new - log π_old)   to stay near 1.
      Standard PPO trick; bounds the per-step update magnitude.

Per-step view:
    For every transition (x_t, x_{t-1}, t) collected during rollout:
        log_prob_old = mean Bernoulli log-prob under the rollout-time policy
        advantage    = R(τ) - baseline               (same for all t in a trajectory)

        # During the inner loop, run the backbone again on the SAME (x_t, t)
        # but compute log_prob of the SAME x_{t-1}:
        log_prob_new = q_posterior_with_logprob(x_t, x_0_pred_new, t, action=x_{t-1})

        ratio        = exp(log_prob_new - log_prob_old)
        surr1        = advantage * ratio
        surr2        = advantage * clamp(ratio, 1 - ε, 1 + ε)
        policy_loss  = -mean(min(surr1, surr2))
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

import wandb
from cado.evaluate import evaluate
from cado.models.model import CADOTSP
from difusco.tsp.models.diffusion import InferenceSchedule

# --------------------------------------------------------------------------- #
# Rollout buffer
# --------------------------------------------------------------------------- #


class Trajectory(BaseModel):
    """
    Stores all per-step quantities needed to recompute log_prob_new.

    Each list has length M-1 (the number of stochastic transitions; the final
    argmax step contributes no log-prob).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Inputs that are constant across all steps of this trajectory
    node_feat: torch.Tensor  # (N, 2)
    edge_index: torch.Tensor  # (2, E)
    edge_dist: torch.Tensor  # (E,)
    # Per-step
    x_t_list: list[torch.Tensor]  # state BEFORE each transition
    x_tm1_list: list[torch.Tensor]  # state AFTER each transition (the "action")
    t_list: list[int]  # timestep index
    log_prob_old: list[torch.Tensor]  # detached scalar per step
    # Trajectory-level
    advantage: float = 0.0  # set after all trajectories are collected


def collect_trajectory(
    model: CADOTSP,
    instance: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    M_train: int,
    schedule_type: str,
) -> tuple[Trajectory, torch.Tensor]:
    """
    Roll out one trajectory with no gradients, saving every per-step quantity
    needed for PPO's recomputation step.

    Returns: (Trajectory, x_0)
    """

    node_feat, edge_index, edge_dist, _ = instance
    device = next(model.parameters()).device
    node_feat = node_feat.to(device)
    edge_index = edge_index.to(device)
    edge_dist = edge_dist.to(device)
    E = edge_index.shape[1]

    timesteps = InferenceSchedule.get_schedule(schedule_type, M_train, model.T)

    x_t = torch.bernoulli(0.5 * torch.ones(E, device=device))

    traj = Trajectory(
        node_feat=node_feat,
        edge_index=edge_index,
        edge_dist=edge_dist,
        x_t_list=[],
        x_tm1_list=[],
        t_list=[],
        log_prob_old=[],
    )

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
            logits = model.backbone(node_feat, edge_index, edge_dist, x_t, t_tensor)
            x_0_prob = F.softmax(logits, dim=-1)[:, 1]

            if i == len(timesteps) - 1:
                # Final deterministic argmax — no transition to log.
                x_0 = (x_0_prob > 0.5).float()
                break

            # Save the pre-transition state.
            traj.x_t_list.append(x_t.clone())
            traj.t_list.append(t)

            # Sample x_{t-1}; we want the SAMPLE and the log-prob of it.
            x_tm1, logp = model.diffusion.q_posterior_with_logprob(x_t, x_0_prob, t)
            traj.x_tm1_list.append(x_tm1.clone())
            traj.log_prob_old.append(logp.detach())

            x_t = x_tm1  # transition

    return traj, x_0


# --------------------------------------------------------------------------- #
# PPO update
# --------------------------------------------------------------------------- #


def recompute_log_prob(model, traj: Trajectory, step_idx: int) -> torch.Tensor:
    """
    Recompute log π_new(x_{t-1} | x_t, g) for one specific step of a trajectory.

    This is the crucial PPO operation: we re-run the backbone with gradients
    enabled, then evaluate the Bernoulli log-prob of the PREVIOUSLY-SAMPLED
    action x_{t-1} under the new policy.
    """
    x_t = traj.x_t_list[step_idx]
    x_tm1 = traj.x_tm1_list[step_idx]
    t = traj.t_list[step_idx]
    device = x_t.device

    t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
    logits = model.backbone(
        traj.node_feat, traj.edge_index, traj.edge_dist, x_t, t_tensor
    )
    x_0_prob = F.softmax(logits, dim=-1)[:, 1]

    # Pass `action=x_tm1` so we don't draw a new sample; we just want the
    # log-prob of the OLD sample under the NEW policy.
    _, log_prob_new = model.diffusion.q_posterior_with_logprob(
        x_t, x_0_prob, t, action=x_tm1
    )
    return log_prob_new


def ppo_update(
    model: CADOTSP,
    trajectories: list[Trajectory],
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float = 1e-4,
    grad_clip: float = 1.0,
) -> dict:
    """
    One PPO inner-epoch pass over all collected trajectories.

    We loop over (trajectory, step) pairs. For each step we recompute
    log_prob_new, form the clipped surrogate, and accumulate gradients.

    Note on clip_epsilon: the CADO supplementary code uses a VERY small clip
    (1e-4), reflecting that the policy changes only slightly per inner epoch.
    Standard PPO for control uses 0.1–0.2. Start with 1e-4 to match the paper.
    """
    losses = []
    ratios_log = []

    for traj in trajectories:
        advantage = torch.tensor(traj.advantage, device=traj.node_feat.device)
        # Loop over the M-1 transitions in this trajectory.
        for step_idx in range(len(traj.x_t_list)):
            log_prob_new = recompute_log_prob(model, traj, step_idx)
            log_prob_old = traj.log_prob_old[step_idx]

            ratio = torch.exp(log_prob_new - log_prob_old)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(
                ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
            )
            # Note: PPO MAXIMIZES min(surr1, surr2). With advantage > 0, this
            # caps the upside; with advantage < 0, it caps the downside.
            step_loss = -torch.min(surr1, surr2)
            losses.append(step_loss)
            ratios_log.append(ratio.detach())

    loss = torch.stack(losses).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
    )
    optimizer.step()

    return {
        "loss": loss.item(),
        "mean_ratio": torch.stack(ratios_log).mean().item(),
        "grad_norm": grad_norm.item(),
    }


# --------------------------------------------------------------------------- #
# Full training loop
# --------------------------------------------------------------------------- #


def _ground_truth_length(edge_dist, edge_label):
    return edge_dist[edge_label.nonzero(as_tuple=True)[0]].sum().item() / 2.0


def ppo_outer_step(
    model: CADOTSP,
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    reward_mode: str = "LCR",
    use_2opt_in_reward: bool = False,
    M_train: int = 10,
    schedule_type: str = "cosine",
    inner_epochs: int = 4,
    clip_epsilon: float = 1e-4,
    grad_clip: float = 1.0,
) -> dict:
    """
    One outer step:
      1. COLLECT  trajectories with no_grad.
      2. COMPUTE  advantages.
      3. UPDATE   model for `inner_epochs` inner passes (PPO core).
    """
    device = next(model.parameters()).device

    # 1. COLLECT
    trajectories = []
    rewards = []
    for instance in batch:
        node_feat, edge_index, edge_dist, edge_label = [t.to(device) for t in instance]
        traj, x_0 = collect_trajectory(
            model,
            (node_feat, edge_index, edge_dist, edge_label),
            M_train=M_train,
            schedule_type=schedule_type,
        )
        gt_length = _ground_truth_length(edge_dist, edge_label)
        reward = model.compute_reward(
            x_0,
            edge_index,
            node_feat,
            gt_length=gt_length,
            mode=reward_mode,
            use_2opt=use_2opt_in_reward,
        )
        rewards.append(reward)
        trajectories.append(traj)

    rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

    # 2. COMPUTE ADVANTAGES (same rule as REINFORCE)
    if reward_mode == "SR":
        advantages = (rewards_tensor - rewards_tensor.mean()) / (
            rewards_tensor.std() + 1e-6
        )
    else:
        advantages = rewards_tensor
    for traj, adv in zip(trajectories, advantages.tolist()):
        traj.advantage = adv

    # 3. INNER LOOP
    metrics = {"loss": 0.0, "mean_ratio": 0.0, "grad_norm": 0.0}
    for k in range(inner_epochs):
        m = ppo_update(
            model,
            trajectories,
            optimizer,
            clip_epsilon=clip_epsilon,
            grad_clip=grad_clip,
        )
        for key in metrics:
            metrics[key] += m[key] / inner_epochs

    metrics["mean_reward"] = rewards_tensor.mean().item()
    return metrics


def train_ppo(
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
    Outer training loop. Mirrors train_reinforce but calls ppo_outer_step.

    Returns:
        best_gap: the lowest validation gap observed during training.
    """

    model.train()
    accum_batch = cfg.cado.batch_size
    samples_per_epoch = cfg.cado.samples_per_epoch
    global_step = 0
    best_gap = float("inf")
    train_iter = iter(train_loader)

    for epoch in range(1, cfg.cado.epochs + 1):
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

            metrics = ppo_outer_step(
                model,
                batch,
                optimizer,
                reward_mode=cfg.cado.reward_mode,
                use_2opt_in_reward=cfg.cado.use_2opt_in_reward,
                M_train=cfg.cado.M_train,
                schedule_type=cfg.cado.schedule_type,
                inner_epochs=cfg.cado.ppo_inner_epochs,
                clip_epsilon=cfg.cado.ppo_clip_epsilon,
                grad_clip=cfg.cado.grad_clip,
            )

            if global_step % cfg.cado.log_interval == 0:
                # No explicit step=; x-axis is bound to train/global_step
                # via define_metric in run_train.py.
                wandb.log(
                    {
                        **{f"train/{k}": v for k, v in metrics.items()},
                        "train/global_step": global_step,
                    }
                )
            global_step += 1
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                ratio=f"{metrics['mean_ratio']:.4f}",
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

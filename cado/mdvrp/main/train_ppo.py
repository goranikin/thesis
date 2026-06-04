"""
PPO fine-tuning for CADO-MDVRP. Mirrors cado/cvrp/main/train_ppo.py with
masked log-probs and the assignment-quality reward.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

import wandb
from cado.mdvrp.evaluate import evaluate
from cado.mdvrp.main.train_rf import MDVRPInstance, _move_meta
from cado.mdvrp.models.model import CADOMDVRP
from difusco.mdvrp.models.diffusion import InferenceSchedule


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_feat: torch.Tensor
    edge_index: torch.Tensor
    edge_dist: torch.Tensor
    edge_mask: torch.Tensor
    x_t_list: list[torch.Tensor]
    x_tm1_list: list[torch.Tensor]
    t_list: list[int]
    log_prob_old: list[torch.Tensor]
    advantage: float = 0.0


def collect_trajectory(
    model: CADOMDVRP,
    instance: MDVRPInstance,
    M_train: int,
    schedule_type: str,
) -> tuple[Trajectory, torch.Tensor]:
    node_feat, edge_index, edge_dist, _, edge_mask, _meta = instance
    device = next(model.parameters()).device
    node_feat = node_feat.to(device)
    edge_index = edge_index.to(device)
    edge_dist = edge_dist.to(device)
    edge_mask = edge_mask.to(device)
    E = edge_index.shape[1]

    timesteps = InferenceSchedule.get_schedule(schedule_type, M_train, model.T)
    x_t = torch.bernoulli(0.5 * torch.ones(E, device=device))

    traj = Trajectory(
        node_feat=node_feat,
        edge_index=edge_index,
        edge_dist=edge_dist,
        edge_mask=edge_mask,
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
                x_0 = (x_0_prob > 0.5).float()
                break

            traj.x_t_list.append(x_t.clone())
            traj.t_list.append(t)
            x_tm1, logp = model._masked_q_posterior_with_logprob(
                x_t, x_0_prob, t, edge_mask
            )
            traj.x_tm1_list.append(x_tm1.clone())
            traj.log_prob_old.append(logp.detach())
            x_t = x_tm1

    return traj, x_0


def recompute_log_prob(model: CADOMDVRP, traj: Trajectory, step_idx: int) -> torch.Tensor:
    x_t = traj.x_t_list[step_idx]
    x_tm1 = traj.x_tm1_list[step_idx]
    t = traj.t_list[step_idx]
    device = x_t.device

    t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
    logits = model.backbone(
        traj.node_feat, traj.edge_index, traj.edge_dist, x_t, t_tensor
    )
    x_0_prob = F.softmax(logits, dim=-1)[:, 1]

    # Use the same masked log-prob reduction as in rollout, conditioning on
    # the action that was actually taken (x_tm1).
    if t == 0:
        return x_t.new_zeros((1,))
    beta_t = model.diffusion.betas[t].float().to(x_t.device)
    alpha_bar_tm1 = model.diffusion.alphas_cumprod[t - 1].float().to(x_t.device)
    p_xt_g_1 = x_t * (1 - beta_t) + (1 - x_t) * beta_t
    p_xt_g_0 = x_t * beta_t + (1 - x_t) * (1 - beta_t)
    p_xtm1_1 = (
        x_0_prob * (1 + alpha_bar_tm1) / 2
        + (1 - x_0_prob) * (1 - alpha_bar_tm1) / 2
    )
    p_xtm1_0 = 1.0 - p_xtm1_1
    prob_1 = p_xt_g_1 * p_xtm1_1 / (p_xt_g_1 * p_xtm1_1 + p_xt_g_0 * p_xtm1_0 + 1e-8)
    prob_1 = prob_1.clamp(1e-6, 1 - 1e-6)

    per_edge_logp = x_tm1 * torch.log(prob_1) + (1.0 - x_tm1) * torch.log(1.0 - prob_1)
    masked_sum = (per_edge_logp * traj.edge_mask).sum()
    denom = traj.edge_mask.sum().clamp(min=1.0)
    return masked_sum / denom


def ppo_update(
    model: CADOMDVRP,
    trajectories: list[Trajectory],
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float = 0.2,
    grad_clip: float = 1.0,
) -> dict:
    losses = []
    ratios_log = []

    for traj in trajectories:
        advantage = torch.tensor(traj.advantage, device=traj.node_feat.device)
        for step_idx in range(len(traj.x_t_list)):
            log_prob_new = recompute_log_prob(model, traj, step_idx)
            log_prob_old = traj.log_prob_old[step_idx]

            ratio = torch.exp(log_prob_new - log_prob_old)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(
                ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
            )
            losses.append(-torch.min(surr1, surr2))
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


def ppo_outer_step(
    model: CADOMDVRP,
    batch: list[MDVRPInstance],
    optimizer: torch.optim.Optimizer,
    *,
    reward_mode: str = "LCR",
    M_train: int = 10,
    schedule_type: str = "cosine",
    inner_epochs: int = 4,
    clip_epsilon: float = 0.2,
    grad_clip: float = 1.0,
) -> dict:
    device = next(model.parameters()).device

    trajectories: list[Trajectory] = []
    rewards: list[float] = []
    for instance in batch:
        node_feat, edge_index, edge_dist, edge_label, edge_mask, meta_list = instance
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)
        edge_mask = edge_mask.to(device)
        meta = _move_meta(meta_list[0], device)

        moved_instance = (node_feat, edge_index, edge_dist, edge_label, edge_mask, [meta])
        traj, x_0 = collect_trajectory(
            model, moved_instance, M_train=M_train, schedule_type=schedule_type
        )
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
        rewards.append(reward)
        trajectories.append(traj)

    rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

    if reward_mode == "SR":
        advantages = (rewards_tensor - rewards_tensor.mean()) / (
            rewards_tensor.std() + 1e-6
        )
    else:
        advantages = rewards_tensor
    for traj, adv in zip(trajectories, advantages.tolist()):
        traj.advantage = adv

    metrics = {"loss": 0.0, "mean_ratio": 0.0, "grad_norm": 0.0}
    for _ in range(inner_epochs):
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
    model: CADOMDVRP,
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
    global_step = 0
    best_accuracy = -1.0
    train_iter = iter(train_loader)

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

            metrics = ppo_outer_step(
                model,
                batch,
                optimizer,
                reward_mode=cfg.cado.reward_mode,
                M_train=cfg.cado.M_train,
                schedule_type=cfg.cado.schedule_type,
                inner_epochs=cfg.cado.ppo_inner_epochs,
                clip_epsilon=cfg.cado.ppo_clip_epsilon,
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
                ratio=f"{metrics['mean_ratio']:.4f}",
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

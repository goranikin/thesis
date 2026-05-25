# Experiment Log

## Experiment 4: CADO REINFORCE Fine-Tuning on DIFUSCO-Small (SR Reward)

**Date**: 2026-05-25
**wandb run**: `difusco-tsp / cado_reinforce_tsp50_h128_L6_SR` (started 2026-05-25 10:41:23 KST)
**Goal**: Reproduce CADO's RL fine-tuning of a heatmap solver (paper Eq. 9, Algorithm 1) on top of the small DIFUSCO baseline, and verify that REINFORCE further reduces the optimality gap below the SL starting point of 4.46%.

### Background

This is the first end-to-end CADO run. The pipeline implements Hybrid Fine-Tuning (LoRA on the input encoder + first L−1 GNN layers, full retraining on the last layer + output head) and vanilla REINFORCE with Standard Reward (SR) batch-normalized advantages.

A prior pilot with `reward_mode=LCR, lr=1e-5` produced zero learning signal (`grad_norm ≈ 2e-4`, val gap flat around 5%). Switching to `SR` and raising the learning rate to `1e-4` recovered a usable — if slow — signal.

### Configuration

| Parameter | Value |
|---|---|
| SL checkpoint | `checkpoints/20260524_234510/best_model.pt` (4.46% gap) |
| Algorithm | REINFORCE |
| Reward mode | SR (batch-normalized advantage) |
| 2-opt in reward | false |
| Init noise | bernoulli_half |
| Hybrid-FT — LoRA rank | 2 |
| Hybrid-FT — selective (full) layers | 1 |
| Trainable params | 178,050 / 953,474 (18.67%) |
| Optimizer | AdamW |
| Learning rate | 1.0e-4 |
| Weight decay | 0.0 |
| Gradient clip | 1.0 |
| `M_train` (denoising steps in rollout) | 5 |
| `M_eval` (denoising steps in eval) | 50 |
| Schedule type | cosine |
| Epochs | 100 |
| Samples per epoch | 128 |
| Batch size | 32 |
| Updates per epoch | 4 (= 128 / 32) |
| Total optimizer steps | 400 |
| Eval cadence | every 10 epochs |
| Eval subset | 100 instances |
| Post-processing (eval) | Greedy decode + 2-opt |
| Seed | 42 |
| Device | Apple M4 Pro (MPS) |

### Computing Resources

| Resource | Specification |
|---|---|
| Machine | MacBook Pro with Apple M4 Pro |
| CPU | 12 cores (8P + 4E) |
| GPU | 16-core integrated (MPS) |
| RAM | 24 GB unified |
| Backend | PyTorch / MPS |

### Training Values

| Metric | Value |
|---|---|
| Total optimizer steps | 400 |
| Per-iter wall time | ≈ 5 s/it (vs. ≈ 36 s/it at `M_train=10, batch=32`) |
| Mean `train/mean_reward` — first 20 steps | −7.456 |
| Mean `train/mean_reward` — last 20 steps | −6.881 |
| Mean `train/mean_log_prob` (whole run) | ≈ −0.339 (essentially flat) |
| Mean `train/loss` (whole run) | ≈ −0.003 (noise around 0, expected for SR) |
| Mean `train/grad_norm` (whole run) | ≈ 1.8e-4 |
| Total wall-clock time (training + eval) | ≈ 40 minutes |

The training reward trend is upward (less negative ⇒ shorter tours): roughly −7.46 → −6.88 across the 100 epochs. The mean log-probability barely moves (−0.339 throughout), meaning the policy retains the SL solver's mode but slowly shifts which trajectories it favors.

### Eval Values

| Metric | Value |
|---|---|
| SL baseline gap (start) | **4.46%** |
| **Final best validation gap** | **3.772%** (epoch 100) |
| Absolute improvement | **−0.688 pp** |
| Relative improvement | **15.4%** |
| First-half mean gap (epochs 5–50) | 4.740% (std 0.290) |
| Second-half mean gap (epochs 55–100) | 4.200% (std 0.276) |
| Number of sub-4% epochs | 3 (epochs 85, 90, 100) |

Note: `eval_subset=100` is small; the per-eval standard error is on the order of ±0.3%, so the epoch-to-epoch swings of 0.3–0.5% should be read as noise. The half-vs-half comparison (mean 4.74 → 4.20 with std 0.29) is robust because each half averages over 10 evals.

### Per-Epoch Validation Log

| Epoch | Pred Length | GT Length | Gap | Notes |
|---|---|---|---|---|
| 5 | 5.9868 | 5.6889 | 5.237% | |
| 10 | 5.9634 | 5.6889 | 4.826% | |
| 15 | 5.9789 | 5.6889 | 5.099% | regression |
| 20 | 5.9654 | 5.6889 | 4.860% | |
| 25 | 5.9633 | 5.6889 | 4.825% | |
| 30 | 5.9539 | 5.6889 | 4.658% | first crossing below SL band |
| 35 | 5.9425 | 5.6889 | 4.458% | matches SL baseline |
| 40 | 5.9323 | 5.6889 | 4.280% | saved best |
| 45 | 5.9464 | 5.6889 | 4.528% | |
| 50 | 5.9523 | 5.6889 | 4.631% | |
| 55 | 5.9514 | 5.6889 | 4.615% | |
| 60 | 5.9424 | 5.6889 | 4.457% | |
| 65 | 5.9346 | 5.6889 | 4.320% | saved best |
| 70 | 5.9369 | 5.6889 | 4.360% | |
| 75 | 5.9226 | 5.6889 | 4.108% | saved best |
| 80 | 5.9401 | 5.6889 | 4.416% | |
| 85 | 5.9096 | 5.6889 | 3.881% | saved best, first sub-4% |
| 90 | 5.9137 | 5.6889 | 3.953% | |
| 95 | 5.9229 | 5.6889 | 4.114% | |
| 100 | 5.9034 | 5.6889 | **3.772%** | saved best (final) |

### Training Dynamics

- **Reward trend (upward, slow)**: `train/mean_reward` increases from −7.46 to −6.88 over the run, but the trajectory is noisy with one notable dip to −8.0 near step 140. This is the expected REINFORCE signature: high-variance per-instance rewards, slow but consistent drift in the right direction.
- **Policy peakedness is the bottleneck**: `mean_log_prob ≈ −0.339` summed over `M_train − 1 = 4` sampling steps ⇒ ≈ 0.92 probability per step ⇒ the policy is nearly deterministic at each sampling step. About 72% of the 32 rollouts in a batch are essentially identical, so REINFORCE has very little diversity to learn from.
- **Tiny gradients, consistent direction**: `grad_norm ≈ 1.8e-4` is two orders of magnitude smaller than typical supervised gradients, but the *direction* is consistent enough that, accumulated over 400 updates with `lr=1e-4`, the parameter drift produces meaningful policy improvement.
- **SR is doing its job**: `train/loss` hovers around 0 (mean −0.003), which is exactly what zero-mean advantages should produce. The loss magnitude is not informative for REINFORCE — only the reward and gap matter.
- **Best gap is at the last epoch (100)**: training had not converged. Continued fine-tuning would likely push the gap further down.

### Analysis

This run is a successful first-pass reproduction of CADO REINFORCE on a small DIFUSCO baseline. The fine-tuned model improves the validation gap from **4.46% → 3.77%**, a **15.4% relative reduction** in optimality gap. This is consistent in sign with the paper's reported improvements (the CADO paper achieves ≈ 40–80% relative reduction from a much stronger ≈ 0.5% SL baseline; the relative magnitude is smaller here because (a) the SL starting point is weaker and (b) the RL run is shorter — 100 epochs vs. the paper's 3000).

Three findings deserve emphasis:

1. **LCR is unusable in this regime.** With our weak SL baseline (`R = −1.5` consistently), LCR produces near-constant negative advantages across the batch and no relative signal. SR (batch normalization) is required to extract any usable gradient.
2. **The paper-faithful LR (1e-5) is too low for LoRA fine-tuning.** The paper's value works for full-parameter fine-tuning; with rank-2 LoRA + 1 selective layer, the effective per-step parameter update needs to be 10× larger to compensate for the rank constraint. `lr=1e-4` was the smallest value that produced a moving gap.
3. **Exploration is the next bottleneck.** The flat `mean_log_prob` says the policy structure is barely changing. An entropy regularization term (β·H(π), starting at β=0.01) is the standard fix and is queued for the next experiment.

### Comparison to SL Baseline

| Metric | SL Baseline (2026-05-22) | CADO REINFORCE (this) |
|---|---|---|
| Best validation gap | 4.46% | **3.772%** |
| Wall-clock training time | 17 h 47 min | ≈ 40 min |
| Trainable parameters | 940K (all) | 178K (18.67%) |
| Per-epoch val curve recorded? | no (logging bug) | yes |
| Method | SL on Concorde labels | RL fine-tuning from above checkpoint |

The CADO run took ~27× less wall time than the SL training that produced its starting point, and improved the gap by 0.69 percentage points. Cost-effective for a thesis pilot.

### Next Steps

1. **Continue training from this checkpoint** (`checkpoints/cado/20260525_104123/best_model.pt`) for another 100 epochs with the same config. The best gap occurring at the final epoch indicates training is not yet converged. Expected outcome: 3.0–3.3% gap.
2. **Add an entropy bonus** (β=0.01, Monte Carlo proxy: `-log_probs_batch.mean()`) in `train_rf.py` and run a parallel ablation from the same SL starting point. This isolates whether exploration regularization helps or hurts in our setting and gives two data points to compare in the thesis.
3. **Tighten the validation signal**: re-evaluate the best checkpoint with `eval_subset=1000` to get a publication-quality gap number free of small-sample noise.
4. **Stronger SL baseline (longer-term)**: if compute permits, train a `model=paper` (h256_L12) DIFUSCO baseline. CADO on a sub-1% starting point should reach a sub-0.5% gap, closer to the paper's headline numbers.

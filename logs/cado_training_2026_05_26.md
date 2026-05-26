# Experiment Log

## Experiment 5: Extended CADO REINFORCE Fine-Tuning (1000 Epochs)

**Date**: 2026-05-26
**wandb run**: `difusco-tsp / auqix4d1` (exported via `wandb/main.py`)
**Goal**: Continue CADO REINFORCE fine-tuning to 1000 epochs (10× the pilot in Experiment 4) and determine whether the validation gap keeps improving with longer training.

**Data sources**: `exports/run_history_val.csv`, `exports/run_history_train.csv`

### Background

Experiment 4 ran 100 epochs (400 optimizer steps) and reported a best validation gap of 3.77% with `eval_subset=100`. This run uses the same SL checkpoint and hyperparameters but extends to **1000 epochs** (4000 steps) and increases `eval_subset` to **200** for a tighter validation estimate. The per-epoch curves in the two experiments are not directly comparable point-for-point because of the eval-subset change.

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
| Optimizer | AdamW |
| Learning rate | 1.0e-4 |
| Weight decay | 0.0 |
| Gradient clip | 1.0 |
| `M_train` (denoising steps in rollout) | 5 |
| `M_eval` (denoising steps in eval) | 50 |
| Schedule type | cosine |
| Epochs | **1000** |
| Samples per epoch | 128 |
| Batch size | 32 |
| Updates per epoch | 4 (= 128 / 32) |
| Total optimizer steps | **4000** |
| Eval cadence | every 10 epochs |
| Eval subset | **200** instances |
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
| Total optimizer steps | 4000 |
| Unique train log rows | 910 (deduplicated by `train/global_step`) |
| Mean `train/mean_reward` (whole run) | −7.490 |
| Mean `train/mean_reward` — first 20 steps | −7.363 |
| Mean `train/mean_reward` — last 20 steps | −6.910 |
| Reward improvement (first → last 20) | +0.453 (less negative ⇒ shorter tours) |
| Mean `train/mean_log_prob` (whole run) | −0.339 (essentially flat) |
| Mean `train/loss` (whole run) | −0.003 (noise around 0, expected for SR) |
| Mean `train/grad_norm` (whole run) | 1.54e-4 |
| Estimated wall-clock time | ≈ 6–7 h (scaled from Exp 4: ≈ 40 min / 400 steps) |

The training reward drifts upward over the full run (−7.36 → −6.91 comparing first vs. last 20 logged steps), but the trajectory is noisy with occasional dips (e.g. reward −7.26 near step 2996). Policy log-probability remains pinned near −0.339 throughout — the policy stays nearly deterministic at each sampling step.

### Eval Values

| Metric | Value |
|---|---|
| SL baseline gap (start) | **4.46%** |
| **Final best validation gap** | **3.955%** (epoch 970) |
| Final epoch gap (epoch 1000) | 4.031% |
| Absolute improvement vs. SL | **−0.505 pp** |
| Relative improvement vs. SL | **11.3%** |
| Mean gap (all 100 evals) | 4.579% (std 0.252) |
| First-half mean gap (epochs 10–500) | 4.653% (std 0.255) |
| Second-half mean gap (epochs 510–1000) | 4.505% (std 0.229) |
| Q1 / Q2 / Q3 / Q4 mean gap | 4.638 / 4.669 / 4.498 / 4.513% |
| Sub-4% epochs | 2 (epochs 970, 990) |
| Worst eval gap | 5.323% (epoch 420) |

### Per-Epoch Validation Log (every 50 epochs)

Full curve: 100 eval points in `exports/run_history_val.csv`.

| Epoch | Pred Length | GT Length | Gap | Notes |
|---|---|---|---|---|
| 50 | 5.9603 | 5.6889 | 4.771% | |
| 100 | 5.9521 | 5.6889 | 4.627% | |
| 150 | 5.9371 | 5.6889 | 4.364% | |
| 200 | 5.9613 | 5.6889 | 4.789% | |
| 250 | 5.9450 | 5.6889 | 4.503% | |
| 300 | 5.9700 | 5.6889 | 4.943% | |
| 350 | 5.9533 | 5.6889 | 4.649% | |
| 400 | 5.9726 | 5.6889 | 4.988% | |
| 450 | 5.9483 | 5.6889 | 4.561% | |
| 500 | 5.9351 | 5.6889 | 4.328% | mid-run low |
| 550 | 5.9583 | 5.6889 | 4.737% | |
| 600 | 5.9357 | 5.6889 | 4.339% | |
| 650 | 5.9505 | 5.6889 | 4.599% | |
| 700 | 5.9370 | 5.6889 | 4.362% | |
| 750 | 5.9452 | 5.6889 | 4.505% | |
| 800 | 5.9474 | 5.6889 | 4.545% | |
| 850 | 5.9703 | 5.6889 | 4.948% | |
| 900 | 5.9443 | 5.6889 | 4.489% | |
| 950 | 5.9627 | 5.6889 | 4.813% | |
| 970 | 5.9139 | 5.6889 | **3.955%** | **best** |
| 990 | 5.9141 | 5.6889 | 3.959% | second best |
| 1000 | 5.9182 | 5.6889 | 4.031% | final |

### Training Dynamics

- **Slow reward improvement, high variance**: `train/mean_reward` improves by ≈ 0.45 over 4000 steps, but the per-step curve is noisy. REINFORCE with SR advantages produces the expected zero-mean loss (−0.003 mean) while the reward signal carries the learning information.
- **Policy barely moves in probability space**: `mean_log_prob ≈ −0.339` throughout (first 20: −0.339, last 20: −0.341). Summed over `M_train − 1 = 4` sampling steps, this implies ≈ 0.92 probability per step — rollouts within a batch are largely identical, limiting gradient diversity.
- **Gradients stay tiny but non-zero**: `grad_norm ≈ 1.5e-4` mean, consistent with Exp 4. Accumulated over 4000 updates at `lr=1e-4`, this produces measurable but slow gap movement.
- **Validation gap is non-monotonic**: The curve oscillates in a ≈ 4.1–5.3% band for most of training. A brief regression spike occurs at epochs 400–420 (gap peaks at 5.32%). The best gaps cluster late (epochs 970, 990) rather than tracking a smooth descent.
- **Diminishing returns after ~500 epochs**: Q3/Q4 mean gaps (4.50%) are only modestly below Q1/Q2 (4.65%), and the global best occurs at epoch 970 — not at the final epoch. Extra training past ≈ 500 epochs adds variance without a clear downward trend.

### Analysis

Extending REINFORCE from 100 to 1000 epochs yields a **best gap of 3.96%** (epoch 970), a modest further improvement over the SL starting point of 4.46% but **not a dramatic gain from 10× more training**. The half-vs-half comparison (mean 4.65 → 4.51) confirms a slow drift in the right direction, but epoch-to-epoch noise (std ≈ 0.25% with `eval_subset=200`) dominates short-term movement.

Three findings:

1. **Longer training helps, but plateaus.** The best gap (3.96%) is only ≈ 0.2 pp better than the mid-run low at epoch 500 (4.33% mean trajectory, local min 4.08% at epoch 480). Most of the 1000-epoch budget is spent re-sampling the same noisy band rather than discovering new policy modes.
2. **Exploration remains the bottleneck.** Flat `mean_log_prob` across 4000 steps means REINFORCE is optimizing within a narrow set of nearly identical rollouts. This matches Exp 4's diagnosis and motivates the queued entropy-bonus ablation.
3. **Eval-subset change vs. Exp 4.** Experiment 4 used `eval_subset=100` and reported 3.77% at epoch 100; this run's epoch-100 eval at `eval_subset=200` reads 4.63%. The two numbers are not directly comparable — use the SL baseline (4.46%) as the common reference.

### Comparison to Prior Experiments

| Metric | SL Baseline (Exp 3) | CADO 100 ep (Exp 4) | CADO 1000 ep (this) |
|---|---|---|---|
| Best validation gap | 4.46% | 3.77%† | **3.96%** |
| Epochs | 50 (SL) | 100 | 1000 |
| Optimizer steps | 359,900 | 400 | 4000 |
| Eval subset | 500 | 100 | 200 |
| Trainable params | 940K (all) | 178K (18.7%) | 178K (18.7%) |
| Wall-clock time | 17 h 47 min | ≈ 40 min | ≈ 6–7 h (est.) |

† Exp 4 best used `eval_subset=100`; not directly comparable to this run's `eval_subset=200`.

### Next Steps

1. **Re-evaluate the epoch-970 checkpoint** with `eval_subset=1000` to obtain a publication-quality gap number free of small-sample noise.
2. **Entropy-bonus ablation** (β=0.01) from the same SL starting point — addresses the flat `mean_log_prob` / low rollout diversity identified in both Exp 4 and this run.
3. **Early stopping at ≈ 500 epochs** for cost efficiency: the Q3/Q4 mean gap (4.50%) is only marginally better than Q1/Q2 (4.65%), and the global best occurs late by chance rather than sustained improvement.
4. **PPO comparison**: config already supports `algorithm: ppo`; a parallel run with the same budget would test whether clipped updates reduce the high-variance oscillation seen in the validation curve.

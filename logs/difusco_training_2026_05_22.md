# Experiment Log

## Experiment 3: Small Model Re-run (h128_L6), Categorical Diffusion

**Date**: 2026-05-22
**wandb run**: `difusco-tsp / tsp50_h128_L6` (started 2026-05-22 13:13:29 UTC)
**Goal**: Re-run the small DIFUSCO baseline on a fresh checkpoint as the starting point for CADO fine-tuning.

### Configuration

| Parameter | Value |
|---|---|
| Model | h128_L6 (≈ 0.94M params) |
| Hidden dimension | 128 |
| AGNN layers | 6 |
| Diffusion type | Categorical (Bernoulli) |
| Diffusion steps (T) | 1000 |
| Beta schedule | linear, 1e-4 to 0.02 |
| Optimizer | AdamW |
| Learning rate | 2e-4 (cosine annealing) |
| Weight decay | 1e-4 |
| Dropout | 0.0 |
| Epochs | 50 |
| Batch size | 16 |
| Training data | 128,000 TSP-50 instances (Concorde solutions) |
| Train/val split | 0.9 (115,200 / 12,800) |
| Eval subset | 500 instances |
| Eval cadence | every 10 epochs (+ epoch 1, epoch 50) |
| Inference steps | 50 (cosine schedule) |
| Post-processing | Greedy decode + 2-opt |
| Gradient clip | 1.0 |
| Seed | 42 |
| Device | Apple M4 Pro (MPS) |
| `num_workers` | 4 |

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
| Total optimizer steps | 359,900 |
| Steps per epoch | ≈ 7,200 |
| Final train loss (last step) | 0.0280 |
| Median train loss, epochs 3–50 | ≈ 0.020 |
| Final gradient norm | 0.152 |
| Total wall-clock time | **17 h 47 min** (64,005 s) |

Training loss converged within the first ~3 epochs and stayed within the 0.017–0.022 band for the remaining 47 epochs — additional training contributed little.

### Eval Values

| Metric | Value |
|---|---|
| **Final best validation gap** | **4.46%** (`val/final_best_gap_pct`) |
| Per-epoch val gap curve | not recorded (see logging bug below) |

### Known Issue: Logging Bug

This run was affected by a `wandb.log(step=...)` collision in `trainer.py`:

- `train_one_epoch` logged with `step=global_step` (reaching 359k by end of training).
- `fit()` then tried to log per-epoch and validation metrics with `step=epoch` (1, 10, 20, 30, 40, 50). Because Wandb requires `step` to be monotonically non-decreasing, every `val/*` and `train/loss_epoch` log was silently dropped.

Only `val/final_best_gap_pct` (logged without an explicit `step` in `run_train.py`) survived. The per-epoch validation curve for this run is therefore unrecoverable.

**Fix**: `define_metric`-based logging is now wired up in `run_train.py` and the explicit `step=` arguments were removed from `trainer.py`. The next run will produce a full per-epoch curve.

### Comparison to Earlier h128_L6 Run (2026-04-12)

| Metric | 2026-04-12 run | 2026-05-22 run (this) |
|---|---|---|
| Best gap | 4.35% (epoch 40) | 4.46% (final summary) |
| Total steps | 319,900 | 359,900 |
| Wall time | ≈ 9.7 h | 17.8 h |
| Per-epoch val curve available? | yes | no (logging bug) |

The headline gap is in the same ballpark (~4.4%), confirming the result is reproducible. The wall-clock difference is most likely due to `wandb.watch(log="gradients", log_freq=500)` overhead — gradient histograms for ~70 parameters per log step are expensive on MPS.

### Next Steps

1. Re-run with the fixed Wandb logging (epochs=15, eval_every=3) to recover the gap-vs-epoch curve.
2. Train one paper-size baseline (`model=paper`, h256_L12, ≈ 7.3M params).
3. Use the paper-size checkpoint as the starting point for CADO fine-tuning.

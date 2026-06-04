# Experiment Log (from `exports/`)

## Experiment 8: CVRP-50 CADO REINFORCE Fine-Tuning (1000 Epochs)

**Date**: 2026-06-03  
**wandb run**: [`cado-cvrp / kw19gbsj`](https://wandb.ai/goranikin-my-project/cado-cvrp/runs/kw19gbsj)  
**Goal**: Evaluate whether CADO-style REINFORCE fine-tuning can improve the CVRP-50 supervised baseline after the 2-opt validation fix from Experiment 7.

**Primary data sources** (all metrics below are computed from these files only):

| File | Rows | Description |
|---|---:|---|
| `exports/run_history_full.csv` | 1,100 | Interleaved train + validation history |
| `exports/run_history_train.csv` | 1,000 | Rows with `train/global_step` set |
| `exports/run_history_val.csv` | 100 | Rows with `val/gap_pct` set |

Exported with `wandb/main.py` from run `kw19gbsj`; `gradients/*` and `_timestamp` columns removed.

### Export columns

| Column | Present on | Meaning |
|---|---|---|
| `train/global_step` | train rows | Optimizer step index |
| `train/mean_reward` | train rows | Mean rollout reward |
| `train/mean_log_prob` | train rows | Mean sampled log probability |
| `train/loss` | train rows | REINFORCE loss |
| `train/grad_norm` | train rows | Gradient norm after clipping |
| `epoch` | validation rows | Training epoch |
| `val/gap_pct` | validation rows | `(pred - gt) / gt * 100` |
| `val/pred_len` | validation rows | Mean decoded solution length |
| `val/gt_len` | validation rows | Mean ground-truth solution length |
| `val/overcapacity_rate` | validation rows | Fraction of capacity-violating solutions |
| `_runtime` | all rows | Wall seconds since run start |
| `_step` | all rows | W&B internal log index |

### Configuration

| Parameter | Value |
|---|---|
| Problem | CVRP-50 |
| Algorithm | CADO / REINFORCE fine-tuning |
| Eval cadence | every 10 epochs |
| Epochs | **1000** |
| Total optimizer steps | **4000** (`train/global_step`: 0 -> 3996) |
| Validation GT length | 14.5845 |
| Validation overcapacity | 0% on all evals |
| Export source | `wandb/main.py` (`RUN_PATH = "/goranikin-my-project/cado-cvrp/runs/kw19gbsj"`) |

The run config itself is not present in the exported CSVs, so this log only records fields visible in the history export plus the run path from `wandb/main.py`.

### Summary

| Metric | Value |
|---|---:|
| Wall-clock | **7 h 0 min 36 s** (`_runtime` max = 25,236 s) |
| Logged train rows | 1,000 |
| Validation rows | 100 |
| First validation gap | 28.19% (epoch 10) |
| **Best validation gap** | **20.24%** (epoch 470) |
| Final validation gap | 47.16% (epoch 1000) |
| Worst validation gap | 47.37% (epoch 800) |
| Mean validation gap | 28.61% (std 7.37) |
| First-half mean gap | 24.24% (epochs 10-500) |
| Second-half mean gap | 32.98% (epochs 510-1000) |
| Sub-25% evals | 37 / 100 |
| Sub-26% evals | 64 / 100 |
| Overcapacity rate | 0% on all 100 evals |

Against the Experiment 7 supervised + 2-opt best gap of 25.96%, this run reaches a substantially better best checkpoint at epoch 470: **20.24%**, a **5.72 pp** absolute reduction. The final checkpoint is unusable, however: by epoch 1000 the validation gap has regressed to **47.16%**, worse than both the starting eval and the prior supervised baseline.

### Training Values

| Metric | Mean | Std | Min | Max | First 20 | Last 20 |
|---|---:|---:|---:|---:|---:|---:|
| `train/mean_reward` | -8.636 | 7.821 | -26.001 | -2.725 | -7.333 | -24.332 |
| `train/mean_log_prob` | -0.339 | 0.0038 | -0.354 | -0.327 | -0.339 | -0.339 |
| `train/loss` | -2.933 | 2.653 | -9.040 | -0.928 | -2.486 | -8.253 |
| `train/grad_norm` | 6.18e-4 | 3.09e-4 | 2.72e-5 | 2.20e-3 | 7.91e-4 | 3.34e-4 |

The reward signal deteriorates badly late in training (`mean_reward` first 20: -7.33; last 20: -24.33), matching the validation collapse. `mean_log_prob` remains almost unchanged around -0.339, so the policy probability profile is flat even while decoded solution quality moves substantially.

### Validation Log

Full curve: 100 eval points in `exports/run_history_val.csv`.

| Epoch | Pred len | GT len | Gap | Notes |
|---:|---:|---:|---:|---|
| 50 | 18.3377 | 14.5845 | 25.73% | |
| 100 | 18.2901 | 14.5845 | 25.41% | |
| 150 | 18.1714 | 14.5845 | 24.59% | |
| 200 | 18.2315 | 14.5845 | 25.01% | |
| 250 | 18.3726 | 14.5845 | 25.97% | |
| 300 | 18.2341 | 14.5845 | 25.02% | |
| 350 | 18.1812 | 14.5845 | 24.66% | |
| 400 | 17.7711 | 14.5845 | 21.85% | |
| 450 | 17.7688 | 14.5845 | 21.83% | |
| 470 | 17.5365 | 14.5845 | **20.24%** | **best** |
| 500 | 17.9947 | 14.5845 | 23.38% | |
| 550 | 18.2510 | 14.5845 | 25.14% | |
| 600 | 18.2027 | 14.5845 | 24.81% | |
| 650 | 18.3192 | 14.5845 | 25.61% | |
| 700 | 18.2637 | 14.5845 | 25.23% | |
| 750 | 18.6143 | 14.5845 | 27.63% | degradation starts |
| 800 | 21.4936 | 14.5845 | **47.37%** | **worst** |
| 850 | 20.7314 | 14.5845 | 42.15% | |
| 900 | 20.6336 | 14.5845 | 41.48% | |
| 950 | 20.3642 | 14.5845 | 39.63% | |
| 1000 | 21.4630 | 14.5845 | 47.16% | final |

### Training Dynamics

- **Useful middle checkpoint**: The run improves over the supervised 2-opt baseline and reaches **20.24%** at epoch 470. Most of the useful gains occur between epochs 330 and 500, where several evals fall near 20-23%.
- **Late collapse**: After epoch 760, validation quality breaks down. Q4 mean gap is **40.31%**, compared with **23.42%** in Q2. The decoded length jumps from the high-17/low-18 range to roughly 20-21.5.
- **No capacity failures**: `val/overcapacity_rate` is 0% throughout, so the regression is not caused by infeasible routes. The model is producing feasible but much longer routes.
- **Policy probability is not diagnostic enough**: `train/mean_log_prob` stays flat near -0.339 across the run, while reward and validation quality diverge. Reward and validation gap should drive checkpoint selection.
- **Reward collapse mirrors validation collapse**: The last 20 train rewards average -24.33, far worse than the first 20 (-7.33). This suggests the optimizer is not merely overfitting the validation subset; the training rollout signal itself has degraded.

### Analysis

CADO fine-tuning is promising for CVRP but unstable in this run. The best checkpoint improves the previous CVRP supervised + 2-opt result from **25.96% to 20.24%**, which is the first clear move toward a stronger CVRP baseline. The same run also shows that a 1000-epoch budget is unsafe without checkpoint selection: the final model is dramatically worse than the best model and worse than the supervised starting point.

The important result is therefore **not** the final checkpoint, but the existence of a mid-run checkpoint around epoch 470 that improves routing quality while preserving feasibility. The late collapse argues for shorter CADO schedules, validation-based early stopping, and stronger guardrails around reward drift.

### Comparison

| Metric | Exp 7: CVRP SL + 2-opt | Exp 8: CVRP CADO |
|---|---:|---:|
| Best validation gap | 25.96% (epoch 10) | **20.24% (epoch 470)** |
| Final validation gap | 35.51% (epoch 50) | 47.16% (epoch 1000) |
| Validation GT length | 14.306 | 14.585 |
| Best pred length | 18.019 | **17.536** |
| Overcapacity | 0% | 0% |
| Wall-clock | 2 h 9 min | 7 h 1 min |

The validation datasets are not necessarily identical (`val/gt_len` differs: 14.306 vs 14.585), so compare gaps rather than raw tour lengths.

### Next Steps

1. **Use the epoch-470 checkpoint**, not the final checkpoint, for any downstream evaluation.
2. Re-evaluate epoch 470 on a larger fixed validation subset to confirm that the 20.24% gap is stable.
3. Add validation-based early stopping or keep-best checkpointing for CADO CVRP runs.
4. Run a shorter CADO schedule centered around 500 epochs, with monitoring that halts if reward drops below the early-run band.
5. Investigate the late reward collapse around epochs 760-800 by comparing decoded route structure before and after the transition.

### Reproducing the exports

```bash
uv run python wandb/main.py
```

`RUN_PATH` in `wandb/main.py` must point at `kw19gbsj` (current default).

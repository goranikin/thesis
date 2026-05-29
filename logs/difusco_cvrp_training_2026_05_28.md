# Experiment Log

## Experiment 6: CVRP-50 Paper Model Supervised Training (Remote)

**Date**: 2026-05-28  
**wandb run**: [`difusco-cvrp / ch78lr8f`](https://wandb.ai/goranikin-my-project/difusco-cvrp/runs/ch78lr8f) (`cvrp50_h256_L12`)  
**Goal**: Train the paper-size DIFUSCO model (h256_L12) on CVRP-50 with categorical diffusion on a remote A100, establishing the first supervised-learning baseline before CADO fine-tuning.

**Data sources**: `exports/run_history_full.csv`, `exports/run_history_train.csv`, `exports/run_history_val.csv` (exported via `wandb/main.py`, gradient columns stripped)

### Background

TSP experiments (Exp 3–5) used the small model (h128_L6, ≈0.94M params) and reached ≈4.4% gap after supervised training. This run is the **first CVRP end-to-end training** with the **paper configuration** (h256_L12, ≈7.3M params) on 128k PyVRP-generated instances (`cvrp50-50_128000_pyvrp.txt`). It was launched on VESSL (`vessl_ai/difusco_cvrp_train.json`) with `training=remote`.

### Configuration

| Parameter | Value |
|---|---|
| Model | h256_L12 (paper; ≈7.3M params) |
| Hidden dimension | 256 |
| AGNN layers | 12 |
| Problem | CVRP-50 (50 customers + depot) |
| Diffusion type | Categorical (Bernoulli) |
| Diffusion steps (T) | 1000 |
| Beta schedule | linear, 1e-4 → 0.02 |
| Optimizer | AdamW |
| Learning rate | 2e-4 (cosine annealing) |
| Weight decay | 1e-4 |
| Dropout | 0.0 |
| Epochs | 50 |
| Batch size | 64 |
| Training data | `cvrp50-50_128000_pyvrp.txt` (128k instances) |
| Train/val split | 0.9 (115,200 / 12,800) |
| `M_eval` (inference steps) | 50 (cosine schedule) |
| Post-processing (eval) | Greedy multi-route decode |
| Eval cadence | every 5 epochs (+ epoch 1, epoch 50) |
| Eval subset | 500 validation instances |
| Gradient clip | 1.0 |
| Log interval | every 100 batches |
| Seed | 42 |
| Checkpoint dir | `/mount/checkpoints` |
| W&B project | `difusco-cvrp` |

### Computing Resources

| Resource | Specification |
|---|---|
| Platform | VESSL (`cluster-betelgeuse`) |
| GPU | 1× NVIDIA A100 (`resourcespec-a100x1`) |
| Image | `quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5` |
| Data / checkpoints | VESSL object volume `@ /mount` |
| Total wall-clock time | **2 h 9 min** (`_runtime` ≈ 7,712 s) |

### Training Values

| Metric | Value |
|---|---|
| Logged `train/global_step` range | 0 → 16,862 |
| Batches per epoch (inferred) | ≈337 |
| Total optimizer steps (inferred) | ≈16,850 (337 × 50) |
| Train log rows (`train/global_step`) | 200 |
| `train/loss_step` — first logged | 0.868 |
| `train/loss_step` — last logged | 0.068 |
| `train/loss_step` — mean | 0.063 |
| `train/loss_epoch` — epoch 1 | 0.117 |
| `train/loss_epoch` — epoch 50 | 0.053 |
| `train/loss_epoch` — minimum | 0.049 (epoch 49) |
| `train/grad_norm` — first logged | 10.85 |
| `train/grad_norm` — last logged | 0.069 |
| `train/grad_norm` — mean | 0.174 |
| `train/lr` — start / end | 2.0e-4 → 0 |

Denosing cross-entropy decreases steadily (epoch loss 0.117 → 0.053), and per-step gradients shrink from an initial spike (10.85) to O(0.1). **Training loss alone does not indicate useful routing quality on CVRP** — see validation below.

### Eval Values

| Metric | Value |
|---|---|
| **Best validation gap** | **29.98%** (epochs 1 and 5; checkpoint saved here) |
| `val/final_best_gap_pct` | 29.98% |
| Final epoch gap (epoch 50) | 33.66% |
| Mean gap (all 11 evals) | 33.72% (std 3.10%) |
| GT tour length (constant) | 14.306 |
| Pred tour length — best eval (epoch 5) | 18.594 |
| Pred tour length — final (epoch 50) | 19.120 |
| `val/overcapacity_rate` | **0%** on every eval |
| Predicted routes (epoch 1 / 50) | ≈16.1 / 14.9 (GT route count ≈ implicit via labels) |

The model satisfies capacity constraints (no overcapacity violations) but produces tours **≈30% longer** than PyVRP ground truth. The best gap is reached at **epoch 5** and never improves; later epochs drift to 33–40% gap despite lower training loss.

### Per-Epoch Validation Log

Full curve: 11 eval points in `exports/run_history_val.csv`.

| Epoch | Pred Length | GT Length | Gap | Routes (pred) | Overcap | Notes |
|---|---|---|---|---|---|---|
| 1 | 18.595 | 14.306 | 29.98% | 16.06 | 0% | **best** (tied) |
| 5 | 18.594 | 14.306 | 29.98% | 14.74 | 0% | **best** (tied) |
| 10 | 18.764 | 14.306 | 31.17% | 14.82 | 0% | |
| 15 | 19.552 | 14.306 | 36.68% | 15.48 | 0% | |
| 20 | 18.989 | 14.306 | 32.74% | 14.66 | 0% | |
| 25 | 19.683 | 14.306 | 37.59% | 15.28 | 0% | |
| 30 | 19.988 | 14.306 | 39.72% | 15.76 | 0% | worst |
| 35 | 19.085 | 14.306 | 33.41% | 14.89 | 0% | |
| 40 | 19.033 | 14.306 | 33.04% | 14.73 | 0% | |
| 45 | 19.023 | 14.306 | 32.97% | 14.83 | 0% | |
| 50 | 19.120 | 14.306 | 33.66% | 14.87 | 0% | final |

### Training Dynamics

- **Loss–gap decoupling**: `train/loss_epoch` falls by more than 50% while `val/gap_pct` worsens after epoch 5. The diffusion objective is optimizing edge classification without translating into shorter decoded routes.
- **Early best, then regression**: `val/best_gap_pct` flatlines at 29.98% from epoch 5 onward; the checkpoint selector correctly freezes the epoch-5 weights as “best,” but that gap is still far from useful.
- **Feasible but long routes**: `val/overcapacity_rate = 0` throughout — greedy decoding respects capacity, but total tour length stays ~4.3 units above GT on average (≈30% gap).
- **Route-count mismatch early**: epoch 1 predicts ≈16 routes vs ≈15 later; excess routes correlate with the highest pred lengths.
- **Remote throughput**: ≈2.1 h for 50 epochs on A100 is reasonable for a 7.3M-param model with 500-instance validation every 5 epochs and `wandb.watch` gradient logging enabled.

### Analysis

This run establishes that the **CVRP training pipeline runs correctly on remote infrastructure** (data mount, VESSL job, W&B logging, checkpointing) but does **not** yet produce a competitive heatmap solver. A ≈30% optimality gap after 50 epochs contrasts sharply with TSP paper-model expectations (sub-5% after comparable SL training).

Likely contributing factors:

1. **Harder decode than TSP**: Multi-route greedy decoding from a heatmap is more brittle than a single Hamiltonian tour; small edge-probability errors compound into longer tours.
2. **Objective mismatch**: Cross-entropy on diffused edge labels may not align with total route length under capacity constraints.
3. **Possible data / label semantics**: GT length uses labeled depot edges (`edge_dist` sum / 2); pred length uses decoded routes — any systematic mismatch would inflate gap (worth verifying on a single instance).
4. **Insufficient or mis-tuned training budget**: Only ≈337 batches/epoch were logged (16.9k total steps vs ≈90k expected if the full 115k train split were consumed at batch 64). Confirm the mounted dataset size matches 128k instances.

### Comparison to TSP Baselines

| Metric | TSP h128_L6 (Exp 3) | TSP CADO target | CVRP h256_L12 (this) |
|---|---|---|---|
| Best validation gap | 4.46% | ≈4% SL → <4% RL | **29.98%** |
| Model size | 0.94M | 0.94M (CADO) | 7.3M |
| Problem | TSP-50 | TSP-50 | CVRP-50 |
| Eval post-process | Greedy + 2-opt | Greedy + 2-opt | Greedy multi-route |
| Wall time | 17.8 h (MPS) | — | 2.1 h (A100) |

### Next Steps

1. **Verify dataset and loader size** on `/mount/data` — confirm 128k instances and ~1,800 batches/epoch at `batch_size=64`.
2. **Sanity-check decoding** on one instance (heatmap visualization, route count, length vs GT decomposition).
3. **Disable or reduce `wandb.watch` gradient logging** for production runs (mirrors TSP Exp 3 wall-time lesson).
4. **Short local smoke test** (`model=small`, `training=local`) to compare gap scale before another 7M-param remote job.
5. **Only after SL gap < ~10%**: queue CADO CVRP fine-tuning (`cado_cvrp_config.yaml`) from the best checkpoint.

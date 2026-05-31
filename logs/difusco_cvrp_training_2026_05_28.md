# Experiment Log (from `exports/`)

> **Superseded** — exports were re-exported from run `myqt6q9i` (2-opt). See [`difusco_cvrp_training_exports_2026_05_29.md`](difusco_cvrp_training_2026_05_29.md).

## Experiment 6: CVRP-50 Paper Model — Supervised Training (Remote)

**Date**: 2026-05-28  
**wandb run**: [`difusco-cvrp / ch78lr8f`](https://wandb.ai/goranikin-my-project/difusco-cvrp/runs/ch78lr8f) (`cvrp50_h256_L12`)  
**Goal**: First remote supervised baseline for CVRP-50 with the paper-size DIFUSCO model (h256_L12).

**Primary data sources** (all metrics below are computed from these files only):

| File | Rows | Description |
|---|---|---|
| `exports/run_history_full.csv` | 262 | Interleaved train + val + system fields |
| `exports/run_history_train.csv` | 200 | Rows with `train/global_step` set |
| `exports/run_history_val.csv` | 11 | Rows with `val/gap_pct` set |

Exported with `wandb/main.py` from run `ch78lr8f`; `gradients/*` and `_timestamp` columns removed.

### Export columns

| Column | Present on | Meaning |
|---|---|---|
| `train/global_step` | train rows | Optimizer step index |
| `train/loss_step` | train rows | Per-batch diffusion CE loss |
| `train/loss_epoch` | epoch rows | Mean train loss for the epoch |
| `train/grad_norm` | train rows | Gradient norm after clipping |
| `train/lr` | epoch rows | LR after cosine scheduler step |
| `epoch` | epoch / val rows | Training epoch (1–50) |
| `val/gap_pct` | val rows | `(pred − gt) / gt × 100` |
| `val/best_gap_pct` | val rows | Running best gap so far |
| `val/final_best_gap_pct` | summary | Final best gap (29.98%) |
| `val/pred_tour_length` | val rows | Mean decoded tour length |
| `val/gt_tour_length` | val rows | Mean GT tour length (14.306) |
| `val/num_routes_pred` | val rows | Mean predicted route count |
| `val/overcapacity_rate` | val rows | Fraction of instances with capacity violations |
| `_runtime` | all | Wall seconds since run start |
| `_step` | all | W&B internal log index |

### Configuration (from W&B run config)

| Parameter | Value |
|---|---|
| Model | h256_L12 (≈7.3M params) |
| Problem | CVRP-50 |
| Data | `/mount/data/cvrp50-50_128000_pyvrp.txt` |
| Train/val split | 0.9 |
| Epochs | 50 |
| Batch size | 64 |
| Learning rate | 2e-4 (cosine) |
| Weight decay | 1e-4 |
| Diffusion T | 1000 (linear β: 1e-4 → 0.02) |
| Inference steps | 50 (cosine schedule) |
| Eval every | 5 epochs |
| Eval subset | 500 instances |
| Log interval | 100 batches |
| Platform | VESSL A100×1 |

### Summary (from exports)

| Metric | Value |
|---|---|
| Wall-clock | **2 h 9 min** (`_runtime` max = 7,712 s) |
| Logged train steps | 0 → 16,862 (200 log points) |
| `train/loss_epoch` (ep 1 → 50) | 0.117 → 0.053 (min **0.049** @ ep 49) |
| `train/loss_step` (first → last log) | 0.868 → 0.068 |
| `train/grad_norm` (first → last log) | 10.85 → 0.069 |
| **Best `val/gap_pct`** | **29.98%** (epochs 1 & 5) |
| `val/final_best_gap_pct` | 29.98% |
| Final `val/gap_pct` (epoch 50) | 33.66% |
| `val/overcapacity_rate` | 0% on all 11 evals |
| Pred tour length range | 18.59 – 19.99 |
| Pred route count range | 14.66 – 16.06 |

### Per-epoch training (`train/loss_epoch`, `train/lr`)

Source: `run_history_full.csv` (50 rows).

| Epoch | Loss | LR |
|---|---|---|
| 1 | 0.1170 | 2.00e-04 |
| 2 | 0.0928 | 1.99e-04 |
| 3 | 0.0845 | 1.98e-04 |
| 4 | 0.0787 | 1.97e-04 |
| 5 | 0.0761 | 1.95e-04 |
| 6 | 0.0732 | 1.93e-04 |
| 7 | 0.0700 | 1.90e-04 |
| 8 | 0.0688 | 1.88e-04 |
| 9 | 0.0641 | 1.84e-04 |
| 10 | 0.0650 | 1.81e-04 |
| 11 | 0.0634 | 1.77e-04 |
| 12 | 0.0658 | 1.73e-04 |
| 13 | 0.0615 | 1.68e-04 |
| 14 | 0.0602 | 1.64e-04 |
| 15 | 0.0605 | 1.59e-04 |
| 16 | 0.0615 | 1.54e-04 |
| 17 | 0.0592 | 1.48e-04 |
| 18 | 0.0607 | 1.43e-04 |
| 19 | 0.0577 | 1.37e-04 |
| 20 | 0.0570 | 1.31e-04 |
| 21 | 0.0574 | 1.25e-04 |
| 22 | 0.0585 | 1.19e-04 |
| 23 | 0.0568 | 1.13e-04 |
| 24 | 0.0562 | 1.06e-04 |
| 25 | 0.0563 | 1.00e-04 |
| 26 | 0.0580 | 9.37e-05 |
| 27 | 0.0557 | 8.75e-05 |
| 28 | 0.0571 | 8.13e-05 |
| 29 | 0.0533 | 7.51e-05 |
| 30 | 0.0548 | 6.91e-05 |
| 31 | 0.0543 | 6.32e-05 |
| 32 | 0.0548 | 5.74e-05 |
| 33 | 0.0546 | 5.18e-05 |
| 34 | 0.0549 | 4.64e-05 |
| 35 | 0.0535 | 4.12e-05 |
| 36 | 0.0526 | 3.63e-05 |
| 37 | 0.0538 | 3.15e-05 |
| 38 | 0.0515 | 2.71e-05 |
| 39 | 0.0529 | 2.29e-05 |
| 40 | 0.0543 | 1.91e-05 |
| 41 | 0.0534 | 1.56e-05 |
| 42 | 0.0511 | 1.24e-05 |
| 43 | 0.0537 | 9.52e-06 |
| 44 | 0.0530 | 7.02e-06 |
| 45 | 0.0518 | 4.89e-06 |
| 46 | 0.0507 | 3.14e-06 |
| 47 | 0.0515 | 1.77e-06 |
| 48 | 0.0551 | 7.89e-07 |
| 49 | 0.0494 | 1.97e-07 |
| 50 | 0.0528 | 0.00e+00 |

### Per-epoch validation

Source: `run_history_val.csv` (11 rows; eval at epochs 1, 5, 10, …, 50).

| Epoch | Pred len | GT len | Gap | Routes | Overcap | Best so far |
|---|---|---|---|---|---|---|
| 1 | 18.595 | 14.306 | 29.98% | 16.06 | 0% | 29.98% |
| 5 | 18.594 | 14.306 | 29.98% | 14.74 | 0% | 29.98% |
| 10 | 18.764 | 14.306 | 31.17% | 14.82 | 0% | 29.98% |
| 15 | 19.552 | 14.306 | 36.68% | 15.48 | 0% | 29.98% |
| 20 | 18.989 | 14.306 | 32.74% | 14.66 | 0% | 29.98% |
| 25 | 19.683 | 14.306 | 37.59% | 15.28 | 0% | 29.98% |
| 30 | 19.988 | 14.306 | 39.72% | 15.76 | 0% | 29.98% |
| 35 | 19.085 | 14.306 | 33.41% | 14.89 | 0% | 29.98% |
| 40 | 19.033 | 14.306 | 33.04% | 14.73 | 0% | 29.98% |
| 45 | 19.023 | 14.306 | 32.97% | 14.83 | 0% | 29.98% |
| 50 | 19.120 | 14.306 | 33.66% | 14.87 | 0% | 29.98% |

Validation statistics (11 evals): mean gap **33.72%**, std **3.10%**, worst **39.72%** (epoch 30).

### Training dynamics

- **Loss improves, gap does not**: Epoch loss falls ~55% while validation gap is flat at ~30% early then drifts to 33–40%. The denoising objective is not yet aligned with decoded tour quality.
- **Best checkpoint at epoch 5**: `val/best_gap_pct` never beats 29.98% after epoch 5; later training does not help routing.
- **Feasible but long tours**: Zero overcapacity violations; predicted lengths stay ~4.3 above GT (~30% gap).
- **Route-count spike at epoch 1**: 16.06 predicted routes vs ~14.7–15.8 later — extra routes correlate with the tied-best but still poor gap at epoch 1.
- **Step budget**: Max `global_step` = 16,862 implies ≈337 batches/epoch, not the ≈1,800 expected for 115k train samples at batch 64. Worth confirming the mounted dataset size on VESSL.

### Analysis

The export files show a **completed 50-epoch remote run** with healthy training-loss convergence but **poor routing performance** (~30% optimality gap). Compared to TSP supervised baselines (~4.4% with h128_L6), CVRP decoding from heatmaps is substantially harder: greedy multi-route construction (`difusco/cvrp/decoding.py`) may amplify edge-probability errors, and cross-entropy on diffused edges may not directly optimize tour length under capacity.

The checkpoint saved at epoch 5 (best gap 29.98%) is the only artifact worth inspecting before further training changes.

### Comparison

| | TSP SL (Exp 3) | CVRP SL (this run) |
|---|---|---|
| Model | h128_L6 (0.94M) | h256_L12 (7.3M) |
| Best val gap | 4.46% | **29.98%** |
| Wall time | 17.8 h (MPS) | 2.1 h (A100) |

### Next steps

1. Confirm dataset size and batches/epoch on the remote mount.
2. Sanity-check `greedy_decode_cvrp` + length computation on a single instance (heatmap vs GT edges).
3. Re-run validation locally with `use_2opt=true` (now in `configs/inference/default.yaml`) to see if gap drops without retraining.
4. Short `model=small` local smoke test before another full remote job.
5. Defer CADO CVRP until SL gap is in a reasonable range (<10%).

### Reproducing the exports

```bash
uv run python wandb/main.py
```

Writes the three CSVs under `exports/` from `RUN_PATH` in `wandb/main.py`.

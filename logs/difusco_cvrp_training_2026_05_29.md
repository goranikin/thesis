# Experiment Log (from `exports/`)

## Experiment 7: CVRP-50 Paper Model — Supervised Training with 2-opt (Remote)

**Date**: 2026-05-29  
**wandb run**: [`difusco-cvrp / myqt6q9i`](https://wandb.ai/goranikin-my-project/difusco-cvrp/runs/myqt6q9i) (`cvrp50_h256_L12`)  
**Goal**: Re-run the paper-size CVRP-50 supervised baseline on VESSL with **per-route 2-opt** enabled at validation (`inference.use_2opt=true`), after Experiment 6 (run `ch78lr8f`) reached ~30% gap with greedy decode only.

**Primary data sources** (all metrics below are computed from these files only):

| File | Rows | Description |
|---|---|---|
| `exports/run_history_full.csv` | 262 | Interleaved train + val + system fields |
| `exports/run_history_train.csv` | 200 | Rows with `train/global_step` set |
| `exports/run_history_val.csv` | 11 | Rows with `val/gap_pct` set |

Exported with `wandb/main.py` from run `myqt6q9i`; `gradients/*` and `_timestamp` columns removed.

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
| `val/final_best_gap_pct` | summary | Final best gap (25.96%) |
| `val/pred_tour_length` | val rows | Mean decoded tour length (after greedy + 2-opt) |
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
| **Post-processing** | **Greedy decode + per-route 2-opt** (`max_2opt_iterations=100`) |
| Eval every | 5 epochs |
| Eval subset | 500 instances |
| Log interval | 100 batches |
| Platform | VESSL A100×1 |

### Summary (from exports)

| Metric | Value |
|---|---|
| Wall-clock | **2 h 9 min** (`_runtime` max = 7,717 s) |
| Logged train steps | 0 → 16,862 (200 log points) |
| `train/loss_epoch` (ep 1 → 50) | 0.114 → 0.053 (min **0.049** @ ep 49) |
| `train/loss_step` (first → last log) | 0.868 → 0.068 |
| `train/grad_norm` (first → last log) | 10.85 → 0.057 |
| **Best `val/gap_pct`** | **25.96%** (epoch 10) |
| `val/final_best_gap_pct` | 25.96% |
| Final `val/gap_pct` (epoch 50) | 35.51% |
| `val/overcapacity_rate` | 0% on all 11 evals |
| Pred tour length @ best eval | 18.019 |
| Pred tour length @ final eval | 19.385 |
| Pred route count @ best / final | 14.57 / 15.76 |

### Per-epoch training (`train/loss_epoch`, `train/lr`)

Source: `run_history_full.csv` (50 rows).

| Epoch | Loss | LR |
|---|---|---|
| 1 | 0.1142 | 2.00e-04 |
| 2 | 0.0908 | 1.99e-04 |
| 3 | 0.0816 | 1.98e-04 |
| 4 | 0.0781 | 1.97e-04 |
| 5 | 0.0760 | 1.95e-04 |
| 6 | 0.0725 | 1.93e-04 |
| 7 | 0.0714 | 1.90e-04 |
| 8 | 0.0698 | 1.88e-04 |
| 9 | 0.0637 | 1.84e-04 |
| 10 | 0.0649 | 1.81e-04 |
| 11 | 0.0634 | 1.77e-04 |
| 12 | 0.0639 | 1.73e-04 |
| 13 | 0.0609 | 1.68e-04 |
| 14 | 0.0598 | 1.64e-04 |
| 15 | 0.0606 | 1.59e-04 |
| 16 | 0.0613 | 1.54e-04 |
| 17 | 0.0589 | 1.48e-04 |
| 18 | 0.0602 | 1.43e-04 |
| 19 | 0.0575 | 1.37e-04 |
| 20 | 0.0567 | 1.31e-04 |
| 21 | 0.0572 | 1.25e-04 |
| 22 | 0.0585 | 1.19e-04 |
| 23 | 0.0567 | 1.13e-04 |
| 24 | 0.0562 | 1.06e-04 |
| 25 | 0.0564 | 1.00e-04 |
| 26 | 0.0581 | 9.37e-05 |
| 27 | 0.0558 | 8.75e-05 |
| 28 | 0.0571 | 8.13e-05 |
| 29 | 0.0532 | 7.51e-05 |
| 30 | 0.0547 | 6.91e-05 |
| 31 | 0.0543 | 6.32e-05 |
| 32 | 0.0548 | 5.74e-05 |
| 33 | 0.0546 | 5.18e-05 |
| 34 | 0.0549 | 4.64e-05 |
| 35 | 0.0535 | 4.12e-05 |
| 36 | 0.0525 | 3.63e-05 |
| 37 | 0.0538 | 3.15e-05 |
| 38 | 0.0514 | 2.71e-05 |
| 39 | 0.0529 | 2.29e-05 |
| 40 | 0.0542 | 1.91e-05 |
| 41 | 0.0533 | 1.56e-05 |
| 42 | 0.0510 | 1.24e-05 |
| 43 | 0.0537 | 9.52e-06 |
| 44 | 0.0530 | 7.02e-06 |
| 45 | 0.0517 | 4.89e-06 |
| 46 | 0.0506 | 3.14e-06 |
| 47 | 0.0515 | 1.77e-06 |
| 48 | 0.0550 | 7.89e-07 |
| 49 | 0.0493 | 1.97e-07 |
| 50 | 0.0527 | 0.00e+00 |

### Per-epoch validation

Source: `run_history_val.csv` (11 rows; eval at epochs 1, 5, 10, …, 50).

| Epoch | Pred len | GT len | Gap | Routes | Overcap | Best so far |
|---|---|---|---|---|---|---|
| 1 | 18.769 | 14.306 | 31.20% | 16.53 | 0% | 31.20% |
| 5 | 18.224 | 14.306 | 27.39% | 14.84 | 0% | 27.39% |
| 10 | 18.019 | 14.306 | **25.96%** | 14.57 | 0% | **25.96%** |
| 15 | 18.220 | 14.306 | 27.36% | 14.75 | 0% | 25.96% |
| 20 | 18.624 | 14.306 | 30.19% | 14.90 | 0% | 25.96% |
| 25 | 18.715 | 14.306 | 30.82% | 14.71 | 0% | 25.96% |
| 30 | 19.158 | 14.306 | 33.92% | 15.45 | 0% | 25.96% |
| 35 | 19.277 | 14.306 | 34.75% | 15.34 | 0% | 25.96% |
| 40 | 19.187 | 14.306 | 34.12% | 15.52 | 0% | 25.96% |
| 45 | 19.325 | 14.306 | 35.09% | 15.74 | 0% | 25.96% |
| 50 | 19.385 | 14.306 | 35.51% | 15.76 | 0% | 25.96% |

Validation statistics (11 evals): mean gap **31.48%**, std **3.45%**.

### Training dynamics

- **Early improvement then regression**: Gap falls 31.2% → 27.4% → **25.96%** by epoch 10, then climbs monotonically to **35.51%** at epoch 50 while training loss still decreases.
- **Best checkpoint at epoch 10**: `val/best_gap_pct` freezes at 25.96% after epoch 10; later epochs never beat it.
- **2-opt helps vs greedy-only baseline**: Compared to Experiment 6 (`ch78lr8f`, no 2-opt), best gap improves **29.98% → 25.96%** (~4.0 pp), with the best epoch shifting from 1/5 to 10.
- **Loss–gap decoupling persists**: Epoch loss drops ~54% (0.114 → 0.053) but validation gap worsens after epoch 10 — the model learns edge classification without sustained routing gains.
- **Feasible tours**: `val/overcapacity_rate = 0%` on every eval; capacity is respected but tours remain ~26–35% longer than GT.
- **Route count drift**: Predicted routes rise from 14.57 (best) to 15.76 (final), correlating with longer tour lengths in later epochs.

### Analysis

Enabling per-route 2-opt at validation is a clear win for the **best** checkpoint (≈4 pp gap reduction vs Experiment 6), but the run still ends far from TSP-scale SL quality (~4% gap). The U-shaped validation curve — best at epoch 10, worst at epoch 50 — suggests **early stopping** around epoch 10–15 would have been appropriate; continuing to epoch 50 only degrades decoded tour quality despite lower diffusion loss.

The heatmap + greedy + 2-opt pipeline produces feasible solutions but the underlying edge probabilities do not yet encode routing structure well enough for sub-10% gaps. Next debugging should focus on decode/length semantics and dataset coverage before scaling training budget or starting CADO.

### Comparison

| Metric | Exp 6 (`ch78lr8f`) | Exp 7 (this run) |
|---|---|---|
| Val decode | Greedy only | Greedy + 2-opt |
| Best `val/gap_pct` | 29.98% (ep 1/5) | **25.96%** (ep 10) |
| Final `val/gap_pct` | 33.66% | 35.51% |
| `train/loss_epoch` ep 50 | 0.053 | 0.053 |
| Wall time | ≈2.1 h | ≈2.1 h |
| Overcapacity | 0% | 0% |

| | TSP SL (Exp 3) | CVRP SL (this) |
|---|---|---|
| Best val gap | 4.46% | **25.96%** |

### Next steps

1. **Early-stop at epoch 10** checkpoint for downstream eval and CADO — do not use the epoch-50 weights.
2. Re-evaluate the epoch-10 checkpoint with a larger `eval_subset` for a stable gap estimate.
3. Confirm full training-set size on VESSL (≈337 batches/epoch vs expected ≈1,800).
4. Inspect heatmaps / decode on single instances (greedy vs greedy+2-opt).
5. Defer CADO CVRP until SL gap is consistently below ~15% on a fixed eval protocol.

### Reproducing the exports

```bash
uv run python wandb/main.py
```

`RUN_PATH` in `wandb/main.py` must point at `myqt6q9i` (current default).

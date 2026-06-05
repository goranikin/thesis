# Experiment Log (from `exports/`)

## Experiment 9: MDVRP-50 Supervised Assignment Training (Remote)

**Date**: 2026-06-04  
**wandb run**: [`difusco-mdvrp / ooi28boi`](https://wandb.ai/goranikin-my-project/difusco-mdvrp/runs/ooi28boi)  
**Goal**: Train the supervised DIFUSCO-style MDVRP assignment model on the generated MDVRP dataset and evaluate whether denoising-edge training improves depot-assignment accuracy.

**Primary data sources**:

| File | Rows | Description |
|---|---:|---|
| `exports/run_history_full.csv` | 512 | Interleaved train, epoch, validation, and summary rows |
| `exports/run_history_train.csv` | 450 | Rows with `train/global_step` set |
| `exports/run_history_val.csv` | 0 | Empty because `wandb/main.py` still splits validation on `val/gap_pct` |

Validation metrics below are computed from `exports/run_history_full.csv`, where MDVRP logs `val/assignment_accuracy` rather than `val/gap_pct`.

### Export columns

| Column | Present on | Meaning |
|---|---|---|
| `train/global_step` | train rows | Optimizer step index |
| `train/loss_step` | train rows | Per-log-step supervised denoising loss |
| `train/loss_epoch` | epoch rows | Mean training loss for the epoch |
| `train/grad_norm` | train rows | Gradient norm after clipping |
| `train/lr` | epoch rows | Cosine LR after scheduler step |
| `epoch` | epoch / validation / summary rows | Training epoch |
| `val/assignment_accuracy` | validation rows | Fraction of customers assigned to the GT depot |
| `val/capacity_violation_rate` | validation rows | Fraction of capacity violations in decoded assignment |
| `val/best_accuracy` | validation rows | Running best assignment accuracy |
| `val/final_best_accuracy` | summary row | Final best assignment accuracy |
| `_runtime` | all rows | Wall seconds since run start |
| `_step` | all rows | W&B internal log index |

### Configuration

| Parameter | Value |
|---|---|
| Model | small (`hidden_dim=128`, `num_layers=6`) |
| Problem | MDVRP-50, 2-5 depots |
| Data | `/mount/data/mdvrp10-10x5-5_64000_pyvrp.txt` |
| Local data path | `data/mdvrp10-10x5-5_64000_pyvrp.txt` |
| Train/val split | 0.9 |
| Epochs | **50** |
| Batch size | 64 |
| Learning rate | 2e-4 (cosine to 0) |
| Weight decay | 1e-4 |
| Diffusion T | 1000 (`beta_start=1e-4`, `beta_end=0.02`) |
| Inference steps | 50 (cosine schedule) |
| Eval cadence | epochs 1, 5, 10, ..., 50 |
| Eval subset | 500 |
| Platform | VESSL A100x1 |

Remote launch command: `uv run python -m difusco.mdvrp.main.run_train training=remote data_path=/mount/data/mdvrp10-10x5-5_64000_pyvrp.txt checkpoint_dir=/mount/checkpoints`.

### Summary

| Metric | Value |
|---|---:|
| Wall-clock | **57 min 44 s** (`_runtime` max = 3,464 s) |
| Logged train steps | 0 -> 44,900 (450 log points) |
| Epoch rows | 50 |
| Validation rows | 11 |
| `train/loss_epoch` (epoch 1 -> 50) | 0.1819 -> 0.0839 |
| Min `train/loss_epoch` | **0.0800** (epoch 44) |
| `train/loss_step` mean | 0.0921 |
| `train/grad_norm` mean | 0.183 |
| **Best assignment accuracy** | **81.89%** (epoch 1) |
| Final assignment accuracy | 55.75% (epoch 50) |
| Worst assignment accuracy | 54.22% (epoch 40) |
| Mean assignment accuracy | 61.02% (std 7.58 pp) |
| Capacity violation rate | 0% on all validation evals |
| `val/final_best_accuracy` | 81.89% |

The best checkpoint is the first validation point, not the final model. Training loss falls by more than half, but assignment accuracy drops from **81.89%** to **55.75%**, indicating that the supervised denoising objective is not aligned with the current assignment decoder/metric.

### Per-Epoch Training

| Epoch | Loss | LR |
|---:|---:|---:|
| 1 | 0.1819 | 2.00e-04 |
| 2 | 0.1119 | 1.99e-04 |
| 3 | 0.1053 | 1.98e-04 |
| 4 | 0.1012 | 1.97e-04 |
| 5 | 0.0978 | 1.95e-04 |
| 10 | 0.0944 | 1.81e-04 |
| 15 | 0.0889 | 1.59e-04 |
| 20 | 0.0874 | 1.31e-04 |
| 25 | 0.0858 | 1.00e-04 |
| 30 | 0.0845 | 6.91e-05 |
| 35 | 0.0842 | 4.12e-05 |
| 40 | 0.0828 | 1.91e-05 |
| 44 | **0.0800** | 7.02e-06 |
| 45 | 0.0852 | 4.89e-06 |
| 50 | 0.0839 | 0.00e+00 |

### Validation Log

Source: `exports/run_history_full.csv` (11 validation rows).

| Epoch | Assignment Acc. | Capacity Violation | Best So Far | Notes |
|---:|---:|---:|---:|---|
| 1 | **81.89%** | 0.00% | **81.89%** | **best** |
| 5 | 68.58% | 0.00% | 81.89% | |
| 10 | 60.76% | 0.00% | 81.89% | |
| 15 | 59.30% | 0.00% | 81.89% | |
| 20 | 56.82% | 0.00% | 81.89% | |
| 25 | 60.26% | 0.00% | 81.89% | small rebound |
| 30 | 60.63% | 0.00% | 81.89% | |
| 35 | 57.62% | 0.00% | 81.89% | |
| 40 | 54.22% | 0.00% | 81.89% | worst |
| 45 | 55.38% | 0.00% | 81.89% | |
| 50 | 55.75% | 0.00% | 81.89% | final |

### Training Dynamics

- **Loss improves, accuracy regresses**: `train/loss_epoch` drops from 0.1819 to 0.0839, but validation assignment accuracy falls by 26.14 pp from epoch 1 to epoch 50.
- **Best checkpoint is immediate**: `val/best_accuracy` freezes at 81.89% after epoch 1. Later training never recovers beyond 68.58%.
- **Feasibility is stable**: `val/capacity_violation_rate` is 0% throughout, so the degradation is not caused by capacity violations.
- **Gradient scale normalizes quickly**: Mean grad norm drops from 1.11 over the first 20 train logs to 0.112 over the last 20, while the LR decays to 0.
- **Validation metric is assignment-only**: This run does not report route length or gap. The log measures depot-assignment quality, not complete MDVRP route quality.

### Analysis

The export shows a completed remote supervised MDVRP run with healthy training-loss convergence but poor validation behavior. The model starts with high assignment accuracy at epoch 1 and then steadily loses alignment with the GT depot assignments. This is the same kind of loss-metric decoupling seen in the early CVRP supervised runs, but here it is even more direct because the validation target is assignment accuracy rather than tour length.

The most likely interpretation is that the current supervised diffusion target, decoding rule, or validation protocol is rewarding a representation that does not preserve depot assignment quality after additional optimization. The absence of capacity violations is encouraging, but it also means feasibility is too easy to explain the metric movement; the core issue is assignment correctness.

### Next Steps

1. **Use the epoch-1 checkpoint** for inspection; do not use the final epoch-50 checkpoint.
2. Fix `wandb/main.py` splitting so MDVRP validation rows are exported into `run_history_val.csv` using `val/assignment_accuracy`.
3. Visualize decoded assignments on a few validation instances at epochs 1, 5, and 50 to see which depots drift.
4. Compare validation before and after any training from the random/pretrained initialization to determine why epoch 1 is already the maximum.
5. Add route-length or assignment-cost metrics; assignment accuracy alone may hide whether wrong depots are near-equivalent or catastrophic.

### Reproducing the exports

```bash
uv run python wandb/main.py
```

`RUN_PATH` in `wandb/main.py` must point at `ooi28boi` (current default).

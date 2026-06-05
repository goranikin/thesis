# Experiment Log (from `exports/`)

## Experiment 10: MDVRP-50 CADO REINFORCE Fine-Tuning (Remote)

**Date**: 2026-06-05  
**wandb run**: [`cado-mdvrp / yrd41ml7`](https://wandb.ai/goranikin-my-project/cado-mdvrp/runs/yrd41ml7)  
**Goal**: Fine-tune the supervised DIFUSCO-MDVRP assignment model with CADO / REINFORCE and check whether validation depot-assignment accuracy improves without introducing capacity violations.

**Primary data sources**:

| File | Rows | Description |
|---|---:|---|
| `exports/run_history_full.csv` | 550 | Interleaved train + validation history |
| `exports/run_history_train.csv` | 500 | Rows with `train/global_step` set |
| `exports/run_history_val.csv` | 0 | Empty because `wandb/main.py` still splits validation on `val/gap_pct` |

Validation metrics below are computed from `exports/run_history_full.csv`, where MDVRP logs `val/assignment_accuracy` rather than `val/gap_pct`.

### Export Columns

| Column | Present on | Meaning |
|---|---|---|
| `train/global_step` | train rows | Optimizer step index |
| `train/mean_reward` | train rows | Mean assignment reward |
| `train/mean_log_prob` | train rows | Mean sampled log probability |
| `train/loss` | train rows | REINFORCE loss |
| `train/grad_norm` | train rows | Gradient norm after clipping |
| `epoch` | validation rows | Training epoch |
| `val/assignment_accuracy` | validation rows | Fraction of customers assigned to the GT depot |
| `val/capacity_violation_rate` | validation rows | Fraction of capacity violations in decoded assignment |
| `_runtime` | all rows | Wall seconds since run start |
| `_step` | all rows | W&B internal log index |

### Configuration

| Parameter | Value |
|---|---|
| Problem | MDVRP-50, 2-5 depots |
| Model | small (`hidden_dim=128`, `num_layers=6`) |
| Data | `/mount/data/mdvrp10-10x5-5_64000_pyvrp.txt` |
| Pretrained checkpoint | latest `/mount/checkpoints/difusco_mdvrp_*/best_model.pt` |
| Algorithm | REINFORCE |
| Reward mode | SR |
| 2-opt in reward | false (not applicable to assignment heatmap) |
| Hybrid-FT LoRA rank | 2 |
| Hybrid-FT selective layers | 1 |
| Learning rate | 1.0e-4 |
| Weight decay | 0.0 |
| Gradient clip | 1.0 |
| `M_train` | 5 |
| `M_eval` | 25 |
| Schedule type | cosine |
| Epochs | **500** |
| Samples per epoch | 128 |
| Batch size | 32 |
| Total optimizer steps | **2000** (`train/global_step`: 0 -> 1996) |
| Eval cadence | every 10 epochs |
| Eval subset | 100 |
| Platform | VESSL A100x1 |

Remote launch command used `cado.mdvrp.main.run_train` with `model=small`, the MDVRP data path, a mounted checkpoint directory, and a discovered pretrained DIFUSCO-MDVRP checkpoint.

### Summary

| Metric | Value |
|---|---:|
| Wall-clock | **1 h 49 min 30 s** (`_runtime` max = 6,570 s) |
| Logged train rows | 500 |
| Validation rows | 50 |
| First validation accuracy | 78.02% (epoch 10) |
| **Best validation accuracy** | **80.86%** (epoch 390) |
| Final validation accuracy | 80.48% (epoch 500) |
| Worst validation accuracy | 73.26% (epoch 100) |
| Mean validation accuracy | 78.30% (std 1.58 pp) |
| First-half mean accuracy | 77.23% (epochs 10-250) |
| Second-half mean accuracy | 79.36% (epochs 260-500) |
| Capacity violation rate | 0% on all 50 evals |

The run recovers from an early dip and finishes close to the best checkpoint. Unlike the supervised MDVRP run, CADO does not show a late collapse; validation accuracy gradually improves in the second half and remains feasible throughout.

### Training Values

| Metric | Mean | Std | Min | Max | First 20 | Last 20 |
|---|---:|---:|---:|---:|---:|---:|
| `train/mean_reward` | -0.528 | 0.233 | -0.833 | -0.154 | -0.341 | -0.296 |
| `train/mean_log_prob` | -0.339 | 0.0078 | -0.364 | -0.315 | -0.339 | -0.344 |
| `train/loss` | -0.0033 | 0.0080 | -0.0317 | 0.0207 | -0.0045 | -0.0023 |
| `train/grad_norm` | 2.93e-4 | 2.25e-4 | 3.90e-6 | 1.44e-3 | 3.95e-4 | 3.53e-4 |

`train/mean_reward` is noisy and non-monotonic, but it ends slightly better than the early-run average. `mean_log_prob` remains near -0.34, similar to the other CADO runs, suggesting the policy probability scale changes little even when validation accuracy moves.

### Validation Log

Full curve: 50 validation points in `exports/run_history_full.csv`.

| Epoch | Assignment Acc. | Capacity Violation | Notes |
|---:|---:|---:|---|
| 10 | 78.02% | 0.00% | first eval |
| 20 | 76.80% | 0.00% | |
| 30 | 76.92% | 0.00% | |
| 40 | 76.42% | 0.00% | |
| 50 | 77.02% | 0.00% | |
| 100 | 73.26% | 0.00% | worst |
| 150 | 77.50% | 0.00% | recovery |
| 200 | 79.16% | 0.00% | |
| 250 | 79.12% | 0.00% | |
| 300 | 79.28% | 0.00% | |
| 350 | 79.84% | 0.00% | |
| 390 | **80.86%** | 0.00% | **best** |
| 400 | 79.34% | 0.00% | |
| 450 | 77.82% | 0.00% | late dip |
| 490 | 80.02% | 0.00% | rebound |
| 500 | 80.48% | 0.00% | final |

### Training Dynamics

- **Early degradation then recovery**: Accuracy falls from 78.02% at epoch 10 to 73.26% at epoch 100, then climbs back above 79% by epoch 200.
- **Second half is stronger**: Mean accuracy improves from 77.23% in epochs 10-250 to 79.36% in epochs 260-500.
- **Best and final are close**: The best checkpoint at epoch 390 is 80.86%; the final checkpoint is 80.48%, only 0.38 pp lower.
- **Feasibility is preserved**: `val/capacity_violation_rate` is 0% for every validation pass.
- **Validation noise remains visible**: The curve oscillates by roughly 1-2 pp after recovery, likely amplified by `eval_subset=100`.

### Analysis

CADO-MDVRP fine-tuning is much more stable than the supervised MDVRP training curve, where the best accuracy occurred immediately and the final checkpoint regressed badly. This run starts below the supervised run's reported best, but it improves over its own early evaluations and finishes near the best observed CADO checkpoint.

The result should be interpreted cautiously against Experiment 9 because the evaluation subset differs: supervised MDVRP used `eval_subset=500`, while this CADO config uses `eval_subset=100`. Still, the CADO curve is useful operationally: it shows that assignment-based REINFORCE can maintain feasibility and recover accuracy after an initial dip, with best/final accuracy around 80-81%.

### Comparison

| Metric | Exp 9: DIFUSCO-MDVRP SL | Exp 10: CADO-MDVRP |
|---|---:|---:|
| Best assignment accuracy | 81.89% (epoch 1) | **80.86% (epoch 390)** |
| Final assignment accuracy | 55.75% (epoch 50) | **80.48% (epoch 500)** |
| Eval subset | 500 | 100 |
| Capacity violation | 0% | 0% |
| Wall-clock | 57 min 44 s | 1 h 49 min 30 s |

The cleanest takeaway is checkpoint stability: CADO preserves a high final assignment accuracy, while the supervised run's final checkpoint degraded substantially.

### Next Steps

1. Re-evaluate epochs 390 and 500 with `eval_subset=500` or larger so the CADO result is directly comparable to Experiment 9.
2. Fix `wandb/main.py` so MDVRP validation rows split on `val/assignment_accuracy` and populate `run_history_val.csv`.
3. Run a shorter CADO schedule starting from the same pretrained checkpoint to test whether 300-400 epochs are sufficient.
4. Add route-level MDVRP metrics beyond assignment accuracy, especially route length and depot-load utilization.
5. Inspect the epoch-100 dip and epoch-390 best checkpoint on fixed validation instances to identify what assignment patterns changed.

### Reproducing the Exports

```bash
uv run python wandb/main.py
```

`RUN_PATH` in `wandb/main.py` must point at `yrd41ml7` (current default).

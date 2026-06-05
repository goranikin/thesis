# Paper Experimental Results

This file is the paper-facing organization of the logged experiments. It intentionally keeps
only the experiments selected for the paper draft:

- **TSP-50**: Exp. 3 (`DIFUSCO SL h128/L6`) and Exp. 5 (`CADO REINFORCE extended`).
- **CVRP-50**: Exp. 6, Exp. 7, and Exp. 8.
- **MDVRP**: excluded for now because the experiments will be fixed/re-run.

All values below come from the experiment logs in `logs/`. These are validation-subset
results, not multi-seed test-set results.

## Figure Inventory

Matplotlib figures for the selected paper experiments were generated with
`logs/generate_paper_result_figures.py`.

| Figure | Files | Purpose |
|---|---|---|
| TSP-50 CADO curve | `logs/figures/paper_tsp50_gap.png`, `logs/figures/paper_tsp50_gap.pdf` | Compares Exp. 5 CADO validation curve against the Exp. 3 SL baseline. |
| CVRP-50 validation curves | `logs/figures/paper_cvrp50_gap_curves.png`, `logs/figures/paper_cvrp50_gap_curves.pdf` | Shows Exp. 6 greedy, Exp. 7 2-opt, and Exp. 8 CADO. |
| Best vs. final summary | `logs/figures/paper_best_final_gap_summary.png`, `logs/figures/paper_best_final_gap_summary.pdf` | Highlights checkpoint-selection effects across selected experiments. |

## Selected Results Summary

| Problem | Experiment | Method | Best metric | Final metric | Paper use |
|---|---|---|---:|---:|---|
| TSP-50 | Exp. 3 | DIFUSCO SL, h128/L6 | 4.46% gap | 4.46% gap | SL baseline for CADO. |
| TSP-50 | Exp. 5 | CADO REINFORCE extended | 3.955% gap | 4.031% gap | Main TSP CADO result. |
| CVRP-50 | Exp. 6 | DIFUSCO SL, greedy decode | 29.98% gap | 33.66% gap | Decoder baseline. |
| CVRP-50 | Exp. 7 | DIFUSCO SL, greedy + 2-opt | 25.96% gap | 35.51% gap | Improved SL/decode baseline. |
| CVRP-50 | Exp. 8 | CADO REINFORCE | 20.24% gap | 47.16% gap | Main CVRP CADO best-checkpoint result. |

## TSP-50

### Kept Experiments

| Experiment | Source | Model | Eval subset | Best gap | Final gap | Notes |
|---|---|---|---:|---:|---:|---|
| Exp. 3 | `difusco_training_2026_05_22.md` | h128/L6 | 500 | 4.46% | 4.46% | Per-epoch validation curve missing due to W&B logging issue. |
| Exp. 5 | `cado_tsp_training_2026_05_26.md` | h128/L6 + Hybrid-FT | 200 | 3.955% at epoch 970 | 4.031% | Extended CADO run from Exp. 3 checkpoint. |

### TSP-50 Interpretation for Paper

- Exp. 3 provides the supervised baseline: **4.46% validation gap**.
- Exp. 5 improves the best observed validation gap to **3.955%**, an absolute reduction of
  **0.505 percentage points** and a relative gap reduction of **11.3%**.
- The final Exp. 5 checkpoint is slightly worse than the best checkpoint but remains near the
  best value (**4.031%** final vs. **3.955%** best).
- The result should be described as a modest but consistent improvement from cost-aware
  fine-tuning, not as a state-of-the-art TSP result.

### TSP-50 Figure Data

| Epoch | Exp. 5 CADO gap |
|---:|---:|
| 50 | 4.771% |
| 100 | 4.627% |
| 150 | 4.364% |
| 200 | 4.789% |
| 250 | 4.503% |
| 300 | 4.943% |
| 350 | 4.649% |
| 400 | 4.988% |
| 450 | 4.561% |
| 500 | 4.328% |
| 550 | 4.737% |
| 600 | 4.339% |
| 650 | 4.599% |
| 700 | 4.362% |
| 750 | 4.505% |
| 800 | 4.545% |
| 850 | 4.948% |
| 900 | 4.489% |
| 950 | 4.813% |
| 970 | **3.955%** |
| 990 | 3.959% |
| 1000 | 4.031% |

Exp. 3 SL baseline line: **4.46%**.

## CVRP-50

### Kept Experiments

| Experiment | Source | Method | Decode | Best gap | Final gap | Notes |
|---|---|---|---|---:|---:|---|
| Exp. 6 | `difusco_cvrp_training_2026_05_28.md` | DIFUSCO SL | greedy | 29.98% at epochs 1/5 | 33.66% | Baseline without 2-opt. |
| Exp. 7 | `difusco_cvrp_training_2026_05_29.md` | DIFUSCO SL | greedy + per-route 2-opt | 25.96% at epoch 10 | 35.51% | 2-opt improves best SL gap by about 4 pp. |
| Exp. 8 | `cado_cvrp_training_2026_06_03.md` | CADO REINFORCE | CADO eval decode | 20.24% at epoch 470 | 47.16% | Best checkpoint improves over SL, but final checkpoint collapses. |

### CVRP-50 Interpretation for Paper

- Exp. 6 shows that greedy heatmap decoding alone gives a weak but feasible baseline:
  **29.98% best gap** with **0% overcapacity**.
- Exp. 7 shows the value of local search: adding per-route 2-opt improves the best supervised
  gap to **25.96%**.
- Exp. 8 shows that CADO can improve the best CVRP checkpoint further to **20.24%**, an
  absolute improvement of **5.72 percentage points** over Exp. 7.
- Exp. 8 must be reported as a **best-checkpoint** result. The final epoch is **47.16%**, so
  long CADO training is unstable without checkpoint selection.
- Compare gaps rather than raw route lengths across CVRP runs, because validation ground-truth
  lengths differ slightly between logs.

### CVRP-50 Figure Data

| Epoch | Exp. 6 SL greedy | Exp. 7 SL + 2-opt |
|---:|---:|---:|
| 1 | 29.98% | 31.20% |
| 5 | 29.98% | 27.39% |
| 10 | 31.17% | **25.96%** |
| 15 | 36.68% | 27.36% |
| 20 | 32.74% | 30.19% |
| 25 | 37.59% | 30.82% |
| 30 | 39.72% | 33.92% |
| 35 | 33.41% | 34.75% |
| 40 | 33.04% | 34.12% |
| 45 | 32.97% | 35.09% |
| 50 | 33.66% | 35.51% |

| Epoch | Exp. 8 CADO gap |
|---:|---:|
| 50 | 25.73% |
| 100 | 25.41% |
| 150 | 24.59% |
| 200 | 25.01% |
| 250 | 25.97% |
| 300 | 25.02% |
| 350 | 24.66% |
| 400 | 21.85% |
| 450 | 21.83% |
| 470 | **20.24%** |
| 500 | 23.38% |
| 550 | 25.14% |
| 600 | 24.81% |
| 650 | 25.61% |
| 700 | 25.23% |
| 750 | 27.63% |
| 800 | 47.37% |
| 850 | 42.15% |
| 900 | 41.48% |
| 950 | 39.63% |
| 1000 | 47.16% |

## Cross-Experiment Paper Claims

These are the claims that should remain in the paper draft:

1. **TSP-50**: CADO improves the selected supervised baseline from **4.46%** to **3.955%** best
   validation gap.
2. **CVRP-50 decoder comparison**: adding 2-opt improves the best supervised CVRP result from
   **29.98%** to **25.96%**.
3. **CVRP-50 CADO**: CADO improves the best CVRP validation checkpoint from **25.96%** to
   **20.24%**, but training is unstable and final checkpoint selection is unsafe.
4. **Checkpoint selection** is central: best and final checkpoints diverge strongly for CVRP.

## Excluded or Pending Results

- **TSP Exp. 1**: excluded from the paper result set; keep only as early pipeline validation.
- **TSP Exp. 4**: excluded from the paper result set because Exp. 5 is the extended CADO run.
- **MDVRP Exp. 9**: excluded for now. The MDVRP experiments will be fixed/re-run before paper
  writing.
- **MDVRP CADO**: not reported; no completed experiment log exists yet.

## Figure Generation

Run:

```bash
uv run python logs/generate_paper_result_figures.py
```

Outputs:

- `logs/figures/paper_tsp50_gap.png`
- `logs/figures/paper_tsp50_gap.pdf`
- `logs/figures/paper_cvrp50_gap_curves.png`
- `logs/figures/paper_cvrp50_gap_curves.pdf`
- `logs/figures/paper_best_final_gap_summary.png`
- `logs/figures/paper_best_final_gap_summary.pdf`

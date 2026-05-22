# TSP dataset line format

One text file contains many instances. Each instance is a **single line**. Empty lines are skipped when loading.

## Layout

```
<x0> <y0> <x1> <y1> ... <xN-1> <yN-1> output <t1> <t2> ... <tN> <t1>
```

| Section | Description |
|---------|-------------|
| Coordinates | `2 × N` floats: `x y` pairs for all nodes, in order |
| `output` | Literal separator (see `OUTPUT_MARKER` in `types/constants.py`) |
| Tour | `N + 1` integers: visit order, **1-indexed**, closed (last index equals first) |

## Conventions

- **N** = number of cities (same for every line in a given dataset file).
- Coordinates are in **[0, 1]** (uniform random from the generator), **not** divided by `COORD_SCALE`.
- Tour lists every city exactly once in the first `N` positions; position `N + 1` repeats the start city to close the tour.
- Node indices in the tour are **1-based** (first city = `1`).

## Example

`N = 5`:

```
0.12 0.34 0.56 0.78 0.90 0.11 0.22 0.33 0.44 0.55 output 3 1 4 2 5 3
```

| Part | Meaning |
|------|---------|
| `0.12 0.34 ... 0.55` | Five `(x, y)` pairs |
| `output` | Separator |
| `3 1 4 2 5 3` | Tour 3 → 1 → 4 → 2 → 5 → back to 3 |

## Pydantic models

| Model | Role |
|-------|------|
| `TspInstance` | Coordinates (`nodes: list[Coordinate]`) |
| `TspTour` | Closed tour (`nodes: list[int]`) |
| `TspSample` | Instance + solution |

**Serialize:** `TspSample.to_line()`  
**Parse:** `TspSample.from_line(line, num_nodes=N)` — requires known `N` (see `difusco/dataset.py` `TSPDataset`).

## Generation

```bash
uv run python -m data_generation.generate_tsp_data
uv run python -m data_generation.generate_tsp_data min_nodes=50 max_nodes=50 num_samples=128000
```

Defaults: `types/constants.py` (`DEFAULT_TSP_GENERATION`), overridable via Hydra / `configs/data_generation/tsp.yaml`.

# MDVRP dataset line format

One text file contains many instances. Each instance is a **single line**.

## Layout

```
<n_depots> <n_customers> <capacity> <num_vehicles_per_depot>
depots <x0> <y0> ... <xD-1> <yD-1>
customers <x0> <y0> ... <xC-1> <yC-1>
demands <q0> <q1> ... <qC-1>
assignment <a0> <a1> ... <aC-1>
output <route1> | <route2> | ...
```

| Section | Description |
|---------|-------------|
| Header | Four integers: depot count, customer count, capacity `Q`, vehicles per depot |
| `depots` | Keyword + `2 × n_depots` normalized coordinates |
| `customers` | Keyword + `2 × n_customers` normalized coordinates |
| `demands` | Keyword + `n_customers` integers (customers only; no depot demand line) |
| `assignment` | Keyword + `n_customers` integers: which depot serves each customer (**1-indexed** depot id) |
| `output` | Keyword + routes (same ` \| ` separator as CVRP) |

## Global node indexing (routes)

Locations are numbered in **one global index space** (matching the PyVRP model):

| Index range | Role |
|-------------|------|
| `1 … n_depots` | Depots |
| `n_depots + 1 … n_depots + n_customers` | Customers (same order as `customers` coordinates) |

Each route in `output` lists **1-indexed global location ids** visited on that route (customers and possibly depots as returned by the solver). Routes are separated by ` | `.

## Conventions

- Coordinates in the file are **normalized to [0, 1]** (`COORD_SCALE = 1000` internally).
- `assignment[i]` = depot index (1 … `n_depots`) for customer `i` (generation-time cluster label; the solver may route differently).
- `num_vehicles_per_depot` is the fleet size **per depot** (homogeneous capacity across depots).

## Example

2 depots, 3 customers:

```
2 3 170 4 depots 0.696000 0.780000 0.719000 0.141000 customers 0.919000 0.264000 0.512000 0.135000 0.573000 0.714000 demands 44 72 6 assignment 1 2 1 output 3 2 | 4 5
```

| Part | Meaning |
|------|---------|
| `2 3 170 4` | 2 depots, 3 customers, capacity 170, 4 vehicles per depot |
| `depots ...` | Two depot `(x, y)` pairs |
| `customers ...` | Three customer `(x, y)` pairs |
| `demands 44 72 6` | Customer demands |
| `assignment 1 2 1` | Customers assigned to depots 1, 2, 1 |
| `output 3 2 \| 4 5` | Two routes over global node indices |

## Pydantic models

| Model | Role |
|-------|------|
| `MdvrpInstance` | Depots, customers, demands, assignment, capacities |
| `VrpRoutes` | `routes: list[list[int]]` |
| `MdvrpSample` | Instance + routes |

**Serialize:** `MdvrpSample.to_line()`  
**Parse:** not implemented yet (only `to_line`); add `from_line` on `MdvrpSample` when a loader is needed.

## Generation

```bash
uv run python -m data_generation.generate_mdvrp_data
uv run python -m data_generation.generate_mdvrp_data min_depots=2 max_depots=5 num_samples=1280
```

Defaults: `DEFAULT_MDVRP_GENERATION` in `types/constants.py`.

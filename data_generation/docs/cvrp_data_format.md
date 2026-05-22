# CVRP dataset line format

One text file contains many instances. Each instance is a **single line**.

## Layout

```
<x0> <y0> ... <xM> <yM> <d0> <d1> ... <dM> <capacity> <num_vehicles> output <route1> | <route2> | ...
```

| Section | Description |
|---------|-------------|
| Coordinates | `2 × M` floats: normalized `x y` pairs for all nodes |
| Demands | `M + 1` integers, one per node |
| `capacity` | Vehicle capacity `Q` |
| `num_vehicles` | Fleet size `M` (upper bound used by the solver) |
| `output` | Literal separator (`OUTPUT_MARKER`) |
| Routes | One or more routes, separated by ` \| ` (`ROUTE_SEPARATOR`) |

## Conventions

- **M + 1** = total nodes = **1 depot** (index 1) + **M customers** (indices 2 … M+1).
- **Node 1** is the depot; its demand is always **0**.
- Coordinates are stored **normalized to [0, 1]** in the file. Internal grid coordinates use `[0, 1000]`; scale factor `COORD_SCALE = 1000` in `types/constants.py`.
- Each route lists **customer** indices only (**1-indexed**), in visit order. The depot is implicit at the start and end of each route.
- Multiple routes are joined with ` | ` (space-pipe-space).

## Example

Depot + 3 customers (`M = 3`, 4 nodes total):

```
0.428000 0.601000 0.743000 0.664000 0.097000 0.493000 0.726000 0.407000 0 37 73 88 89 228 5 output 4 6 5 11 | 3 8 9 | 2 7 10
```

| Part | Meaning |
|------|---------|
| 8 floats | 4 nodes × (`x`, `y`), normalized |
| `0 37 73 88 89` | Demands (depot + 3 customers) |
| `228 5` | Capacity 228, 5 vehicles |
| `output` | Separator |
| `4 6 5 11 \| 3 8 9 \| 2 7 10` | Three routes |

## Pydantic models

| Model | Role |
|-------|------|
| `CvrpInstance` | `nodes`, `demands`, `vehicle_capacity`, `num_vehicles` |
| `VrpRoutes` | `routes: list[list[int]]` |
| `CvrpSample` | Instance + routes |

**Serialize:** `CvrpSample.to_line()`  
**Parse:** `CvrpSample.from_line(line)`

## Generation

```bash
uv run python -m data_generation.generate_cvrp_data
uv run python -m data_generation.generate_cvrp_data min_nodes=20 max_nodes=50 num_samples=1280
```

Instances are built with `generate_x_instance` (X-style benchmark distribution), then solved with PyVRP.  
Defaults: `DEFAULT_CVRP_GENERATION` in `types/constants.py`.

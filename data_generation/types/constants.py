from typing import Any

# --- Dataset line format ---
COORD_SCALE = 1000.0
OUTPUT_MARKER = "output"
ROUTE_SEPARATOR = " | "

# --- Shared batch settings ---
NUM_SAMPLES = 128_000
BATCH_SIZE = 128
SEED = 1234
FILENAME: str | None = None

# --- TSP ---
TSP_MIN_NODES = 50
TSP_MAX_NODES = 50
TSP_SOLVER = "concorde"
TSP_LKH_TRIALS = 1000

DEFAULT_TSP_GENERATION: dict[str, Any] = {
    "num_samples": NUM_SAMPLES,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "filename": FILENAME,
    "min_nodes": TSP_MIN_NODES,
    "max_nodes": TSP_MAX_NODES,
    "solver": TSP_SOLVER,
    "lkh_trials": TSP_LKH_TRIALS,
}

# --- CVRP ---
CVRP_MIN_NODES = 20
CVRP_MAX_NODES = 50
CVRP_BATCH_SIZE = 128
CVRP_SOLVER_RUNTIME = 5.0
CVRP_DEPOT_POSITIONING = "random"
CVRP_CUSTOMER_POSITIONING = "random"
CVRP_DEMAND_DISTRIBUTION = "CV"

DEFAULT_CVRP_GENERATION: dict[str, Any] = {
    "num_samples": NUM_SAMPLES,
    "batch_size": CVRP_BATCH_SIZE,
    "seed": SEED,
    "filename": FILENAME,
    "min_nodes": CVRP_MIN_NODES,
    "max_nodes": CVRP_MAX_NODES,
    "solver_runtime": CVRP_SOLVER_RUNTIME,
    "depot_positioning": CVRP_DEPOT_POSITIONING,
    "customer_positioning": CVRP_CUSTOMER_POSITIONING,
    "demand_distribution": CVRP_DEMAND_DISTRIBUTION,
}

# --- MDVRP ---
MDVRP_MIN_CUSTOMERS_PER_DEPOT = 10
MDVRP_MAX_CUSTOMERS_PER_DEPOT = 20
MDVRP_MIN_DEPOTS = 2
MDVRP_MAX_DEPOTS = 10
MDVRP_BATCH_SIZE = 64
MDVRP_SOLVER_RUNTIME = 5.0

DEFAULT_MDVRP_GENERATION: dict[str, Any] = {
    "num_samples": NUM_SAMPLES,
    "batch_size": MDVRP_BATCH_SIZE,
    "seed": SEED,
    "filename": FILENAME,
    "min_depots": MDVRP_MIN_DEPOTS,
    "max_depots": MDVRP_MAX_DEPOTS,
    "min_customers_per_depot": MDVRP_MIN_CUSTOMERS_PER_DEPOT,
    "max_customers_per_depot": MDVRP_MAX_CUSTOMERS_PER_DEPOT,
    "solver_runtime": MDVRP_SOLVER_RUNTIME,
}

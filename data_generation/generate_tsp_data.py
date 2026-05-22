import argparse
import os
import pprint as pp
import sys
import time
import warnings
from multiprocessing import Pool

import lkh
import numpy as np
import tqdm
import tsplib95
from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde

warnings.filterwarnings("ignore")

_solver_config = {}


def _init_worker(config):
    global _solver_config
    _solver_config = config
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


def solve_tsp(nodes_coord):
    solver_name = _solver_config["solver"]
    num_nodes = nodes_coord.shape[0]

    if solver_name == "concorde":
        scale = 1e6
        solver = TSPSolver.from_data(
            nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D"
        )
        solution = solver.solve(verbose=False)
        tour = solution.tour
    elif solver_name == "lkh":
        scale = 1e6
        lkh_path = "LKH-3.0.6/LKH"
        problem = tsplib95.models.StandardProblem(
            name="TSP",
            type="TSP",
            dimension=num_nodes,
            edge_weight_type="EUC_2D",
            node_coords={n + 1: nodes_coord[n] * scale for n in range(num_nodes)},
        )

        solution = lkh.solve(
            lkh_path, problem=problem, max_trials=_solver_config["lkh_trails"], runs=10
        )
        tour = [n - 1 for n in solution[0]]
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    return tour


# uv run python -m data.generate_tsp_data --min_nodes 50 --max_nodes 50 --num_samples 128000 --batch_size 128 --filename "tsp50-50_concorde.txt" --solver concorde --seed 42
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=50)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=128000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--solver", type=str, default="concorde")
    parser.add_argument("--lkh_trails", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, (
        "Number of samples must be divisible by batch size"
    )

    _solver_config["solver"] = opts.solver
    _solver_config["lkh_trails"] = opts.lkh_trails

    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde.txt"

    # Pretty print the run args
    pp.pprint(vars(opts))

    num_workers = min(opts.batch_size, os.cpu_count() or 4)
    with (
        open(opts.filename, "w") as f,
        Pool(num_workers, initializer=_init_worker, initargs=(_solver_config,)) as p,
    ):
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

            batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])

            tours = p.map(
                solve_tsp,
                [batch_nodes_coord[idx] for idx in range(opts.batch_size)],
            )

            for idx, tour in enumerate(tours):
                if (np.sort(tour) == np.arange(num_nodes)).all():
                    f.write(
                        " ".join(
                            str(x) + str(" ") + str(y)
                            for x, y in batch_nodes_coord[idx]
                        )
                    )
                    f.write(str(" ") + str("output") + str(" "))
                    f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                    f.write(str(" ") + str(tour[0] + 1) + str(" "))
                    f.write("\n")

        end_time = time.time() - start_time

        assert b_idx == opts.num_samples // opts.batch_size - 1

    print(
        f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}."
    )
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")

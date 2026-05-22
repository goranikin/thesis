import logging
import warnings

import numpy as np

from data_generation.cli import parse_config
from data_generation.runner import log_generation_summary, run_batch_generation
from data_generation.types import TspGenerationConfig, TspInstance, TspSample, TspTour
from data_generation.tsp_solver import _init_worker, solve_tsp

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# uv run python -m data_generation.generate_tsp_data \
#   --min-nodes 50 --max-nodes 50 --num-samples 128000 --batch-size 128 \
#   --filename tsp50-50_concorde.txt --solver concorde --seed 42


def _build_tsp_batch(config: TspGenerationConfig) -> list[np.ndarray]:
    num_nodes = config.sample_num_nodes()
    return [np.random.random((num_nodes, 2)) for _ in range(config.batch_size)]


def _process_instance(coords: np.ndarray) -> str | None:
    tour = solve_tsp(coords)
    sample = TspSample(
        instance=TspInstance.from_coords_array(coords),
        tour=TspTour.from_solver_tour(tour),
    )
    if not sample.tour.is_valid(sample.instance.num_nodes):
        return None
    return sample.to_line()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = parse_config(TspGenerationConfig)
    np.random.seed(config.seed)
    logger.info("Run options: %s", config.model_dump())

    stats = run_batch_generation(
        config,
        config.output_path,
        pool_initializer=_init_worker,
        pool_initargs=(config.model_dump(include={"solver", "lkh_trials"}),),
        build_tasks=lambda: _build_tsp_batch(config),
        process_task=_process_instance,
    )

    log_generation_summary(
        problem_label=f"TSP{config.min_nodes}-{config.max_nodes}",
        written=stats.written,
        requested=config.num_samples,
        elapsed_seconds=stats.elapsed_seconds,
    )


if __name__ == "__main__":
    main()

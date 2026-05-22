import logging

import numpy as np

from data_generation.cli import parse_config
from data_generation.generators.x_instance.x_intance_generator import (
    generate_x_instance,
)
from data_generation.runner import log_generation_summary, run_batch_generation
from data_generation.types import (
    COORD_SCALE,
    CvrpGenerationConfig,
    XInstanceGeneratorConfig,
)
from data_generation.vrp_common import _init_worker, build_cvrp_sample

logger = logging.getLogger(__name__)

# uv run python -m data_generation.generate_cvrp_data \
#   --min-nodes 20 --max-nodes 50 --num-samples 1280 --batch-size 16 \
#   --filename cvrp20-50_pyvrp.txt --seed 42


def _build_cvrp_batch(
    config: CvrpGenerationConfig,
) -> list[tuple[int, XInstanceGeneratorConfig]]:
    n_customers = config.sample_num_nodes()
    return [(n_customers, config)] * config.batch_size


def _process_instance(
    args: tuple[int, XInstanceGeneratorConfig],
) -> str | None:
    n_customers, generator_config = args
    x_data = generate_x_instance(
        n_customers,
        depot_positioning=generator_config.depot_positioning,
        customer_positioning=generator_config.customer_positioning,
        demand_distribution=generator_config.demand_distribution,
    )
    sample = build_cvrp_sample(x_data)
    if sample is None:
        return None
    return sample.to_line()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = parse_config(CvrpGenerationConfig)
    np.random.seed(config.seed)
    logger.info("Run options: %s", config.model_dump())

    stats = run_batch_generation(
        config,
        config.output_path,
        pool_initializer=_init_worker,
        pool_initargs=(config.model_dump(include={"solver_runtime"}),),
        build_tasks=lambda: _build_cvrp_batch(config),
        process_task=_process_instance,
    )

    log_generation_summary(
        problem_label=f"CVRP{config.min_nodes}-{config.max_nodes}",
        written=stats.written,
        requested=config.num_samples,
        elapsed_seconds=stats.elapsed_seconds,
        extra_lines=[
            f"Coordinates are normalized to [0, 1] (divided by {COORD_SCALE:.0f}).",
        ],
    )


if __name__ == "__main__":
    main()

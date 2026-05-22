import logging

import numpy as np

from data_generation.cli import parse_config
from data_generation.generators.cvrp_instance.cvrp_from_mdvrp import generate_mdvrp_instance
from data_generation.runner import log_generation_summary, run_batch_generation
from data_generation.types import COORD_SCALE, MdvrpGenerationConfig
from data_generation.vrp_common import _init_worker, build_mdvrp_sample

logger = logging.getLogger(__name__)

# uv run python -m data_generation.generate_mdvrp_data \
#   --min-customers-per-depot 10 --max-customers-per-depot 20 \
#   --min-depots 2 --max-depots 5 --num-samples 1280 --batch-size 8 \
#   --filename mdvrp10-20x2-5_pyvrp.txt --seed 42


def _build_mdvrp_batch(config: MdvrpGenerationConfig) -> list[MdvrpGenerationConfig]:
    return [config] * config.batch_size


def _process_instance(config: MdvrpGenerationConfig) -> str | None:
    instance = generate_mdvrp_instance(
        n_range=config.customer_range,
        min_depots=config.min_depots,
        max_depots=config.max_depots,
    )
    sample = build_mdvrp_sample(instance)
    if sample is None:
        return None
    return sample.to_line()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = parse_config(MdvrpGenerationConfig)
    np.random.seed(config.seed)
    logger.info("Run options: %s", config.model_dump())

    stats = run_batch_generation(
        config,
        config.output_path,
        pool_initializer=_init_worker,
        pool_initargs=(config.model_dump(include={"solver_runtime"}),),
        build_tasks=lambda: _build_mdvrp_batch(config),
        process_task=_process_instance,
    )

    log_generation_summary(
        problem_label="MDVRP",
        written=stats.written,
        requested=config.num_samples,
        elapsed_seconds=stats.elapsed_seconds,
        extra_lines=[
            f"Coordinates are normalized to [0, 1] (divided by {COORD_SCALE:.0f}).",
        ],
    )


if __name__ == "__main__":
    main()

"""
uv run python -m data_generation.generate_mdvrp_data
uv run python -m data_generation.generate_mdvrp_data min_customers_per_depot=10 max_customers_per_depot=20 min_depots=2 max_depots=5
"""

import logging

import hydra
import numpy as np
from omegaconf import DictConfig

from data_generation.generators.cvrp_instance.cvrp_from_mdvrp import (
    generate_mdvrp_instance,
)
from data_generation.hydra_config import CONFIG_DIR
from data_generation.runner import log_generation_summary, run_batch_generation
from data_generation.types import COORD_SCALE, MdvrpGenerationConfig
from data_generation.vrp_common import _init_worker, build_mdvrp_sample

logger = logging.getLogger(__name__)


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


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR),
    config_name="mdvrp",
)
def main(hydra_cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = MdvrpGenerationConfig.from_hydra(hydra_cfg)
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

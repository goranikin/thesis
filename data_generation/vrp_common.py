import math
import os
import sys

import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxRuntime

from data_generation.types import (
    Coordinate,
    CvrpInstance,
    CvrpSample,
    MdvrpInstance,
    MdvrpSample,
    VrpRoutes,
    VrpSolverConfig,
)

_solver_config: VrpSolverConfig | None = None


def _init_worker(config: dict) -> None:
    global _solver_config
    _solver_config = VrpSolverConfig.model_validate(config)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


def _add_complete_edges(model: Model) -> None:
    for frm in model.locations:
        for to in model.locations:
            if frm is to:
                continue
            distance = round(math.hypot(frm.x - to.x, frm.y - to.y))
            model.add_edge(frm, to, distance)


def _routes_to_location_indices(model: Model, result) -> list[list[int]]:
    routes: list[list[int]] = []
    for route in result.best.routes():
        visits = list(route.visits())
        if visits:
            routes.append(visits)
    return routes


def _to_one_indexed_routes(model: Model, result) -> VrpRoutes:
    return VrpRoutes(
        routes=[
            [visit + 1 for visit in route]
            for route in _routes_to_location_indices(model, result)
        ],
    )


def solve_cvrp(instance: CvrpInstance) -> VrpRoutes | None:
    """Solve a single-depot CVRP instance."""
    if _solver_config is None:
        raise RuntimeError("VRP solver config not initialized in worker process")

    model = Model()
    depot = model.add_depot(
        x=float(instance.nodes[0].x),
        y=float(instance.nodes[0].y),
    )

    for idx in range(1, instance.num_nodes):
        node = instance.nodes[idx]
        model.add_client(
            x=float(node.x),
            y=float(node.y),
            delivery=[int(instance.demands[idx])],
        )

    model.add_vehicle_type(
        num_available=int(instance.num_vehicles),
        capacity=[int(instance.vehicle_capacity)],
        start_depot=depot,
        end_depot=depot,
    )
    _add_complete_edges(model)

    result = model.solve(
        stop=MaxRuntime(_solver_config.solver_runtime),
        display=False,
    )
    if not result.is_feasible():
        return None

    return _to_one_indexed_routes(model, result)


def solve_mdvrp(instance: MdvrpInstance) -> VrpRoutes | None:
    """Solve a multi-depot VRP instance."""
    if _solver_config is None:
        raise RuntimeError("VRP solver config not initialized in worker process")

    model = Model()
    depots = [
        model.add_depot(x=float(depot.x), y=float(depot.y)) for depot in instance.depots
    ]

    for idx in range(instance.n_customers):
        customer = instance.customers[idx]
        model.add_client(
            x=float(customer.x),
            y=float(customer.y),
            delivery=[int(instance.customer_demands[idx])],
        )

    for depot in depots:
        model.add_vehicle_type(
            num_available=int(instance.num_vehicles_per_depot),
            capacity=[int(instance.vehicle_capacity)],
            start_depot=depot,
            end_depot=depot,
        )

    _add_complete_edges(model)

    result = model.solve(
        stop=MaxRuntime(_solver_config.solver_runtime),
        display=False,
    )
    if not result.is_feasible():
        return None

    return _to_one_indexed_routes(model, result)


def cvrp_instance_from_x_data(x_data: dict, num_vehicles: int) -> CvrpInstance:
    nodes = [
        Coordinate.from_pair((x, y))
        for x, y in zip(x_data["x_coordinates"], x_data["y_coordinates"], strict=True)
    ]
    return CvrpInstance(
        nodes=nodes,
        demands=[int(d) for d in x_data["demands"]],
        vehicle_capacity=int(x_data["vehicle_capacity"]),
        num_vehicles=num_vehicles,
    )


def estimate_cvrp_num_vehicles(demands: list[int], capacity: int) -> int:
    total_demand = float(sum(demands[1:]))
    return max(1, int(np.ceil((2 + 2 * np.random.random()) * total_demand / capacity)))


def build_cvrp_sample(x_data: dict) -> CvrpSample | None:
    instance = cvrp_instance_from_x_data(
        x_data,
        num_vehicles=estimate_cvrp_num_vehicles(
            [int(d) for d in x_data["demands"]],
            int(x_data["vehicle_capacity"]),
        ),
    )
    routes = solve_cvrp(instance)
    if routes is None:
        return None
    return CvrpSample(instance=instance, routes=routes)


def build_mdvrp_sample(instance: MdvrpInstance) -> MdvrpSample | None:
    routes = solve_mdvrp(instance)
    if routes is None:
        return None
    return MdvrpSample(instance=instance, routes=routes)

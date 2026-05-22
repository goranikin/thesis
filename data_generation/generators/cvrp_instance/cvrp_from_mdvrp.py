import random

import numpy as np

from data_generation.generators.x_instance.x_intance_generator import generate_x_instance
from data_generation.types import Coordinate, MdvrpInstance


def _estimate_num_vehicles(total_demand: float, capacity: float, n_depots: int) -> int:
    return int(np.ceil((2 + 2 * random.random()) * total_demand / (capacity * n_depots)))


def generate_mdvrp_instance(
    n_range: tuple[int, int],
    min_depots: int = 2,
    max_depots: int = 10,
) -> MdvrpInstance:
    """Generate a multi-depot VRP instance from an X-style customer distribution."""
    customer_per_depot = random.randint(n_range[0], n_range[1])
    n_depots = random.randint(min_depots, max_depots)
    n_customers = customer_per_depot * n_depots

    x_data = generate_x_instance(n_customers)
    x_list = list(x_data["x_coordinates"])
    y_list = list(x_data["y_coordinates"])
    demand_list = list(x_data["demands"])
    capacity = int(x_data["vehicle_capacity"])

    depots = [Coordinate.from_pair((x_list.pop(0), y_list.pop(0)))]
    demand_list.pop(0)

    for _ in range(1, n_depots):
        depots.append(
            Coordinate(x=random.randint(1, 1000), y=random.randint(1, 1000))
        )

    num_vehicles = _estimate_num_vehicles(sum(demand_list), capacity, n_depots)
    depot_assignment = np.random.randint(0, n_depots, size=n_customers).tolist()

    customers = [
        Coordinate.from_pair((x_list[i], y_list[i])) for i in range(n_customers)
    ]
    customer_demands = [int(demand_list[i]) for i in range(n_customers)]

    return MdvrpInstance(
        n_depots=n_depots,
        n_customers=n_customers,
        depots=depots,
        customers=customers,
        customer_demands=customer_demands,
        depot_assignment=depot_assignment,
        vehicle_capacity=capacity,
        num_vehicles_per_depot=num_vehicles,
    )


def generate_cvrp_instances(n_range: tuple[int, int]) -> list[dict]:
    """Split an MDVRP instance into per-depot CVRP sub-instances (legacy helper)."""
    instance = generate_mdvrp_instance(n_range)
    depot_clusters: list[list[int]] = [[] for _ in range(instance.n_depots)]
    for node in range(instance.n_customers):
        depot_clusters[instance.depot_assignment[node]].append(node)

    cvrp_lists: list[dict] = []
    for depot_idx in range(instance.n_depots):
        sub_instance = {
            "x_coordinates": [instance.depots[depot_idx].x],
            "y_coordinates": [instance.depots[depot_idx].y],
            "demands": [0],
            "service_times": np.zeros(len(depot_clusters[depot_idx]) + 1),
            "vehicle_capacity": instance.vehicle_capacity,
            "num_vehicles": instance.num_vehicles_per_depot,
            "depot": 0,
        }
        for customer_idx in depot_clusters[depot_idx]:
            customer = instance.customers[customer_idx]
            sub_instance["x_coordinates"].append(customer.x)
            sub_instance["y_coordinates"].append(customer.y)
            sub_instance["demands"].append(instance.customer_demands[customer_idx])
        cvrp_lists.append(sub_instance)

    return cvrp_lists

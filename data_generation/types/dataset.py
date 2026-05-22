import numpy as np
from pydantic import Field, field_validator, model_validator

from data_generation.types.base import Schema
from data_generation.types.constants import COORD_SCALE, OUTPUT_MARKER, ROUTE_SEPARATOR


class Coordinate(Schema):
    x: float
    y: float

    @classmethod
    def from_pair(cls, pair: tuple[float, float]):
        x, y = pair
        return cls(x=float(x), y=float(y))

    def normalized(self) -> tuple[float, float]:
        return self.x / COORD_SCALE, self.y / COORD_SCALE

    def to_normalized_str(self) -> str:
        nx, ny = self.normalized()
        return f"{nx:.6f} {ny:.6f}"


class VrpRoutes(Schema):
    """Vehicle routes as 1-indexed location visits (depot excluded)."""

    routes: list[list[int]] = Field(default_factory=list)

    def to_output_string(self) -> str:
        return ROUTE_SEPARATOR.join(
            " ".join(str(node) for node in route) for route in self.routes
        )

    @classmethod
    def from_output_string(cls, text: str) -> "VrpRoutes":
        if not text.strip():
            return cls(routes=[])
        routes = [
            [int(node) for node in route.split()]
            for route in text.split(ROUTE_SEPARATOR)
        ]
        return cls(routes=routes)


class TspInstance(Schema):
    nodes: list[Coordinate]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @classmethod
    def from_coords_array(cls, coords: np.ndarray) -> "TspInstance":
        return cls(
            nodes=[Coordinate(x=float(x), y=float(y)) for x, y in coords],
        )

    def to_coords_array(self) -> np.ndarray:
        return np.array([[node.x, node.y] for node in self.nodes], dtype=np.float64)


class TspTour(Schema):
    """Closed TSP tour with 1-indexed nodes; last node repeats the start."""

    nodes: list[int]

    @classmethod
    def from_solver_tour(cls, tour: list[int] | np.ndarray) -> "TspTour":
        one_indexed = [int(node) + 1 for node in tour]
        return cls(nodes=[*one_indexed, one_indexed[0]])

    def is_valid(self, num_nodes: int) -> bool:
        visit_order = np.array(self.nodes[:-1]) - 1
        return bool((np.sort(visit_order) == np.arange(num_nodes)).all())


class TspSample(Schema):
    instance: TspInstance
    tour: TspTour

    def to_line(self) -> str:
        coords = " ".join(f"{node.x} {node.y}" for node in self.instance.nodes)
        tour = " ".join(str(node) for node in self.tour.nodes)
        return f"{coords} {OUTPUT_MARKER} {tour}\n"

    @classmethod
    def from_line(cls, line: str, *, num_nodes: int) -> "TspSample":
        coords_part, tour_part = line.strip().split(f" {OUTPUT_MARKER} ")
        values = [float(value) for value in coords_part.split()]
        nodes = [
            Coordinate(x=values[i], y=values[i + 1]) for i in range(0, len(values), 2)
        ]
        tour_nodes = [int(value) for value in tour_part.split()]
        sample = cls(
            instance=TspInstance(nodes=nodes),
            tour=TspTour(nodes=tour_nodes),
        )
        if sample.instance.num_nodes != num_nodes:
            msg = f"Expected {num_nodes} nodes, got {sample.instance.num_nodes}"
            raise ValueError(msg)
        return sample


class CvrpInstance(Schema):
    nodes: list[Coordinate]
    demands: list[int]
    vehicle_capacity: int
    num_vehicles: int

    @field_validator("demands")
    @classmethod
    def demands_match_nodes(cls, demands: list[int], info) -> list[int]:
        nodes = info.data.get("nodes")
        if nodes is not None and len(demands) != len(nodes):
            raise ValueError("demands must have one entry per node")
        if demands and demands[0] != 0:
            raise ValueError("depot demand must be 0")
        return demands

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


class CvrpSample(Schema):
    instance: CvrpInstance
    routes: VrpRoutes

    def to_line(self) -> str:
        coords = " ".join(node.to_normalized_str() for node in self.instance.nodes)
        demands = " ".join(str(demand) for demand in self.instance.demands)
        meta = f"{self.instance.vehicle_capacity} {self.instance.num_vehicles}"
        return (
            f"{coords} {demands} {meta} {OUTPUT_MARKER} "
            f"{self.routes.to_output_string()}\n"
        )

    @classmethod
    def from_line(cls, line: str) -> "CvrpSample":
        before_output, routes_text = line.strip().split(f" {OUTPUT_MARKER} ")
        tokens = before_output.split()
        capacity = int(tokens[-2])
        num_vehicles = int(tokens[-1])
        num_nodes = (len(tokens) - 2) // 3
        coord_values = [float(value) for value in tokens[: 2 * num_nodes]]
        demand_values = tokens[2 * num_nodes : 2 * num_nodes + num_nodes]
        nodes = [
            Coordinate(
                x=coord_values[i] * COORD_SCALE,
                y=coord_values[i + 1] * COORD_SCALE,
            )
            for i in range(0, len(coord_values), 2)
        ]
        return cls(
            instance=CvrpInstance(
                nodes=nodes,
                demands=[int(value) for value in demand_values],
                vehicle_capacity=capacity,
                num_vehicles=num_vehicles,
            ),
            routes=VrpRoutes.from_output_string(routes_text),
        )


class MdvrpInstance(Schema):
    n_depots: int
    n_customers: int
    depots: list[Coordinate]
    customers: list[Coordinate]
    customer_demands: list[int]
    depot_assignment: list[int]
    vehicle_capacity: int
    num_vehicles_per_depot: int

    @model_validator(mode="after")
    def counts_are_consistent(self) -> "MdvrpInstance":
        if len(self.depots) != self.n_depots:
            raise ValueError("n_depots does not match depots length")
        if len(self.customers) != self.n_customers:
            raise ValueError("n_customers does not match customers length")
        if len(self.customer_demands) != self.n_customers:
            raise ValueError("customer_demands length must match n_customers")
        if len(self.depot_assignment) != self.n_customers:
            raise ValueError("depot_assignment length must match n_customers")
        return self


class MdvrpSample(Schema):
    instance: MdvrpInstance
    routes: VrpRoutes

    def to_line(self) -> str:
        depot_coords = " ".join(
            depot.to_normalized_str() for depot in self.instance.depots
        )
        customer_coords = " ".join(
            customer.to_normalized_str() for customer in self.instance.customers
        )
        demands = " ".join(str(demand) for demand in self.instance.customer_demands)
        assignment = " ".join(
            str(depot_idx + 1) for depot_idx in self.instance.depot_assignment
        )
        meta = (
            f"{self.instance.n_depots} {self.instance.n_customers} "
            f"{self.instance.vehicle_capacity} "
            f"{self.instance.num_vehicles_per_depot}"
        )
        return (
            f"{meta} depots {depot_coords} customers {customer_coords} "
            f"demands {demands} assignment {assignment} "
            f"{OUTPUT_MARKER} {self.routes.to_output_string()}\n"
        )

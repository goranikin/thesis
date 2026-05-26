"""
CVRP dataset for DIFUSCO.

Each instance has:
    - 1 depot at node 0
    - N customers at nodes 1..N
    - Vehicle capacity Q
    - Demands d_0, d_1, ..., d_N (d_0 = 0)

Per-node features (4D):
    [x, y, demand_normalized, is_depot]

The solution is a multi-set of routes, each "depot -> c1 -> c2 -> ... -> cL -> depot".
The edge label is 1 iff that directed/undirected edge appears in any route.
Total positive edges in a solution = N + K, where K is the number of routes
(every customer has degree 2, depot has degree 2K).

Graph topology:
    - Dense (sparse_factor <= 0): fully connected, both directions.
    - Sparse (sparse_factor > 0): each customer connects to its k nearest
      customers AND to the depot. The depot connects to every customer.
      This guarantees that every depot<->customer edge needed for any
      valid CVRP solution exists in the graph.
"""

import logging

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset

from data_generation.types.dataset import CvrpSample

logger = logging.getLogger(__name__)


class CVRPDataset(Dataset):
    """
    Args:
        file_path:      path to a CVRP data file (one instance per line).
        num_customers:  number of customers per instance (excludes the depot).
                        Total nodes per graph is num_customers + 1.
        sparse_factor:  if > 0, build a (customer k-NN + depot-fully-connected)
                        graph with k = sparse_factor. If <= 0, build a dense graph.
    """

    def __init__(
        self,
        file_path: str,
        num_customers: int,
        sparse_factor: int = -1,
    ):
        self.num_customers = num_customers
        self.num_nodes = num_customers + 1  # +1 for the depot
        self.sparse_factor = sparse_factor

        self.file_lines = open(file_path).read().splitlines()
        self.file_lines = [line for line in self.file_lines if line.strip()]
        logger.info(
            f"Loaded {len(self.file_lines)} CVRP-{num_customers} instances "
            f"({self.num_nodes} nodes/graph)"
        )

        # Pre-compute the dense edge_index once (same topology for every instance).
        if sparse_factor <= 0:
            N = self.num_nodes
            src, dst = [], []
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    src.append(i)
                    dst.append(j)
            self._edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            self._edge_index = None

    def __len__(self):
        return len(self.file_lines)

    def _parse_line(self, line: str):
        """
        Returns:
            coords:    (N+1, 2) float, normalized to [0, 1]
            demands:   (N+1,)   int, demands[0] = 0 (depot)
            capacity:  int
            routes:    list[list[int]] of 0-indexed customer indices
                       (e.g., routes[0] = [3, 7, 12] means depot->c3->c7->c12->depot)
        """
        sample = CvrpSample.from_line(line)
        instance = sample.instance

        # CvrpSample.from_line stores coords in internal [0, 1000] units; normalize.
        coords = np.array(
            [(node.x, node.y) for node in instance.nodes], dtype=np.float64
        )
        coords = coords / 1000.0

        demands = np.array(instance.demands, dtype=np.int64)
        capacity = int(instance.vehicle_capacity)

        # Routes use 1-indexed customer ids (depot is 1, customers are 2..N+1).
        # Convert to 0-indexed node ids (depot is 0, customers are 1..N).
        routes = [
            [int(c) - 1 for c in route] for route in sample.routes.routes if route
        ]

        if coords.shape[0] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, got {coords.shape[0]}. "
                f"Set num_customers={coords.shape[0] - 1} if this is intentional."
            )

        return coords, demands, capacity, routes

    def __getitem__(self, index: int):
        coords, demands, capacity, routes = self._parse_line(self.file_lines[index])
        if self.sparse_factor > 0:
            graph = self._build_sparse_graph(coords, demands, capacity, routes)
        else:
            graph = self._build_dense_graph(coords, demands, capacity, routes)
        node_feat, edge_index, edge_dist, edge_label = graph
        # Carry per-instance capacity through to the trainer. Validation runs
        # with batch_size=1, so this is just a 1-element tensor in that case.
        capacity_t = torch.tensor([capacity], dtype=torch.long)
        return node_feat, edge_index, edge_dist, edge_label, capacity_t

    # ---------------------------------------------------------------- features
    def _node_features(
        self, coords: np.ndarray, demands: np.ndarray, capacity: int
    ) -> torch.Tensor:
        """
        (N+1, 4):
            col 0: x          in [0, 1]
            col 1: y          in [0, 1]
            col 2: demand / Q in [0, 1]; depot is always 0
            col 3: is_depot   {0, 1}; row 0 is 1, rest are 0
        """
        N1 = coords.shape[0]
        demand_norm = demands.astype(np.float32) / max(capacity, 1)
        is_depot = np.zeros(N1, dtype=np.float32)
        is_depot[0] = 1.0

        node_feat = np.stack(
            [
                coords[:, 0].astype(np.float32),
                coords[:, 1].astype(np.float32),
                demand_norm,
                is_depot,
            ],
            axis=1,
        )
        return torch.from_numpy(node_feat).float()

    # ---------------------------------------------------------------- labels
    @staticmethod
    def _route_edge_set(routes: list[list[int]]) -> set[tuple[int, int]]:
        """
        Collect undirected edges {min(u,v), max(u,v)} for every consecutive pair
        in every route, padding each route with the depot (node 0) at the front
        and back.
        """
        edges: set[tuple[int, int]] = set()
        for route in routes:
            path = [0, *route, 0]
            for u, v in zip(path[:-1], path[1:]):
                edges.add((min(u, v), max(u, v)))
        return edges

    # ---------------------------------------------------------------- dense
    def _build_dense_graph(
        self,
        coords: np.ndarray,
        demands: np.ndarray,
        capacity: int,
        routes: list[list[int]],
    ):
        assert self._edge_index is not None
        edge_index = self._edge_index

        node_feat = self._node_features(coords, demands, capacity)

        src_coords = coords[edge_index[0].numpy()]
        dst_coords = coords[edge_index[1].numpy()]
        distances = np.sqrt(((src_coords - dst_coords) ** 2).sum(axis=1))
        edge_dist = torch.from_numpy(distances).float()

        tour_edges = self._route_edge_set(routes)
        labels = np.zeros(edge_index.shape[1], dtype=np.float32)
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        for i in range(edge_index.shape[1]):
            u, v = int(src[i]), int(dst[i])
            if (min(u, v), max(u, v)) in tour_edges:
                labels[i] = 1.0
        edge_label = torch.from_numpy(labels)

        return node_feat, edge_index, edge_dist, edge_label

    # ---------------------------------------------------------------- sparse
    def _build_sparse_graph(
        self,
        coords: np.ndarray,
        demands: np.ndarray,
        capacity: int,
        routes: list[list[int]],
    ):
        """
        Customer k-NN (over customers only) PLUS the depot connected to every
        customer in both directions. Self-loops are excluded. Every customer is
        guaranteed to have the depot in its neighborhood.
        """
        N = self.num_customers  # customer count
        k = self.sparse_factor

        # ---------- customer-only KNN ----------
        customer_coords = coords[1:]  # (N, 2)
        kdt = KDTree(customer_coords, leaf_size=30, metric="euclidean")
        # Query k+1 because the first neighbour is the node itself; drop it.
        dists_knn, idx_knn = kdt.query(
            customer_coords, k=min(k + 1, N), return_distance=True
        )
        dists_knn = dists_knn[:, 1:]  # (N, k)
        idx_knn = idx_knn[:, 1:] + 1  # +1 to map back to 0-indexed (depot=0, cust=1..N)

        cust_src_local = np.arange(1, N + 1).reshape(-1, 1).repeat(idx_knn.shape[1], axis=1).reshape(-1)
        cust_dst_local = idx_knn.reshape(-1)
        cust_dist_local = dists_knn.reshape(-1)

        # ---------- depot <-> every customer ----------
        depot_to_cust_src = np.zeros(N, dtype=np.int64)
        depot_to_cust_dst = np.arange(1, N + 1, dtype=np.int64)
        cust_to_depot_src = depot_to_cust_dst
        cust_to_depot_dst = depot_to_cust_src
        depot_dists = np.sqrt(((coords[1:] - coords[0]) ** 2).sum(axis=1))

        src = np.concatenate(
            [cust_src_local, depot_to_cust_src, cust_to_depot_src]
        )
        dst = np.concatenate(
            [cust_dst_local, depot_to_cust_dst, cust_to_depot_dst]
        )
        dists = np.concatenate(
            [cust_dist_local, depot_dists, depot_dists]
        )

        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        edge_dist = torch.from_numpy(dists).float()

        node_feat = self._node_features(coords, demands, capacity)

        tour_edges = self._route_edge_set(routes)
        labels = np.zeros(edge_index.shape[1], dtype=np.float32)
        src_np = edge_index[0].numpy()
        dst_np = edge_index[1].numpy()
        for i in range(edge_index.shape[1]):
            u, v = int(src_np[i]), int(dst_np[i])
            if (min(u, v), max(u, v)) in tour_edges:
                labels[i] = 1.0
        edge_label = torch.from_numpy(labels)

        return node_feat, edge_index, edge_dist, edge_label


def collate_cvrp(batch):
    """
    Collate B individual CVRP graphs into a single super-graph.

    Identical to ``collate_tsp`` — each graph's node indices are offset by the
    running node count so the GNN sees B disconnected components.

    Input: list of (node_feat, edge_index, edge_dist, edge_label, capacity)
    Output: (node_feat, edge_index, edge_dist, edge_label, capacities) for the super-graph
        - capacities is a 1-D LongTensor of length B, one entry per source graph.
    """
    all_node_feat = []
    all_edge_index = []
    all_edge_dist = []
    all_edge_label = []
    all_capacities = []

    node_offset = 0
    for node_feat, edge_index, edge_dist, edge_label, capacity in batch:
        all_node_feat.append(node_feat)
        all_edge_index.append(edge_index + node_offset)
        all_edge_dist.append(edge_dist)
        all_edge_label.append(edge_label)
        all_capacities.append(capacity)
        node_offset += node_feat.shape[0]

    return (
        torch.cat(all_node_feat, dim=0),
        torch.cat(all_edge_index, dim=1),
        torch.cat(all_edge_dist, dim=0),
        torch.cat(all_edge_label, dim=0),
        torch.cat(all_capacities, dim=0),
    )


def test_cvrp_dataset():
    from torch.utils.data import DataLoader

    dataset = CVRPDataset(
        file_path="data/cvrp50-50_128000_pyvrp.txt",
        num_customers=50,
        sparse_factor=-1,
    )
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_cvrp
    )
    for batch in dataloader:
        node_feat, edge_index, edge_dist, edge_label, capacities = batch
        logger.debug(
            f"node_feat={node_feat.shape} edge_index={edge_index.shape} "
            f"edge_dist={edge_dist.shape} edge_label={edge_label.shape} "
            f"capacities={capacities.tolist()} "
            f"positive_edges={int(edge_label.sum().item())}"
        )
        break


# uv run python -m difusco.cvrp.dataset
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_cvrp_dataset()

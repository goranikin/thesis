"""
MDVRP dataset for DIFUSCO.

Each instance has:
    - K depots (K is per-instance)
    - N customers
    - Vehicle capacity Q and num_vehicles_per_depot V
    - Demands d_1, ..., d_N (depots have demand 0)

Node layout (for the bipartite assignment graph):
    indices [0, K)         -> depots
    indices [K, K + N)     -> customers
    total nodes  N_total = K + N

Node features (4D):
    col 0: x              in [0, 1]
    col 1: y              in [0, 1]
    col 2: demand / Q     in [0, 1] (depots are 0)
    col 3: role           {0 = depot, 1 = customer}

Edges:
    Customer-depot bipartite, both directions: E = 2 * K * N.
    Optionally, customer-customer k-NN edges may be added (configured via
    DataConfig.customer_knn). These are purely structural — their labels are
    always 0 and they are excluded from the heatmap loss.

Edge labels (binary):
    label = 1 iff customer c is assigned to depot d in the ground-truth
    solution (i.e., one of the routes from depot d visits customer c).

Label extraction:
    The serialized MDVRP file does NOT record which depot each route came
    from. We infer it with the *nearest-depot-to-route-centroid* heuristic:
    for each route, take the centroid of its customers and assign every
    customer in that route to the depot closest to that centroid. This is
    approximate but very reliable for X-style instances where PyVRP picks
    each route's depot to minimize total tour length.

    To get exact labels, extend MdvrpSample to record depot indices per
    route (see notes in vrp_common.py).
"""

import logging

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset

from data_generation.types.constants import COORD_SCALE, OUTPUT_MARKER, ROUTE_SEPARATOR

logger = logging.getLogger(__name__)


def _parse_mdvrp_line(line: str) -> dict:
    """
    Parse one MDVRP line as written by ``MdvrpSample.to_line()``.

    Returns a dict with:
        n_depots, n_customers, capacity, num_vehicles_per_depot,
        depots:    (K, 2) normalized to [0, 1]
        customers: (N, 2) normalized to [0, 1]
        demands:   (N,) ints
        random_assignment: (N,) ints, the RANDOM initial assignment
                           (PyVRP overwrites this; do NOT use as label)
        routes:    list[list[int]] of 0-indexed customer ids
    """
    text = line.strip()
    before_output, routes_text = text.split(f" {OUTPUT_MARKER} ", maxsplit=1)

    # before_output:
    #   "<n_depots> <n_customers> <capacity> <num_veh_per_depot> "
    #   "depots <2K floats> customers <2N floats> demands <N ints> "
    #   "assignment <N ints>"
    tokens = before_output.split()
    n_depots = int(tokens[0])
    n_customers = int(tokens[1])
    capacity = int(tokens[2])
    num_vehicles_per_depot = int(tokens[3])

    # Find section markers by index — they appear in fixed order.
    rest = tokens[4:]
    sections = {"depots": None, "customers": None, "demands": None, "assignment": None}
    section_starts = []
    for i, tok in enumerate(rest):
        if tok in sections:
            section_starts.append((tok, i))

    # Build (name, start, end) for each section.
    spans: dict[str, list[str]] = {}
    for j, (name, idx) in enumerate(section_starts):
        start = idx + 1
        end = section_starts[j + 1][1] if j + 1 < len(section_starts) else len(rest)
        spans[name] = rest[start:end]

    depot_vals = [float(v) for v in spans["depots"]]
    customer_vals = [float(v) for v in spans["customers"]]
    demand_vals = [int(v) for v in spans["demands"]]
    assign_vals = [int(v) for v in spans["assignment"]]

    if len(depot_vals) != 2 * n_depots:
        raise ValueError(f"depots: expected {2*n_depots} floats, got {len(depot_vals)}")
    if len(customer_vals) != 2 * n_customers:
        raise ValueError(
            f"customers: expected {2*n_customers} floats, got {len(customer_vals)}"
        )
    if len(demand_vals) != n_customers:
        raise ValueError(
            f"demands: expected {n_customers} ints, got {len(demand_vals)}"
        )
    if len(assign_vals) != n_customers:
        raise ValueError(
            f"assignment: expected {n_customers} ints, got {len(assign_vals)}"
        )

    depots = np.array(depot_vals, dtype=np.float64).reshape(n_depots, 2)
    customers = np.array(customer_vals, dtype=np.float64).reshape(n_customers, 2)
    demands = np.array(demand_vals, dtype=np.int64)
    random_assignment = np.array(assign_vals, dtype=np.int64) - 1  # to 0-indexed

    # Routes use 1-indexed *location* IDs (PyVRP convention, matching the
    # behaviour of `_to_one_indexed_routes` in data_generation/vrp_common.py).
    # Location ordering: depots first [1..K], then customers [K+1..K+N].
    # Convert to 0-indexed customer IDs in [0, n_customers - 1] by subtracting
    # 1 (1-indexed → 0-indexed) and then subtracting n_depots (drop depots).
    routes: list[list[int]] = []
    if routes_text.strip():
        for piece in routes_text.split(ROUTE_SEPARATOR):
            visit_ids = piece.strip().split()
            if not visit_ids:
                continue
            customer_ids = [int(v) - 1 - n_depots for v in visit_ids]
            for c in customer_ids:
                if not (0 <= c < n_customers):
                    raise ValueError(
                        f"Route contained out-of-range customer id "
                        f"{c} (n_customers={n_customers}, n_depots={n_depots}). "
                        f"Raw route token = '{piece}'."
                    )
            routes.append(customer_ids)

    return {
        "n_depots": n_depots,
        "n_customers": n_customers,
        "capacity": capacity,
        "num_vehicles_per_depot": num_vehicles_per_depot,
        "depots": depots,
        "customers": customers,
        "demands": demands,
        "random_assignment": random_assignment,
        "routes": routes,
    }


def _infer_depot_per_route(
    routes: list[list[int]], customers: np.ndarray, depots: np.ndarray
) -> np.ndarray:
    """
    Heuristic: assign each route to the depot nearest to that route's
    customer centroid. Returns a (n_customers,) int array of 0-indexed depots.

    Caveat: this is an approximation. To get exact labels, extend the data
    generation pipeline so MdvrpSample records the depot index per route
    (see _to_one_indexed_routes in vrp_common.py).
    """
    N = customers.shape[0]
    K = depots.shape[0]
    assignment = -np.ones(N, dtype=np.int64)

    for route in routes:
        if not route:
            continue
        centroid = customers[route].mean(axis=0)               # (2,)
        d2 = ((depots - centroid) ** 2).sum(axis=1)            # (K,)
        depot_idx = int(np.argmin(d2))
        for c in route:
            assignment[c] = depot_idx

    # Customers that somehow didn't appear in any route — fall back to
    # nearest-depot per customer. (Should not happen for a feasible PyVRP
    # output; defensive only.)
    missing = np.where(assignment < 0)[0]
    for c in missing:
        d2 = ((depots - customers[c]) ** 2).sum(axis=1)
        assignment[c] = int(np.argmin(d2))

    assert (assignment >= 0).all() and (assignment < K).all()
    return assignment


class MDVRPDataset(Dataset):
    """
    Args:
        file_path:     path to an MDVRP data file (one instance per line).
        num_customers: number of customers per instance.
        min_depots / max_depots: expected range of depot counts for
                                 sanity-checking. K is read from each line.
        customer_knn:  if > 0, add customer-customer k-NN edges (label=0)
                       for structural message passing only.
    """

    def __init__(
        self,
        file_path: str,
        num_customers: int,
        min_depots: int = 2,
        max_depots: int = 5,
        customer_knn: int = 0,
    ):
        self.num_customers = num_customers
        self.min_depots = min_depots
        self.max_depots = max_depots
        self.customer_knn = customer_knn

        self.file_lines = open(file_path).read().splitlines()
        self.file_lines = [line for line in self.file_lines if line.strip()]
        logger.info(
            f"Loaded {len(self.file_lines)} MDVRP-{num_customers} instances "
            f"(K in [{min_depots}, {max_depots}], customer_knn={customer_knn})"
        )

    def __len__(self):
        return len(self.file_lines)

    def _build_graph(self, parsed: dict):
        K = parsed["n_depots"]
        N = parsed["n_customers"]
        depots = parsed["depots"]
        customers = parsed["customers"]
        demands = parsed["demands"]
        capacity = parsed["capacity"]
        routes = parsed["routes"]

        if N != self.num_customers:
            raise ValueError(
                f"Expected {self.num_customers} customers, got {N}. "
                f"Set num_customers={N} if intentional."
            )

        # ---- node features (K + N, 4) ----
        coords = np.concatenate([depots, customers], axis=0)        # (K+N, 2)
        demand_norm = np.zeros(K + N, dtype=np.float32)
        demand_norm[K:] = demands.astype(np.float32) / max(capacity, 1)
        role = np.zeros(K + N, dtype=np.float32)
        role[K:] = 1.0  # 0 = depot, 1 = customer

        node_feat = np.stack(
            [
                coords[:, 0].astype(np.float32),
                coords[:, 1].astype(np.float32),
                demand_norm,
                role,
            ],
            axis=1,
        )
        node_feat_t = torch.from_numpy(node_feat).float()

        # ---- bipartite customer-depot edges ----
        # For every (customer c, depot d): two directed edges.
        cust_idx_global = np.arange(K, K + N)                       # (N,)
        depot_idx_global = np.arange(K)                             # (K,)
        # Cartesian product: for each customer, every depot.
        c_for_pair = np.repeat(cust_idx_global, K)                  # (N*K,)
        d_for_pair = np.tile(depot_idx_global, N)                   # (N*K,)

        # Both directions.
        bipartite_src = np.concatenate([c_for_pair, d_for_pair])
        bipartite_dst = np.concatenate([d_for_pair, c_for_pair])

        # Distances per directed edge.
        src_coords = coords[bipartite_src]
        dst_coords = coords[bipartite_dst]
        bipartite_dist = np.sqrt(((src_coords - dst_coords) ** 2).sum(axis=1)).astype(
            np.float32
        )

        # ---- labels for bipartite edges ----
        gt_assignment = _infer_depot_per_route(routes, customers, depots)  # (N,)
        # An undirected (c, d) edge is positive iff gt_assignment[c] == d.
        # For directed edges we set the label on both directions identically.
        labels_cd = (gt_assignment[None, :] == np.arange(K)[:, None]).T  # (N, K) bool
        labels_flat = labels_cd.astype(np.float32).reshape(-1)          # (N*K,)
        # Repeat for the two directions (cust→depot and depot→cust).
        bipartite_label = np.concatenate([labels_flat, labels_flat])    # (2*N*K,)

        # An "is bipartite" mask — separates structural-only customer-customer
        # edges (added below) from the predictive bipartite edges, so the loss
        # can ignore the former.
        bipartite_mask = np.ones(bipartite_src.shape[0], dtype=np.float32)

        # ---- optional: customer-customer k-NN (structural only) ----
        if self.customer_knn > 0 and N > 1:
            k = min(self.customer_knn + 1, N)
            kdt = KDTree(customers, leaf_size=30, metric="euclidean")
            knn_dist, knn_idx = kdt.query(customers, k=k, return_distance=True)
            knn_dist = knn_dist[:, 1:]                                  # drop self
            knn_idx = knn_idx[:, 1:] + K                                # → global idx
            cc_src = (np.arange(N).reshape(-1, 1).repeat(knn_idx.shape[1], axis=1) + K).reshape(-1)
            cc_dst = knn_idx.reshape(-1)
            cc_dist = knn_dist.reshape(-1).astype(np.float32)

            cc_label = np.zeros_like(cc_src, dtype=np.float32)
            cc_mask = np.zeros_like(cc_src, dtype=np.float32)           # excluded from loss

            edge_src = np.concatenate([bipartite_src, cc_src])
            edge_dst = np.concatenate([bipartite_dst, cc_dst])
            edge_dist = np.concatenate([bipartite_dist, cc_dist])
            edge_label = np.concatenate([bipartite_label, cc_label])
            edge_mask = np.concatenate([bipartite_mask, cc_mask])
        else:
            edge_src = bipartite_src
            edge_dst = bipartite_dst
            edge_dist = bipartite_dist
            edge_label = bipartite_label
            edge_mask = bipartite_mask

        edge_index = torch.from_numpy(np.stack([edge_src, edge_dst], axis=0)).long()
        edge_dist_t = torch.from_numpy(edge_dist).float()
        edge_label_t = torch.from_numpy(edge_label).float()
        edge_mask_t = torch.from_numpy(edge_mask).float()

        # Per-instance metadata needed by the decoder / trainer at val time.
        # ``n_bipartite_edges`` and ``n_total_edges`` are needed by the per-customer
        # K-way softmax loss in training_step so it can slice out the bipartite
        # portion of the super-graph's edge logits per instance.
        n_bipartite_edges = int(bipartite_src.shape[0])  # = 2 * K * N
        n_total_edges = int(edge_src.shape[0])
        meta = {
            "n_depots": K,
            "n_customers": N,
            "capacity": capacity,
            "num_vehicles_per_depot": parsed["num_vehicles_per_depot"],
            "gt_assignment": torch.from_numpy(gt_assignment).long(),
            "demands": torch.from_numpy(demands).long(),
            "n_bipartite_edges": n_bipartite_edges,
            "n_total_edges": n_total_edges,
        }

        return node_feat_t, edge_index, edge_dist_t, edge_label_t, edge_mask_t, meta

    def __getitem__(self, index: int):
        parsed = _parse_mdvrp_line(self.file_lines[index])
        return self._build_graph(parsed)


def collate_mdvrp(batch):
    """
    Collate B individual MDVRP graphs into one super-graph.

    Returns a 6-tuple:
        node_feat   : (sum_i (K_i + N_i), 4)
        edge_index  : (2, sum_i E_i)  with node offsets applied
        edge_dist   : (sum_i E_i,)
        edge_label  : (sum_i E_i,)
        edge_mask   : (sum_i E_i,)    1=use in loss (bipartite), 0=structural
        meta_list   : list[dict]      per-instance metadata for the decoder
    """
    all_node_feat = []
    all_edge_index = []
    all_edge_dist = []
    all_edge_label = []
    all_edge_mask = []
    all_meta: list[dict] = []

    node_offset = 0
    for node_feat, edge_index, edge_dist, edge_label, edge_mask, meta in batch:
        all_node_feat.append(node_feat)
        all_edge_index.append(edge_index + node_offset)
        all_edge_dist.append(edge_dist)
        all_edge_label.append(edge_label)
        all_edge_mask.append(edge_mask)

        # Preserve the per-instance node offset so the trainer can recover
        # which slice of the super-graph belongs to instance i.
        meta_with_offset = dict(meta)
        meta_with_offset["node_offset"] = node_offset
        meta_with_offset["n_total_nodes"] = int(node_feat.shape[0])
        all_meta.append(meta_with_offset)

        node_offset += node_feat.shape[0]

    return (
        torch.cat(all_node_feat, dim=0),
        torch.cat(all_edge_index, dim=1),
        torch.cat(all_edge_dist, dim=0),
        torch.cat(all_edge_label, dim=0),
        torch.cat(all_edge_mask, dim=0),
        all_meta,
    )


def test_mdvrp_dataset():
    from torch.utils.data import DataLoader

    dataset = MDVRPDataset(
        file_path="data/mdvrp50_pyvrp.txt",
        num_customers=50,
        min_depots=2,
        max_depots=5,
    )
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_mdvrp
    )
    for batch in dataloader:
        node_feat, edge_index, edge_dist, edge_label, edge_mask, meta = batch
        logger.debug(
            f"node_feat={node_feat.shape} edge_index={edge_index.shape} "
            f"positive={int(edge_label.sum().item())} "
            f"in_loss={int(edge_mask.sum().item())} "
            f"n_graphs={len(meta)}"
        )
        break


# uv run python -m difusco.mdvrp.dataset
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_mdvrp_dataset()
    _ = COORD_SCALE  # silence unused-import

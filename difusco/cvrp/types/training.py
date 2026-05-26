from difusco.cvrp.types.base import Schema


class EpochRecord(Schema):
    epoch: int
    loss: float
    lr: float
    time_s: float
    pred_length: float | None = None
    gt_length: float | None = None
    gap: float | None = None
    num_routes_pred: float | None = None
    num_routes_gt: float | None = None
    overcapacity_rate: float | None = None
    saved_best: bool = False


class FitResult(Schema):
    best_gap: float
    history: list[EpochRecord]

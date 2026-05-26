from difusco.tsp.types.base import Schema


class EpochRecord(Schema):
    epoch: int
    loss: float
    lr: float
    time_s: float
    pred_length: float | None = None
    gt_length: float | None = None
    gap: float | None = None
    saved_best: bool = False


class FitResult(Schema):
    best_gap: float
    history: list[EpochRecord]

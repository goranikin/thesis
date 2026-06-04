from difusco.mdvrp.types.base import Schema


class EpochRecord(Schema):
    epoch: int
    loss: float
    lr: float
    time_s: float
    assignment_accuracy: float | None = None
    capacity_violation_rate: float | None = None
    saved_best: bool = False


class FitResult(Schema):
    best_accuracy: float
    history: list[EpochRecord]

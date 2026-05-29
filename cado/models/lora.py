"""
LoRA wrapper + Hybrid Fine-Tuning for CADO (TSP and CVRP).

Implements Hybrid-FT from CADO Section 4.3:
  - LoRA on early GNN layers
  - Full retraining on the last GNN layer + output head
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 2):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)

        nn.init.normal_(self.lora_down.weight, std=1.0)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (1.0 / self.rank) * self.lora_up(self.lora_down(x))


def apply_hybrid_ft(
    model: nn.Module,
    lora_rank: int = 2,
    num_selective: int = 1,
) -> nn.Module:
    """
    Apply Hybrid Fine-Tuning in-place to a DIFUSCO/CADO model with
    ``backbone.layers`` and ``backbone.edge_head``.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    L = len(model.backbone.layers)
    if num_selective >= L:
        raise ValueError(
            f"num_selective={num_selective} must be < num_layers={L}; "
            "otherwise no layers receive LoRA."
        )

    for layer in model.backbone.layers[L - num_selective :]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in model.backbone.edge_head.parameters():
        p.requires_grad_(True)

    for layer in model.backbone.layers[: L - num_selective]:
        for name in ("P", "Q", "R", "U", "V"):
            base = getattr(layer, name)
            if not isinstance(base, nn.Linear):
                continue
            setattr(layer, name, LoRALinear(base, rank=lora_rank))

    return model


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}

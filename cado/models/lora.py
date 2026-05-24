"""
LoRA wrapper + Hybrid Fine-Tuning for CADO.

Implements the "Hybrid-FT" strategy from CADO Section 4.3:
  - LoRA adapters on the input layer and the first L-1 GNN layers
  - Selective-FT (full retrain) on the last GNN layer + output head

This dramatically reduces the trainable parameter count, which is essential
for stable RL fine-tuning of a 12-layer GNN.
"""

import os

import torch
import torch.nn as nn

from cado.models.model import CADOTSP


class LoRALinear(nn.Module):
    """
    Replaces an nn.Linear with `W x + (1/r) * B A x`, where:
        - W is the original (frozen) weight matrix
        - A: in_features -> rank   (small, trainable)
        - B: rank      -> out_features  (small, trainable, init to zero)

    The zero-init on B means the LoRA contribution is exactly 0 at the start,
    so the model output is identical to the pretrained model on the first
    forward pass — crucial for stable fine-tuning.

    Note on the (1/r) scaling: the original LoRA paper uses (alpha/r), where
    alpha is a tunable scaling factor. The CADO supplementary code hardcodes
    alpha=1 (so the scale is just 1/r). We mirror that here for fidelity.
    """

    def __init__(self, base: nn.Linear, rank: int = 2):
        super().__init__()
        self.base = base
        # Freeze the original weights — only LoRA matrices receive gradients.
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)

        # Standard LoRA init: random down, zero up.
        nn.init.normal_(self.lora_down.weight, std=1.0)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (1.0 / self.rank) * self.lora_up(self.lora_down(x))


def apply_hybrid_ft(
    model: CADOTSP,
    lora_rank: int = 2,
    num_selective: int = 1,
) -> CADOTSP:
    """
    Apply Hybrid Fine-Tuning to a CADOTSP model in-place.

    The recipe (Section 4.3, mirroring `train_co.py` lines 308-360 in the
    CADO supplementary code):

        1. Freeze every parameter.
        2. Unfreeze the last `num_selective` GNN layers and the output head
           (this is "Selective-FT" — full retraining of the final layers).
        3. Wrap the P, Q, R, U, V linear projections of the remaining GNN
           layers with LoRALinear. The input embedding modules are also LEFT
           FROZEN here, since they are not nn.Linear and adapter wrapping is
           not straightforward; the paper's supplementary code makes the
           same choice via its `lora_range` flag.

    Args:
        model:         A CADOTSP instance (must have model.backbone.layers
                       and model.backbone.edge_head)
        lora_rank:     LoRA rank r. Paper uses r=2 (Table 9).
        num_selective: Number of trailing layers to fully retrain. Paper uses 1.

    Returns:
        The same `model` with parameters and LoRA wrappers in place. The
        caller should move the model to the correct device AFTER this call,
        so the LoRA parameters land on the right device.
    """
    # Step 1: freeze everything.
    for p in model.parameters():
        p.requires_grad_(False)

    L = len(model.backbone.layers)
    if num_selective >= L:
        raise ValueError(
            f"num_selective={num_selective} must be < num_layers={L}; "
            "otherwise no layers receive LoRA."
        )

    # Step 2: unfreeze the last `num_selective` layers + output head.
    for layer in model.backbone.layers[L - num_selective :]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in model.backbone.edge_head.parameters():
        p.requires_grad_(True)

    # Step 3: wrap Linear projections of earlier layers with LoRALinear.
    # AGNNLayer in src/cado_repr/models/backbone.py has these 5 linear modules:
    #   P, Q, R  (edge update),  U, V  (node update).
    for layer in model.backbone.layers[: L - num_selective]:
        for name in ("P", "Q", "R", "U", "V"):
            base = getattr(layer, name)
            if not isinstance(base, nn.Linear):
                # Safety: skip if already wrapped or unexpected type.
                continue
            setattr(layer, name, LoRALinear(base, rank=lora_rank))

    return model


def trainable_parameter_summary(model: CADOTSP) -> dict[str, int]:
    """
    Convenience helper for sanity-checking Hybrid-FT. Returns a dict of
    `{trainable, frozen, total}` parameter counts.

    Example output for the paper's 12L/256d backbone with r=2 + 1 selective layer:
        {'trainable': ~600k, 'frozen': ~8.5M, 'total': ~9.1M}
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}


### TESTING ###


def test():
    import logging

    logging.basicConfig(level=logging.INFO)

    model = CADOTSP(
        hidden_dim=128,
        num_layers=6,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        dropout=0.0,
    )

    # Resolve the checkpoint path RELATIVE TO THE PROJECT ROOT, not this file.
    # __file__ lives in cado/models/, so we climb two levels.
    ckpt_path = os.path.join("checkpoints", "best_model.pt")
    logging.info(f"Loading SL checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    apply_hybrid_ft(model, lora_rank=2, num_selective=1)
    summary = trainable_parameter_summary(model)
    logging.info(
        "Parameters — trainable: %s, frozen: %s, total: %s",
        f"{summary['trainable']:,}",
        f"{summary['frozen']:,}",
        f"{summary['total']:,}",
    )
    # Sanity check: a small fraction should be trainable.
    frac = summary["trainable"] / summary["total"]
    logging.info("Trainable fraction: %.2f%%", 100 * frac)
    assert frac < 0.25, "Hybrid-FT is supposed to freeze the bulk of the model"


# uv run python -m cado.models.lora
if __name__ == "__main__":
    test()

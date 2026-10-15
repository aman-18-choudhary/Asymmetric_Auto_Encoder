"""Model-size utilities shared by all autoencoders."""

from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    """Trainable parameters per top-level child module (e.g. encoder vs decoder parts)."""
    return {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad)
        for name, child in model.named_children()
    }

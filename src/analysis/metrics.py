"""Metric helpers."""

from __future__ import annotations


def accuracy_from_logits(logits, targets) -> float:
    """Compute top-1 accuracy for a batch of logits."""
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

"""Evaluation helpers."""

from __future__ import annotations

from typing import Optional


def evaluate(model, loader, criterion, device: str = "cpu", max_batches: Optional[int] = None) -> dict[str, float]:
    """Evaluate a model without gradient updates."""
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    non_blocking = str(device).startswith("cuda")
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())
            total += batch_size

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total, "accuracy": correct / total}

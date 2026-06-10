"""Evaluation helpers."""

from __future__ import annotations


def evaluate(model, loader, criterion, device: str = "cpu") -> dict[str, float]:
    """Evaluate a model without gradient updates."""
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())
            total += batch_size

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total, "accuracy": correct / total}

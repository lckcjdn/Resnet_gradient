"""Minimal training loop utilities."""

from __future__ import annotations

from typing import Optional


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Run one training epoch and return aggregate metrics."""
    import torch

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((outputs.argmax(dim=1) == targets).sum().item())
        total += batch_size

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total, "accuracy": correct / total}

"""Minimal training loop utilities."""

from __future__ import annotations

import time
from typing import Optional


def _planned_batches(loader, max_batches: Optional[int]) -> Optional[int]:
    try:
        loader_batches = len(loader)
    except TypeError:
        return max_batches
    if max_batches is None:
        return loader_batches
    return min(loader_batches, max_batches)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    log_interval: int = 0,
    progress_prefix: str = "",
) -> dict[str, float]:
    """Run one training epoch and return aggregate metrics."""
    import torch

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    planned_batches = _planned_batches(loader, max_batches)
    non_blocking = str(device).startswith("cuda")
    started = time.perf_counter()
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((outputs.argmax(dim=1) == targets).sum().item())
        total += batch_size
        current_batch = batch_index + 1

        should_log = log_interval > 0 and current_batch % log_interval == 0
        if planned_batches is not None and current_batch == planned_batches:
            should_log = log_interval > 0
        if should_log:
            prefix = f"{progress_prefix} " if progress_prefix else ""
            batch_total = planned_batches if planned_batches is not None else "?"
            elapsed = time.perf_counter() - started
            print(
                f"{prefix}batch {current_batch}/{batch_total} "
                f"samples={total} loss={total_loss / total:.4f} "
                f"acc={correct / total:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total, "accuracy": correct / total}

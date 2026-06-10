"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def save_checkpoint(
    path: PathLike,
    model,
    optimizer=None,
    epoch: Optional[int] = None,
    metrics=None,
) -> None:
    """Save a model checkpoint."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics or {}}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: PathLike, model, optimizer=None):
    """Load a model checkpoint and return the payload."""
    import torch

    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload

"""Residual branch lesion helpers."""

from __future__ import annotations

import random


def select_dropped_blocks(
    num_blocks: int,
    drop_ratio: float,
    strategy: str,
    seed: int = 0,
) -> list[int]:
    """Select residual block indices to disable."""
    if not 0.0 <= drop_ratio <= 1.0:
        raise ValueError("drop_ratio must be between 0 and 1.")
    drop_count = round(num_blocks * drop_ratio)
    if drop_count == 0:
        return []

    indices = list(range(num_blocks))
    if strategy == "random_drop":
        rng = random.Random(seed)
        return sorted(rng.sample(indices, drop_count))
    if strategy == "early_drop":
        return indices[:drop_count]
    if strategy == "late_drop":
        return indices[-drop_count:]
    if strategy == "uniform_interval_drop":
        if drop_count == 1:
            return [num_blocks // 2]
        step = (num_blocks - 1) / (drop_count - 1)
        return sorted({round(i * step) for i in range(drop_count)})
    raise ValueError(f"Unknown lesion strategy: {strategy}")


def make_block_masks(num_blocks: int, dropped_indices: list[int]) -> list[float]:
    """Create 1/0 residual branch masks from dropped block indices."""
    dropped = set(dropped_indices)
    return [0.0 if index in dropped else 1.0 for index in range(num_blocks)]


def apply_block_lesion(model, dropped_indices: list[int]) -> list[float]:
    """Apply residual branch masks to models exposing `iter_blocks`."""
    if not hasattr(model, "iter_blocks"):
        raise TypeError("Model does not expose iter_blocks(); lesion requires residual blocks.")
    blocks = list(model.iter_blocks())
    masks = make_block_masks(len(blocks), dropped_indices)
    for block, mask in zip(blocks, masks):
        block.set_residual_mask(mask)
    return masks

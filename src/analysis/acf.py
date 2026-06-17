"""Autocorrelation helpers for one-dimensional gradient sequences."""

from __future__ import annotations

import numpy as np


EPS = 1e-12
GRADIENT_COLLAPSE_THRESHOLD = 1e-8


def autocorrelation_1d(values: np.ndarray, max_lag: int, eps: float = EPS) -> np.ndarray:
    """Compute the normalized ACF used in the Toy 1D Figure-2 experiment."""
    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    if sequence.size == 0:
        raise ValueError("ACF requires at least one value")

    max_lag = min(int(max_lag), sequence.size - 1)
    centered = sequence - float(np.mean(sequence))
    denominator = float(np.dot(centered, centered))
    if denominator <= eps:
        return np.asarray([1.0, *([np.nan] * max_lag)], dtype=np.float64)

    acf_values = [1.0]
    for lag in range(1, max_lag + 1):
        acf_values.append(float(np.dot(centered[:-lag], centered[lag:]) / denominator))
    return np.asarray(acf_values, dtype=np.float64)


def sequence_diagnostics(
    values: np.ndarray,
    threshold: float = GRADIENT_COLLAPSE_THRESHOLD,
) -> dict[str, float | bool]:
    """Return norm/std diagnostics and the collapse flag for a gradient sequence."""
    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    gradient_norm = float(np.linalg.norm(sequence))
    gradient_std = float(np.std(sequence))
    collapse_flag = gradient_norm < threshold or gradient_std < threshold
    return {
        "gradient_norm": gradient_norm,
        "gradient_std": gradient_std,
        "collapse_flag": bool(collapse_flag),
    }


def normalized_noise(kind: str, length: int, seed: int) -> np.ndarray:
    """Generate normalized white or Brown-noise reference sequences."""
    rng = np.random.default_rng(seed)
    if kind == "white":
        values = rng.standard_normal(length)
    elif kind == "brown":
        values = np.cumsum(rng.standard_normal(length))
    else:
        raise ValueError(f"Unsupported noise kind: {kind}")

    values = values.astype(np.float64)
    values = values - float(np.mean(values))
    std = float(np.std(values))
    if std <= EPS:
        return values
    return values / std

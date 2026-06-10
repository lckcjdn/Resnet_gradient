"""Layer-wise gradient statistics for residual block convolutions."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def collect_gradient_stats(model, epoch: int, model_name: str, seed: int, run_id: str) -> list[dict]:
    """Collect gradient statistics for modules whose name ends with `conv2`."""
    records = []
    layer_index = 0
    for module_name, module in model.named_modules():
        if not module_name.endswith("conv2") or not hasattr(module, "weight"):
            continue
        grad = module.weight.grad
        if grad is None:
            continue
        values = grad.detach().float()
        l2_norm = float(values.pow(2).sum().sqrt().item())
        l1_norm = float(values.abs().sum().item())
        abs_mean = float(values.abs().mean().item())
        abs_max = float(values.abs().max().item())
        records.append(
            {
                "epoch": epoch,
                "run_id": run_id,
                "model": model_name,
                "seed": seed,
                "layer_index": layer_index,
                "layer_name": f"{module_name}.weight",
                "grad_norm_l2": l2_norm,
                "grad_norm_l1": l1_norm,
                "grad_abs_mean": abs_mean,
                "grad_abs_max": abs_max,
                "log10_grad_norm": math.log10(l2_norm + 1e-12),
            }
        )
        layer_index += 1
    return records


def gradient_stability_summary(records: list[dict], k: int = 5) -> dict[str, float]:
    """Summarize layer-wise gradient stability."""
    if not records:
        return {
            "mean_grad_norm": 0.0,
            "std_grad_norm": 0.0,
            "min_grad_norm": 0.0,
            "max_grad_norm": 0.0,
            "shallow_grad_norm": 0.0,
            "deep_grad_norm": 0.0,
            "shallow_to_deep_ratio": 0.0,
            "log_grad_norm_variance": 0.0,
        }

    grad_norms = [float(row["grad_norm_l2"]) for row in records]
    log_norms = [float(row["log10_grad_norm"]) for row in records]
    mean_grad = sum(grad_norms) / len(grad_norms)
    variance = sum((value - mean_grad) ** 2 for value in grad_norms) / len(grad_norms)
    log_mean = sum(log_norms) / len(log_norms)
    log_variance = sum((value - log_mean) ** 2 for value in log_norms) / len(log_norms)
    first = grad_norms[:k]
    last = grad_norms[-k:]
    shallow = sum(first) / len(first)
    deep = sum(last) / len(last)
    return {
        "mean_grad_norm": mean_grad,
        "std_grad_norm": math.sqrt(variance),
        "min_grad_norm": min(grad_norms),
        "max_grad_norm": max(grad_norms),
        "shallow_grad_norm": shallow,
        "deep_grad_norm": deep,
        "shallow_to_deep_ratio": shallow / (deep + 1e-12),
        "log_grad_norm_variance": log_variance,
    }


def write_gradient_csv(records: list[dict], path: PathLike) -> None:
    """Write gradient records to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "run_id",
        "model",
        "seed",
        "layer_index",
        "layer_name",
        "grad_norm_l2",
        "grad_norm_l1",
        "grad_abs_mean",
        "grad_abs_max",
        "log10_grad_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

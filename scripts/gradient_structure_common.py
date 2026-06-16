"""Shared utilities for gradient-structure experiments.

The scripts in this group intentionally keep their outputs under the four
plan-specific result directories so they do not mix with earlier experiments.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12
GRADIENT_COLLAPSE_THRESHOLD = 1e-8
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class ExperimentModelSpec:
    slug: str
    label: str
    family: str
    shortcut_lambda: float = 1.0


MINIMUM_MODEL_SPECS = [
    ExperimentModelSpec("plain", "PlainNet", "PlainNet", 1.0),
    ExperimentModelSpec("resnet", "Standard ResNet", "ResNetV1", 1.0),
    ExperimentModelSpec("preact", "PreAct ResNet", "PreActResNet", 1.0),
    ExperimentModelSpec("scaled_l1p0", "ScaledShortcut lambda=1.0", "ScaledShortcutResNet", 1.0),
]

SCALED_ABLATION_SPECS = [
    ExperimentModelSpec("scaled_l0p5", "ScaledShortcut lambda=0.5", "ScaledShortcutResNet", 0.5),
    ExperimentModelSpec("scaled_l0p9", "ScaledShortcut lambda=0.9", "ScaledShortcutResNet", 0.9),
    ExperimentModelSpec("scaled_l1p0", "ScaledShortcut lambda=1.0", "ScaledShortcutResNet", 1.0),
    ExperimentModelSpec("scaled_l1p1", "ScaledShortcut lambda=1.1", "ScaledShortcutResNet", 1.1),
]

PLOT_COLORS = {
    "plain": "#333333",
    "resnet": "#1f77b4",
    "preact": "#2ca02c",
    "scaled_l0p5": "#9467bd",
    "scaled_l0p9": "#17becf",
    "scaled_l1p0": "#ff7f0e",
    "scaled_l1p1": "#d62728",
}


def model_specs(include_scaled_ablation: bool = False) -> list[ExperimentModelSpec]:
    if not include_scaled_ablation:
        return list(MINIMUM_MODEL_SPECS)
    merged: dict[str, ExperimentModelSpec] = {spec.slug: spec for spec in MINIMUM_MODEL_SPECS}
    for spec in SCALED_ABLATION_SPECS:
        merged[spec.slug] = spec
    return list(merged.values())


def set_global_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def ensure_subdirs(root: Path, names: Sequence[str] = ("figures", "tables", "raw", "metadata", "checkpoints")) -> dict[str, Path]:
    paths = {name: root / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_metadata(path: Path, command: str, extra: dict) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "python": sys.executable,
        **extra,
    }
    write_json(path, payload)


def append_run_docs(
    title: str,
    command: str,
    result_root: Path,
    figures: Sequence[Path],
    tables: Sequence[Path],
    notes: Sequence[str],
    dataset: str = "",
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    figure_lines = [f"- `{path}`" for path in figures]
    table_lines = [f"- `{path}`" for path in tables]
    note_lines = [f"- {note}" for note in notes]
    append_text(
        ROOT / "docs" / "experiment_log.md",
        "\n".join(
            [
                "",
                f"## Run: {title}",
                "",
                f"- Date: {now}",
                f"- Command: `{command}`",
                f"- Dataset: {dataset or 'not applicable'}",
                f"- Output root: `{result_root}`",
                "- Status: completed",
                "",
                "### Output Files",
                "",
                *figure_lines,
                *table_lines,
                "",
                "### Notes",
                "",
                *note_lines,
                "",
                "---",
                "",
            ]
        ),
    )
    append_text(
        ROOT / "docs" / "process_conclusions.md",
        "\n".join(
            [
                "",
                f"## {title} Observations",
                "",
                *notes,
                "",
            ]
        ),
    )
    artifact_text = "\n".join(
        [
            "",
            f"## Generated Artifacts: {title}",
            "",
            f"- Command: `{command}`",
            f"- Dataset: {dataset or 'not applicable'}",
            f"- Output root: `{result_root}`",
            "",
            "### Figures",
            *figure_lines,
            "",
            "### Tables",
            *table_lines,
            "",
        ]
    )
    append_text(ROOT / "docs" / "expected_figures_tables.md", artifact_text)
    expected_alt = ROOT / "docs" / "expected_figures_and_tables.md"
    if expected_alt.exists():
        append_text(expected_alt, artifact_text)


def save_line_plot(path: Path, x_values: Sequence[float], series: Sequence[tuple[str, Sequence[float], str]], title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, y_values, color in series:
        ax.plot(x_values, y_values, linewidth=1.5, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(path: Path, matrix: np.ndarray, title: str, xlabel: str = "sample index", ylabel: str = "sample index", vmin: float | None = None, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.3, 4.7))
    image = ax.imshow(matrix, cmap="coolwarm", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_errorbar_plot(path: Path, rows: Sequence[dict], x_key: str, y_key: str, group_key: str, title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    groups = sorted({str(row[group_key]) for row in rows})
    for group in groups:
        group_rows = [row for row in rows if str(row[group_key]) == group]
        xs = sorted({float(row[x_key]) for row in group_rows})
        ys = []
        yerrs = []
        for x in xs:
            vals = [float(row[y_key]) for row in group_rows if float(row[x_key]) == x]
            ys.append(float(np.mean(vals)))
            yerrs.append(float(np.std(vals)))
        slug = group.lower().replace(" ", "_").replace("=", "")
        ax.errorbar(xs, ys, yerr=yerrs, marker="o", linewidth=1.5, capsize=3, label=group, color=PLOT_COLORS.get(slug))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def covariance_from_vector(gradient: np.ndarray) -> np.ndarray:
    centered = gradient.astype(np.float64) - float(np.mean(gradient))
    variance = float(np.var(centered)) + EPS
    return np.outer(centered, centered) / variance


def acf_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = values.astype(np.float64)
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered)) + EPS
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        acf.append(float(np.dot(left, right) / denom))
    return np.asarray(acf, dtype=np.float64)


def covariance_structure_score(matrix: np.ndarray, band_width: int = 5) -> float:
    if matrix.size == 0:
        return 0.0
    rows, cols = matrix.shape
    yy, xx = np.ogrid[:rows, :cols]
    band = np.abs(yy - xx) <= band_width
    band_mean = float(np.mean(np.abs(matrix[band])))
    global_mean = float(np.mean(np.abs(matrix))) + EPS
    return band_mean / global_mean


def gradient_summary(
    *,
    model: str,
    depth: int,
    seed: int,
    gradient: np.ndarray,
    acf: np.ndarray,
    covariance_matrix: np.ndarray,
    extra: dict | None = None,
) -> dict:
    row = {
        "model": model,
        "depth": depth,
        "seed": seed,
        "gradient_mean": float(np.mean(gradient)),
        "gradient_std": float(np.std(gradient)),
        "gradient_l2_norm": float(np.linalg.norm(gradient)),
        "acf_lag1": float(acf[1]) if len(acf) > 1 else 0.0,
        "acf_lag5": float(acf[5]) if len(acf) > 5 else 0.0,
        "acf_lag10": float(acf[10]) if len(acf) > 10 else 0.0,
        "acf_area_under_curve": float(np.trapz(np.abs(acf[1:]))),
        "covariance_structure_score": covariance_structure_score(covariance_matrix),
    }
    if extra:
        row.update(extra)
    return row


def cosine_correlation(gradient_matrix: np.ndarray) -> np.ndarray:
    matrix = gradient_matrix.astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + EPS
    normalized = matrix / norms
    return normalized @ normalized.T


def gradient_diagnostic_label(
    mean_gradient_norm_value: float,
    gradient_std_value: float,
    threshold: float = GRADIENT_COLLAPSE_THRESHOLD,
) -> str:
    if mean_gradient_norm_value < threshold or gradient_std_value < threshold:
        return "gradient_collapse"
    return "shattered_gradient_candidate"


def diagnostic_allows_shattering(label: str) -> bool:
    return label != "gradient_collapse"


def effective_rank(gradient_matrix: np.ndarray) -> float:
    matrix = gradient_matrix.astype(np.float64)
    gram = matrix @ matrix.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.maximum(eigvals, 0.0)
    trace = float(np.sum(eigvals))
    spectral = float(np.max(eigvals)) if eigvals.size else 0.0
    if spectral <= EPS:
        return 0.0
    return trace / spectral


def relative_effective_rank(gradient_matrix: np.ndarray, seed: int) -> float:
    rank = effective_rank(gradient_matrix)
    rng = np.random.default_rng(seed)
    reference = rng.standard_normal(gradient_matrix.shape)
    reference_rank = effective_rank(reference)
    if reference_rank <= EPS:
        return 0.0
    return rank / reference_rank


def mean_gradient_norm(gradient_matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(gradient_matrix.astype(np.float64), axis=0)))


def label_structure_score(correlation: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    same = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    same_mask = same & off_diagonal
    diff_mask = (~same) & off_diagonal
    if not np.any(same_mask) or not np.any(diff_mask):
        return 0.0
    same_mean = float(np.mean(correlation[same_mask]))
    diff_mean = float(np.mean(correlation[diff_mask]))
    return same_mean - diff_mean


def spatial_acf_from_gradients(gradients: np.ndarray, max_lag: int = 15) -> np.ndarray:
    if gradients.ndim != 4:
        raise ValueError(f"Expected gradients with shape B,C,H,W, got {gradients.shape}")
    maps = np.linalg.norm(gradients.astype(np.float64), axis=1)
    values = [1.0]
    for lag in range(1, max_lag + 1):
        lag_scores = []
        for sample in maps:
            centered = sample - float(np.mean(sample))
            denom = float(np.sum(centered * centered)) + EPS
            vertical = float(np.sum(centered[:-lag, :] * centered[lag:, :]) / denom) if lag < centered.shape[0] else 0.0
            horizontal = float(np.sum(centered[:, :-lag] * centered[:, lag:]) / denom) if lag < centered.shape[1] else 0.0
            lag_scores.append((vertical + horizontal) / 2.0)
        values.append(float(np.mean(lag_scores)))
    return np.asarray(values, dtype=np.float64)


def cifar_model_label(spec: ExperimentModelSpec, depth: int) -> str:
    if spec.family == "ScaledShortcutResNet":
        return f"ScaledShortcutResNet-{depth}-lambda{spec.shortcut_lambda:.1f}"
    if spec.family == "PlainNet":
        return f"PlainNet-{depth}"
    if spec.family == "ResNetV1":
        return f"ResNet-{depth}"
    if spec.family == "PreActResNet":
        return f"PreActResNet-{depth}"
    return f"{spec.family}-{depth}"


def build_cifar_model(spec: ExperimentModelSpec, depth: int):
    from src.models import build_model

    return build_model(spec.family, depth=depth, num_classes=10, shortcut_lambda=spec.shortcut_lambda)


def collect_input_gradient_batch(model, inputs, targets, device: str):
    import torch
    from torch.nn import functional as F

    model.eval()
    inputs = inputs.to(device).detach().clone().requires_grad_(True)
    targets = targets.to(device)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    losses = F.cross_entropy(logits, targets, reduction="none")
    grad = torch.autograd.grad(losses.sum(), inputs, retain_graph=False, create_graph=False)[0]
    accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
    return {
        "gradients": grad.detach().cpu().numpy(),
        "gradient_matrix": grad.detach().flatten(1).cpu().numpy(),
        "loss": float(losses.mean().item()),
        "accuracy": accuracy,
    }


def load_cifar10_dataset(root: str, train: bool = False):
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return datasets.CIFAR10(root=root, train=train, transform=transform, download=False)


def make_fixed_batches(dataset_size: int, batch_size: int, num_batches: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    count = min(dataset_size, batch_size * num_batches)
    indices = rng.permutation(dataset_size)[:count].tolist()
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def dataset_batch(dataset, indices: Sequence[int]):
    import torch

    images = []
    labels = []
    for index in indices:
        image, label = dataset[int(index)]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def aggregate_rows(rows: Sequence[dict], keys: Sequence[str], value_keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key_values, group_rows in groups.items():
        out = {key: value for key, value in zip(keys, key_values)}
        for value_key in value_keys:
            values = [float(row[value_key]) for row in group_rows]
            out[f"{value_key}_mean"] = float(np.mean(values))
            out[f"{value_key}_std"] = float(np.std(values))
        output.append(out)
    return output


def normalize_epoch_name(name: str) -> str:
    return str(name).lower().replace("epoch", "")


def parse_epoch_value(name: str) -> int:
    norm = normalize_epoch_name(name)
    if norm == "final":
        return 10**9
    return int(norm)


def sort_epoch_names(names: Iterable[str]) -> list[str]:
    return sorted(names, key=parse_epoch_value)


def checkpoint_display_epoch(epoch_name: str, final_epoch: int | None = None) -> int:
    norm = normalize_epoch_name(epoch_name)
    if norm == "final":
        return int(final_epoch or 0)
    return int(norm)


def save_artifact_manifest(root: Path) -> Path:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
    manifest = root / "artifact_manifest.csv"
    write_csv(manifest, rows, ["path", "bytes", "modified"])
    return manifest


def print_progress(message: str) -> None:
    print(f"[gradient-structure] {message}", flush=True)

"""Small experimental pipelines used for smoke and identity-mapping runs."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.gradient_monitor import collect_gradient_stats, gradient_stability_summary
from src.analysis.metrics import count_parameters
from src.analysis.table_writer import csv_to_markdown
from src.data.cifar import build_small_image_loaders
from src.models import build_model
from src.training.evaluator import evaluate
from src.training.trainer import train_one_epoch


@dataclass(frozen=True)
class ModelSpec:
    label: str
    name: str
    depth: int
    shortcut_lambda: float = 1.0


@dataclass(frozen=True)
class RunSettings:
    result_root: Path
    data_root: str
    dataset: str
    download: bool
    train_size: int
    val_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int
    device: str
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    num_workers: int = 0
    torch_threads: int = 2
    log_interval: int = 100


def set_global_seed(seed: int) -> None:
    import numpy as np
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


def cuda_memory_text(device: str) -> str:
    if not str(device).startswith("cuda"):
        return ""
    import torch

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return f" cuda_mem={allocated:.1f}MiB reserved={reserved:.1f}MiB peak={peak:.1f}MiB"


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "logs": root / "logs",
        "figures": root / "figures",
        "tables": root / "tables",
        "gradients": root / "gradients",
        "checkpoints": root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clean_tag(tag: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in tag).strip("_")


def tagged_name(base: str, tag: str) -> str:
    tag = clean_tag(tag)
    return base if not tag else f"{base}_{tag}"


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _group_rows(rows: Iterable[Dict], key: str) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def plot_metric_curve(rows: Sequence[Dict], metric: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in _group_rows(rows, "model").items():
        ordered = sorted(group, key=lambda item: int(item["epoch"]))
        ax.plot(
            [int(item["epoch"]) for item in ordered],
            [float(item[metric]) for item in ordered],
            marker="o",
            label=label,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layerwise_gradients(rows: Sequence[Dict], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped = _group_rows(rows, "model")
    for label, group in grouped.items():
        last_epoch = max(int(item["epoch"]) for item in group)
        last_rows = sorted(
            [item for item in group if int(item["epoch"]) == last_epoch],
            key=lambda item: int(item["layer_index"]),
        )
        ax.plot(
            [int(item["layer_index"]) for item in last_rows],
            [float(item["log10_grad_norm"]) for item in last_rows],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("layer index")
    ax.set_ylabel("log10(gradient norm + 1e-12)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gradient_heatmap(rows: Sequence[Dict], output_path: Path, title: str, model: Optional[str] = None) -> None:
    filtered = [row for row in rows if model is None or row["model"] == model]
    if not filtered:
        return
    grouped = _group_rows(filtered, "model")
    figure_count = len(grouped)
    fig, axes = plt.subplots(figure_count, 1, figsize=(8, max(3, 2.4 * figure_count)), squeeze=False)
    for axis, (label, group) in zip(axes[:, 0], grouped.items()):
        epochs = sorted({int(item["epoch"]) for item in group})
        layers = sorted({int(item["layer_index"]) for item in group})
        matrix = []
        for epoch in epochs:
            epoch_rows = {int(item["layer_index"]): float(item["log10_grad_norm"]) for item in group if int(item["epoch"]) == epoch}
            matrix.append([epoch_rows.get(layer, math.nan) for layer in layers])
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
        axis.set_title(label)
        axis.set_xlabel("layer index")
        axis.set_ylabel("epoch")
        axis.set_xticks(range(len(layers)))
        axis.set_xticklabels(layers, fontsize=7)
        axis.set_yticks(range(len(epochs)))
        axis.set_yticklabels(epochs, fontsize=7)
        fig.colorbar(image, ax=axis, label="log10 grad norm")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lambda_ablation(summary_rows: Sequence[Dict], output_path: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if row["model_name"] == "ScaledShortcutResNet" or row["model"].startswith("Scaled")
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda item: float(item["shortcut_lambda"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [float(item["shortcut_lambda"]) for item in rows],
        [float(item["shallow_to_deep_ratio"]) for item in rows],
        marker="o",
    )
    ax.set_xlabel("shortcut lambda")
    ax.set_ylabel("shallow-to-deep gradient ratio")
    ax.set_title("Lambda Ablation Gradient Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_checkpoint(path: Path, model, spec: ModelSpec, metadata: Dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": spec.name,
            "model_label": spec.label,
            "depth": spec.depth,
            "shortcut_lambda": spec.shortcut_lambda,
            "metadata": metadata,
        },
        path,
    )


def run_model_suite(
    specs: Sequence[ModelSpec],
    settings: RunSettings,
    run_name: str,
    command: str,
    figure_names: Dict[str, str],
    table_prefix: str,
) -> Dict[str, object]:
    import torch
    from torch import nn, optim

    set_global_seed(settings.seed)
    torch.set_num_threads(max(1, settings.torch_threads))
    paths = ensure_dirs(settings.result_root)
    device = choose_device(settings.device)
    _, _, dataset_info = build_small_image_loaders(
        dataset=settings.dataset,
        root=settings.data_root,
        batch_size=settings.batch_size,
        train_size=settings.train_size,
        val_size=settings.val_size,
        seed=settings.seed,
        download=settings.download,
        num_workers=settings.num_workers,
        pin_memory=str(device).startswith("cuda"),
    )

    metrics_rows: List[Dict] = []
    gradient_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    status_rows: List[Dict] = []
    checkpoints: Dict[str, str] = {}
    started = datetime.now().isoformat(timespec="seconds")
    metrics_path = paths["tables"] / f"{table_prefix}_metrics.csv"
    model_table = paths["tables"] / f"{table_prefix}_model_comparison.csv"
    grad_table = paths["tables"] / f"{table_prefix}_gradient_stability.csv"
    status_table = paths["tables"] / f"{table_prefix}_status.csv"
    gradient_path = paths["gradients"] / f"{table_prefix}_gradient_stats.csv"
    print(
        f"[suite:{run_name}] device={device} dataset={dataset_info.name} "
        f"train={dataset_info.train_size} val={dataset_info.val_size} "
        f"epochs={settings.epochs} batch_size={settings.batch_size} "
        f"num_workers={settings.num_workers}",
        flush=True,
    )

    for spec_index, spec in enumerate(specs, start=1):
        model_started = time.perf_counter()
        status = "completed"
        notes = ""
        checkpoint_path = paths["checkpoints"] / f"{spec.label}.pt"
        try:
            train_loader, val_loader, dataset_info = build_small_image_loaders(
                dataset=settings.dataset,
                root=settings.data_root,
                batch_size=settings.batch_size,
                train_size=settings.train_size,
                val_size=settings.val_size,
                seed=settings.seed,
                download=settings.download,
                num_workers=settings.num_workers,
                pin_memory=str(device).startswith("cuda"),
            )
            set_global_seed(settings.seed)
            model = build_model(
                spec.name,
                depth=spec.depth,
                num_classes=10,
                shortcut_lambda=spec.shortcut_lambda,
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=settings.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            best_val_accuracy = 0.0
            final_grad_summary: Dict[str, float] = {}
            model_device = next(model.parameters()).device
            print(
                f"[model {spec_index}/{len(specs)}] {spec.label} start "
                f"params={count_parameters(model)} model_device={model_device}"
                f"{cuda_memory_text(device)}",
                flush=True,
            )
            for epoch in range(1, settings.epochs + 1):
                epoch_started = time.perf_counter()
                print(
                    f"[model {spec_index}/{len(specs)}:{spec.label}] "
                    f"epoch {epoch}/{settings.epochs} train start",
                    flush=True,
                )
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device=device,
                    max_batches=settings.max_train_batches,
                    log_interval=settings.log_interval,
                    progress_prefix=(
                        f"[model {spec_index}/{len(specs)}:{spec.label}] "
                        f"epoch {epoch}/{settings.epochs}"
                    ),
                )
                grad_records = collect_gradient_stats(
                    model,
                    epoch=epoch,
                    model_name=spec.label,
                    seed=settings.seed,
                    run_id=run_name,
                )
                gradient_rows.extend(grad_records)
                final_grad_summary = gradient_stability_summary(grad_records)
                print(
                    f"[model {spec_index}/{len(specs)}:{spec.label}] "
                    f"epoch {epoch}/{settings.epochs} eval start",
                    flush=True,
                )
                val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device=device,
                    max_batches=settings.max_eval_batches,
                )
                best_val_accuracy = max(best_val_accuracy, float(val_metrics["accuracy"]))
                epoch_runtime = time.perf_counter() - epoch_started
                print(
                    f"[model {spec_index}/{len(specs)}:{spec.label}] "
                    f"epoch {epoch}/{settings.epochs} done "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_acc={train_metrics['accuracy']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"best_val_acc={best_val_accuracy:.4f} "
                    f"elapsed={epoch_runtime:.1f}s"
                    f"{cuda_memory_text(device)}",
                    flush=True,
                )
                metrics_rows.append(
                    {
                        "run": run_name,
                        "model": spec.label,
                        "model_name": spec.name,
                        "depth": spec.depth,
                        "shortcut_lambda": spec.shortcut_lambda,
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "shallow_to_deep_ratio": final_grad_summary.get("shallow_to_deep_ratio", 0.0),
                        "log_grad_norm_variance": final_grad_summary.get("log_grad_norm_variance", 0.0),
                        "dataset": dataset_info.name,
                        "device": device,
                    }
                )
                write_csv(metrics_path, metrics_rows)
                write_csv(gradient_path, gradient_rows)

            runtime_sec = time.perf_counter() - model_started
            save_checkpoint(
                checkpoint_path,
                model,
                spec,
                {
                    "run": run_name,
                    "dataset": dataset_info.__dict__,
                    "epochs": settings.epochs,
                    "batch_size": settings.batch_size,
                    "learning_rate": settings.learning_rate,
                    "runtime_sec": runtime_sec,
                    "best_val_accuracy": best_val_accuracy,
                },
            )
            checkpoints[spec.label] = str(checkpoint_path)
            print(
                f"[model {spec_index}/{len(specs)}] {spec.label} finished "
                f"best_val_acc={best_val_accuracy:.4f} runtime={runtime_sec:.1f}s "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )
            last_metrics = [row for row in metrics_rows if row["model"] == spec.label][-1]
            summary_rows.append(
                {
                    "run": run_name,
                    "model": spec.label,
                    "model_name": spec.name,
                    "depth": spec.depth,
                    "shortcut_lambda": spec.shortcut_lambda,
                    "params": count_parameters(model),
                    "epochs": settings.epochs,
                    "final_train_loss": last_metrics["train_loss"],
                    "final_train_acc": last_metrics["train_accuracy"],
                    "final_val_loss": last_metrics["val_loss"],
                    "final_val_acc": last_metrics["val_accuracy"],
                    "best_val_acc": best_val_accuracy,
                    "mean_grad_norm": final_grad_summary.get("mean_grad_norm", 0.0),
                    "std_grad_norm": final_grad_summary.get("std_grad_norm", 0.0),
                    "min_grad_norm": final_grad_summary.get("min_grad_norm", 0.0),
                    "max_grad_norm": final_grad_summary.get("max_grad_norm", 0.0),
                    "shallow_grad_norm": final_grad_summary.get("shallow_grad_norm", 0.0),
                    "deep_grad_norm": final_grad_summary.get("deep_grad_norm", 0.0),
                    "shallow_to_deep_ratio": final_grad_summary.get("shallow_to_deep_ratio", 0.0),
                    "log_grad_norm_variance": final_grad_summary.get("log_grad_norm_variance", 0.0),
                    "runtime_sec": runtime_sec,
                    "dataset": dataset_info.name,
                    "seed": settings.seed,
                }
            )
        except Exception as exc:
            print(f"[model {spec_index}/{len(specs)}] {spec.label} failed: {exc}", flush=True)
            status = "failed"
            notes = repr(exc)
        status_rows.append(
            {
                "run": run_name,
                "model": spec.label,
                "status": status,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else "",
                "notes": notes,
            }
        )
        write_csv(status_table, status_rows)
        if summary_rows:
            write_csv(model_table, summary_rows)
            write_csv(grad_table, summary_rows)

    write_csv(metrics_path, metrics_rows)
    write_csv(model_table, summary_rows)
    write_csv(grad_table, summary_rows)
    write_csv(status_table, status_rows)
    write_csv(gradient_path, gradient_rows)

    for csv_path in (metrics_path, model_table, grad_table, status_table):
        csv_to_markdown(csv_path, Path("docs") / "tables" / f"{csv_path.stem}.md")

    figure_paths: List[Path] = []
    if metrics_rows:
        loss_path = paths["figures"] / figure_names["loss"]
        accuracy_path = paths["figures"] / figure_names["accuracy"]
        plot_metric_curve(metrics_rows, "train_loss", loss_path, f"{run_name} Training Loss")
        plot_metric_curve(metrics_rows, "val_accuracy", accuracy_path, f"{run_name} Validation Accuracy")
        figure_paths.extend([loss_path, accuracy_path])
    if gradient_rows:
        grad_path = paths["figures"] / figure_names["layerwise"]
        heatmap_path = paths["figures"] / figure_names["heatmap"]
        plot_layerwise_gradients(gradient_rows, grad_path, f"{run_name} Layer-wise Gradient Norm")
        plot_gradient_heatmap(gradient_rows, heatmap_path, f"{run_name} Gradient Heatmap")
        figure_paths.extend([grad_path, heatmap_path])

    log_path = paths["logs"] / f"{run_name}_log.md"
    log_text = [
        f"# {run_name} Log",
        "",
        f"- Started: {started}",
        f"- Command: `{command}`",
        f"- Python: `{sys.executable}`",
        f"- Device: `{device}`",
        f"- Dataset: `{dataset_info.name}`",
        f"- Train size: {dataset_info.train_size}",
        f"- Validation size: {dataset_info.val_size}",
        f"- Fallback note: {dataset_info.fallback_reason or 'none'}",
        f"- Epochs: {settings.epochs}",
        f"- Batch size: {settings.batch_size}",
        f"- Learning rate: {settings.learning_rate}",
        "",
        "## Outputs",
        "",
    ]
    for path in figure_paths + [metrics_path, model_table, grad_table, status_table, gradient_path]:
        log_text.append(f"- `{path.as_posix()}`")
    log_path.write_text("\n".join(log_text) + "\n", encoding="utf-8")

    summary_path = paths["tables"] / f"{table_prefix}_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run": run_name,
                "command": command,
                "dataset": dataset_info.__dict__,
                "figures": [str(path) for path in figure_paths],
                "tables": [str(path) for path in (model_table, grad_table, status_table)],
                "gradient_path": str(gradient_path),
                "checkpoints": checkpoints,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_rows,
        "gradients": gradient_rows,
        "summary": summary_rows,
        "status": status_rows,
        "figures": figure_paths,
        "tables": [model_table, grad_table, status_table],
        "metrics_path": metrics_path,
        "gradient_path": gradient_path,
        "log_path": log_path,
        "checkpoints": checkpoints,
        "dataset_info": dataset_info,
    }


def append_common_docs(run_name: str, command: str, result: Dict[str, object], observations: str) -> None:
    dataset_info = result["dataset_info"]
    figures = result["figures"]
    tables = result["tables"]
    gradient_path = result["gradient_path"]
    log_path = result["log_path"]
    now = datetime.now().isoformat(timespec="seconds")
    append(
        Path("docs") / "experiment_log.md",
        "\n".join(
            [
                "",
                f"## Run: {run_name}",
                "",
                f"- Date: {now}",
                f"- Command: `{command}`",
                f"- Dataset: {dataset_info.name}",
                f"- Fallback note: {dataset_info.fallback_reason or 'none'}",
                f"- Log: `{log_path}`",
                f"- Gradient stats: `{gradient_path}`",
                f"- Figures: {', '.join(f'`{path}`' for path in figures)}",
                f"- Tables: {', '.join(f'`{path}`' for path in tables)}",
                "- Status: completed",
                "",
                "---",
                "",
            ]
        ),
    )
    append(
        Path("docs") / "process_conclusions.md",
        "\n".join(
            [
                "",
                f"## {run_name} Observations",
                "",
                observations,
                "",
            ]
        ),
    )
    generated_text = "\n".join(
        [
            "",
            f"## Generated Artifacts: {run_name}",
            "",
            f"- Command: `{command}`",
            f"- Dataset: {dataset_info.name}",
            "",
            "### Figures",
            *[f"- `{path}`" for path in figures],
            "",
            "### Tables",
            *[f"- `{path}`" for path in tables],
            f"- `{gradient_path}`",
            "",
        ]
    )
    append(Path("docs") / "expected_figures_tables.md", generated_text)
    append(Path("docs") / "expected_figures_and_tables.md", generated_text)


def smoke_specs() -> List[ModelSpec]:
    return [
        ModelSpec("PlainNet-20", "PlainNet", 20, 1.0),
        ModelSpec("ResNet-20", "ResNetV1", 20, 1.0),
        ModelSpec("PreActResNet-20", "PreActResNet", 20, 1.0),
        ModelSpec("ScaledShortcutResNet-20-lambda1.0", "ScaledShortcutResNet", 20, 1.0),
    ]


def identity_specs() -> List[ModelSpec]:
    return [
        ModelSpec("PlainNet-56", "PlainNet", 56, 1.0),
        ModelSpec("ResNet-56", "ResNetV1", 56, 1.0),
        ModelSpec("PreActResNet-56", "PreActResNet", 56, 1.0),
        ModelSpec("ScaledShortcutResNet-56-lambda0.5", "ScaledShortcutResNet", 56, 0.5),
        ModelSpec("ScaledShortcutResNet-56-lambda0.9", "ScaledShortcutResNet", 56, 0.9),
        ModelSpec("ScaledShortcutResNet-56-lambda1.0", "ScaledShortcutResNet", 56, 1.0),
        ModelSpec("ScaledShortcutResNet-56-lambda1.1", "ScaledShortcutResNet", 56, 1.1),
    ]


def run_smoke(args) -> Dict[str, object]:
    output_tag = clean_tag(getattr(args, "output_tag", ""))
    run_name = tagged_name("smoke_test", output_tag)
    table_prefix = tagged_name("smoke", output_tag)
    output_root = getattr(args, "output_root", "")
    result_root = (
        Path(output_root)
        if output_root
        else Path("results")
        if not output_tag
        else Path("results") / f"smoke_{output_tag}"
    )
    settings = RunSettings(
        result_root=result_root,
        data_root=getattr(args, "data_root", "data"),
        dataset=args.dataset,
        download=args.download,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        num_workers=getattr(args, "num_workers", 0),
        torch_threads=args.torch_threads,
        log_interval=getattr(args, "log_interval", 100),
    )
    command = " ".join(sys.argv)
    result = run_model_suite(
        smoke_specs(),
        settings,
        run_name=run_name,
        command=command,
        figure_names={
            "loss": "smoke_loss_curve.png",
            "accuracy": "smoke_accuracy_curve.png",
            "layerwise": "smoke_layerwise_grad_norm.png",
            "heatmap": "smoke_gradient_heatmap.png",
        },
        table_prefix=table_prefix,
    )
    append_common_docs(
        "Smoke Test" if not output_tag else f"Smoke Test ({output_tag})",
        command,
        result,
        (
            "The smoke test exercises model construction, one short training loop, "
            "evaluation, gradient recording, figure generation, and table generation. "
            "These outputs validate the pipeline mechanics; they should not be used as "
            "scientific evidence about ResNet behavior."
        ),
    )
    reproducibility = Path("docs") / "reproducibility.md"
    if not reproducibility.exists():
        reproducibility.write_text("# Reproducibility\n", encoding="utf-8")
    append(
        reproducibility,
        "\n".join(
            [
                "",
                "## Smoke Test Command",
                "",
                "Run inside the project conda environment:",
                "",
                "```bash",
                "conda activate resnet-gradient-path-study",
                command,
                "```",
                "",
                "The smoke test uses a small subset and may fall back to `torchvision.datasets.FakeData` when CIFAR-10 is unavailable.",
                "",
            ]
        ),
    )
    return result


def run_identity(args) -> Dict[str, object]:
    output_tag = clean_tag(getattr(args, "output_tag", ""))
    run_name = tagged_name("identity_mapping", output_tag)
    table_prefix = tagged_name("identity", output_tag)
    output_root = getattr(args, "output_root", "")
    if output_root:
        result_root = Path(output_root)
    elif output_tag:
        result_root = Path("results") / f"identity_mapping_{output_tag}"
    else:
        result_root = Path("results") / "identity_mapping"
    settings = RunSettings(
        result_root=result_root,
        data_root=getattr(args, "data_root", "data"),
        dataset=args.dataset,
        download=args.download,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        num_workers=getattr(args, "num_workers", 0),
        torch_threads=args.torch_threads,
        log_interval=getattr(args, "log_interval", 100),
    )
    command = " ".join(sys.argv)
    result = run_model_suite(
        identity_specs(),
        settings,
        run_name=run_name,
        command=command,
        figure_names={
            "loss": "fig_identity_loss_curve.png",
            "accuracy": "fig_identity_accuracy_curve.png",
            "layerwise": "fig_identity_layerwise_grad_norm.png",
            "heatmap": "fig_identity_gradient_heatmap_all.png",
        },
        table_prefix=table_prefix,
    )

    gradients = result["gradients"]
    fig_dir = result_root / "figures"
    for model, filename in [
        ("PlainNet-56", "fig_identity_gradient_heatmap_plainnet.png"),
        ("ResNet-56", "fig_identity_gradient_heatmap_resnet.png"),
        ("PreActResNet-56", "fig_identity_gradient_heatmap_preact_resnet.png"),
    ]:
        plot_gradient_heatmap(gradients, fig_dir / filename, f"{model} Gradient Heatmap", model=model)
        result["figures"].append(fig_dir / filename)
    lambda_path = fig_dir / "fig_identity_lambda_ablation_grad_ratio.png"
    plot_lambda_ablation(result["summary"], lambda_path)
    result["figures"].append(lambda_path)

    table_dir = result_root / "tables"
    table_stem = tagged_name("table_identity", output_tag)
    model_table = table_dir / f"{table_stem}_model_comparison.csv"
    gradient_table = table_dir / f"{table_stem}_gradient_stability.csv"
    lambda_table = table_dir / f"{table_stem}_lambda_ablation.csv"
    write_csv(model_table, result["summary"])
    write_csv(gradient_table, result["summary"])
    write_csv(
        lambda_table,
        [row for row in result["summary"] if row["model_name"] == "ScaledShortcutResNet"],
    )
    for path in (model_table, gradient_table, lambda_table):
        csv_to_markdown(path, Path("docs") / "tables" / f"{path.stem}.md")
    result["tables"] = [model_table, gradient_table, lambda_table]

    append_common_docs(
        "Identity Mapping Experiment" if not output_tag else f"Identity Mapping Experiment ({output_tag})",
        command,
        result,
        (
            "This lightweight identity-mapping run produced comparable metrics and "
            "layer-wise gradient statistics for PlainNet, standard ResNet, PreAct "
            "ResNet, and scaled shortcut variants. Because the run is intentionally "
            "small, conclusions should be treated as preliminary pipeline evidence."
        ),
    )
    write_identity_doc(command, result, output_tag)
    return result


def write_identity_doc(command: str, result: Dict[str, object], output_tag: str = "") -> None:
    dataset_info = result["dataset_info"]
    figures = result["figures"]
    tables = result["tables"]
    dataset_limit = (
        "- This run used real CIFAR-10 data, but still only a subset and a short training budget."
        if dataset_info.name == "CIFAR10"
        else "- FakeData fallback validates pipeline behavior rather than CIFAR-10 scientific trends."
    )
    text = [
        "# Identity Mapping Experiment",
        "",
        "## Objective",
        "",
        "Evaluate whether shortcuts close to identity mapping are associated with more stable optimization and gradient propagation.",
        "",
        "## Theory Background",
        "",
        "For a residual block `x_{l+1} = x_l + F_l(x_l)`, the identity term creates a direct gradient component. Scaled shortcuts test `x_{l+1} = lambda * x_l + F_l(x_l)`.",
        "",
        "## Compared Models",
        "",
        "- PlainNet-56",
        "- Standard ResNet-56",
        "- PreAct ResNet-56",
        "- ScaledShortcut ResNet-56 with lambda values 0.5, 0.9, 1.0, and 1.1",
        "",
        "## Training Settings",
        "",
        f"- Command: `{command}`",
        f"- Dataset: {dataset_info.name}",
        f"- Dataset note: {dataset_info.fallback_reason or 'none'}",
        "",
        "## Generated Figures",
        "",
        *[f"- `{path}`" for path in figures],
        "",
        "## Generated Tables",
        "",
        *[f"- `{path}`" for path in tables],
        "",
        "## Preliminary Interpretation",
        "",
        "The run provides saved evidence for comparing optimization and gradient statistics. Because the setting is lightweight, wording should remain cautious: results may support or be consistent with the identity shortcut hypothesis, but they do not prove it generally.",
        "",
        "## Limitations",
        "",
        "- Small dataset subset and short training budget.",
        dataset_limit,
        "- Single seed unless rerun with additional seeds.",
        "",
    ]
    doc_name = "identity_mapping_experiment.md" if not output_tag else f"identity_mapping_experiment_{output_tag}.md"
    Path("docs", doc_name).write_text("\n".join(text), encoding="utf-8")

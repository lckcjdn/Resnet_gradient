"""Residual branch lesion study runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.lesion import apply_block_lesion, select_dropped_blocks
from src.analysis.table_writer import csv_to_markdown
from src.data.cifar import build_small_image_loaders
from src.models import build_model
from src.training.evaluator import evaluate


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


def ensure_dirs() -> Dict[str, Path]:
    root = Path("results") / "lesion_study"
    paths = {
        "logs": root / "logs",
        "figures": root / "figures",
        "tables": root / "tables",
        "masks": root / "masks",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def find_checkpoint(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("results/identity_mapping/checkpoints/PreActResNet-56.pt"),
            Path("results/identity_mapping/checkpoints/ResNet-56.pt"),
            Path("results/checkpoints/PreActResNet-20.pt"),
            Path("results/checkpoints/ResNet-20.pt"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No usable ResNet checkpoint found for lesion study.")


def load_model_from_checkpoint(checkpoint_path: Path, device: str):
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        payload["model_name"],
        depth=int(payload["depth"]),
        shortcut_lambda=float(payload.get("shortcut_lambda", 1.0)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def choose_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def save_mask(path: Path, masks: Sequence[float], dropped: Sequence[int]) -> None:
    rows = [
        {
            "block_index": index,
            "mask": mask,
            "dropped": int(index in set(dropped)),
        }
        for index, mask in enumerate(masks)
    ]
    write_csv(path, rows, fieldnames=["block_index", "mask", "dropped"])


def aggregate_rows(rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["drop_strategy"], row["drop_ratio"])].append(row)

    aggregate = []
    for (strategy, ratio), group in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        accuracies = [float(row["test_accuracy"]) for row in group]
        drops = [float(row["accuracy_drop"]) for row in group]
        losses = [float(row["test_loss"]) for row in group]
        active = [float(row["active_blocks"]) for row in group]
        mean_acc = sum(accuracies) / len(accuracies)
        mean_drop = sum(drops) / len(drops)
        mean_loss = sum(losses) / len(losses)
        mean_active = sum(active) / len(active)
        std_acc = math.sqrt(sum((value - mean_acc) ** 2 for value in accuracies) / len(accuracies))
        aggregate.append(
            {
                "drop_strategy": strategy,
                "drop_ratio": ratio,
                "mean_test_accuracy": mean_acc,
                "std_test_accuracy": std_acc,
                "mean_accuracy_drop": mean_drop,
                "mean_test_loss": mean_loss,
                "mean_active_blocks": mean_active,
                "runs": len(group),
            }
        )
    return aggregate


def plot_by_strategy(rows: Sequence[Dict], y_key: str, output_path: Path, title: str, ylabel: str) -> None:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["drop_strategy"]].append(row)
    fig, ax = plt.subplots(figsize=(7, 4))
    for strategy, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: float(item["drop_ratio"]))
        ax.plot(
            [float(item["drop_ratio"]) for item in ordered],
            [float(item[y_key]) for item in ordered],
            marker="o",
            label=strategy,
        )
    ax.set_xlabel("drop ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_active_blocks(rows: Sequence[Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["drop_strategy"]].append(row)
    for strategy, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: float(item["mean_active_blocks"]))
        ax.plot(
            [float(item["mean_active_blocks"]) for item in ordered],
            [float(item["mean_test_accuracy"]) for item in ordered],
            marker="o",
            label=strategy,
        )
    ax.set_xlabel("active residual branches")
    ax.set_ylabel("mean test accuracy")
    ax.set_title("Active Blocks vs Accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows: Sequence[Dict], output_path: Path) -> None:
    strategies = sorted({row["drop_strategy"] for row in rows})
    ratios = sorted({float(row["drop_ratio"]) for row in rows})
    lookup = {(row["drop_strategy"], float(row["drop_ratio"])): float(row["mean_test_accuracy"]) for row in rows}
    matrix = [[lookup.get((strategy, ratio), math.nan) for ratio in ratios] for strategy in strategies]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([str(ratio) for ratio in ratios])
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel("drop ratio")
    ax.set_title("Lesion Accuracy Heatmap")
    fig.colorbar(image, ax=ax, label="mean test accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_lesion(args) -> Dict[str, object]:
    import torch
    from torch import nn

    torch.set_num_threads(max(1, args.torch_threads))
    paths = ensure_dirs()
    device = choose_device(args.device)
    checkpoint_path = find_checkpoint(args.checkpoint)
    model, payload = load_model_from_checkpoint(checkpoint_path, device)
    if not hasattr(model, "iter_blocks"):
        raise TypeError("Selected checkpoint model does not support residual branch masking.")

    _, val_loader, dataset_info = build_small_image_loaders(
        dataset=args.dataset,
        root="data",
        batch_size=args.batch_size,
        train_size=args.val_size,
        val_size=args.val_size,
        seed=args.seed,
        download=args.download,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()
    baseline = evaluate(model, val_loader, criterion, device=device, max_batches=args.max_eval_batches)
    baseline_accuracy = float(baseline["accuracy"])
    blocks = list(model.iter_blocks())
    num_blocks = len(blocks)

    ratios = [float(value) for value in args.drop_ratios.split(",")]
    random_seeds = [int(value) for value in args.random_seeds.split(",")]
    strategies = ["random_drop", "early_drop", "late_drop"]
    rows: List[Dict] = []
    started = datetime.now().isoformat(timespec="seconds")
    command = " ".join(sys.argv)

    for strategy in strategies:
        seeds = random_seeds if strategy == "random_drop" else [args.seed]
        for ratio in ratios:
            for lesion_seed in seeds:
                model.set_block_masks([1.0] * num_blocks)
                dropped = select_dropped_blocks(num_blocks, ratio, strategy, lesion_seed)
                masks = apply_block_lesion(model, dropped)
                mask_name = f"{strategy}_ratio{str(ratio).replace('.', 'p')}_seed{lesion_seed}.csv"
                mask_path = paths["masks"] / mask_name
                save_mask(mask_path, masks, dropped)
                eval_started = time.perf_counter()
                metrics = evaluate(model, val_loader, criterion, device=device, max_batches=args.max_eval_batches)
                runtime_sec = time.perf_counter() - eval_started
                active_blocks = int(sum(masks))
                rows.append(
                    {
                        "model": payload.get("model_label", checkpoint_path.stem),
                        "checkpoint": str(checkpoint_path),
                        "drop_strategy": strategy,
                        "drop_ratio": ratio,
                        "seed": lesion_seed,
                        "test_loss": metrics["loss"],
                        "test_accuracy": metrics["accuracy"],
                        "baseline_accuracy": baseline_accuracy,
                        "accuracy_drop": baseline_accuracy - float(metrics["accuracy"]),
                        "num_total_blocks": num_blocks,
                        "num_dropped_blocks": len(dropped),
                        "active_blocks": active_blocks,
                        "effective_path_length_approx": active_blocks,
                        "mask_path": str(mask_path),
                        "runtime_sec": runtime_sec,
                        "dataset": dataset_info.name,
                    }
                )

    model.set_block_masks([1.0] * num_blocks)
    aggregate = aggregate_rows(rows)

    table_accuracy = paths["tables"] / "table_lesion_accuracy.csv"
    table_strategy = paths["tables"] / "table_lesion_drop_strategy_comparison.csv"
    table_active = paths["tables"] / "table_active_blocks_summary.csv"
    write_csv(table_accuracy, rows)
    write_csv(table_strategy, aggregate)
    write_csv(table_active, aggregate)
    for table in (table_accuracy, table_strategy, table_active):
        csv_to_markdown(table, Path("docs") / "tables" / f"{table.stem}.md")

    fig_accuracy = paths["figures"] / "fig_lesion_accuracy_vs_drop_ratio.png"
    fig_drop = paths["figures"] / "fig_lesion_accuracy_drop_vs_drop_ratio.png"
    fig_strategy = paths["figures"] / "fig_lesion_random_vs_early_vs_late.png"
    fig_active = paths["figures"] / "fig_active_blocks_vs_accuracy.png"
    fig_heatmap = paths["figures"] / "fig_lesion_heatmap.png"
    plot_by_strategy(aggregate, "mean_test_accuracy", fig_accuracy, "Lesion Accuracy vs Drop Ratio", "mean test accuracy")
    plot_by_strategy(aggregate, "mean_accuracy_drop", fig_drop, "Accuracy Drop vs Drop Ratio", "mean accuracy drop")
    plot_by_strategy(aggregate, "mean_test_accuracy", fig_strategy, "Random vs Early vs Late Lesion", "mean test accuracy")
    plot_active_blocks(aggregate, fig_active)
    plot_heatmap(aggregate, fig_heatmap)

    log_path = paths["logs"] / "lesion_study_log.md"
    log_path.write_text(
        "\n".join(
            [
                "# Lesion Study Log",
                "",
                f"- Started: {started}",
                f"- Command: `{command}`",
                f"- Checkpoint: `{checkpoint_path}`",
                f"- Model: {payload.get('model_label', checkpoint_path.stem)}",
                f"- Dataset: {dataset_info.name}",
                f"- Dataset note: {dataset_info.fallback_reason or 'none'}",
                f"- Baseline accuracy: {baseline_accuracy}",
                f"- Total residual blocks: {num_blocks}",
                f"- Drop ratios: {ratios}",
                f"- Random seeds: {random_seeds}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "model": payload.get("model_label", checkpoint_path.stem),
        "dataset": dataset_info.__dict__,
        "baseline": baseline,
        "tables": [str(table_accuracy), str(table_strategy), str(table_active)],
        "figures": [str(fig_accuracy), str(fig_drop), str(fig_strategy), str(fig_active), str(fig_heatmap)],
        "log": str(log_path),
    }
    (paths["tables"] / "lesion_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_docs(command, summary, dataset_info.fallback_reason or "none")
    return summary


def update_docs(command: str, summary: Dict[str, object], dataset_note: str) -> None:
    figures = [Path(path) for path in summary["figures"]]
    tables = [Path(path) for path in summary["tables"]]
    now = datetime.now().isoformat(timespec="seconds")
    append(
        Path("docs") / "experiment_log.md",
        "\n".join(
            [
                "",
                "## Run: Short-path Ensemble Lesion Study",
                "",
                f"- Date: {now}",
                f"- Command: `{command}`",
                f"- Checkpoint: `{summary['checkpoint']}`",
                f"- Dataset note: {dataset_note}",
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
                "## Lesion Study Observations",
                "",
                "The lesion study verifies that residual branch masks can be applied without breaking the forward path. Because this run uses a lightweight checkpoint and may use FakeData fallback, the observed accuracy trend should be interpreted as pipeline evidence and a preliminary illustration rather than a final scientific result.",
                "",
            ]
        ),
    )
    generated_text = "\n".join(
        [
            "",
            "## Generated Artifacts: Short-path Ensemble Lesion Study",
            "",
            f"- Command: `{command}`",
            f"- Checkpoint: `{summary['checkpoint']}`",
            "",
            "### Figures",
            *[f"- `{path}`" for path in figures],
            "",
            "### Tables",
            *[f"- `{path}`" for path in tables],
            "",
        ]
    )
    append(Path("docs") / "expected_figures_tables.md", generated_text)
    append(Path("docs") / "expected_figures_and_tables.md", generated_text)

    text = [
        "# Short-path Ensemble Lesion Experiment",
        "",
        "## Objective",
        "",
        "Evaluate whether a trained ResNet remains partially functional when selected residual branches are removed.",
        "",
        "## Theory Background",
        "",
        "Residual blocks contain a shortcut path and a residual branch. Lesion masks evaluate `y = x + m_l * F(x)`, where `m_l = 0` disables a residual branch while preserving the shortcut.",
        "",
        "## Lesion Method",
        "",
        f"- Checkpoint used: `{summary['checkpoint']}`",
        "- Drop ratios: 0%, 10%, 30%, 50%, 70%",
        "- Random Drop uses multiple seeds when requested.",
        "- Early Drop removes branches near the input side.",
        "- Late Drop removes branches near the output side.",
        "",
        "## Generated Figures",
        "",
        *[f"- `{path}`" for path in figures],
        "",
        "## Generated Tables",
        "",
        *[f"- `{path}`" for path in tables],
        "",
        "## Conceptual PlainNet Comparison",
        "",
        "PlainNet cannot remove intermediate layers without breaking the only forward path. In contrast, ResNet can disable residual branches while shortcuts still preserve an information path.",
        "",
        "## Preliminary Interpretation",
        "",
        "The generated tables and figures show the mechanics of residual branch lesion evaluation. If accuracy declines gradually rather than failing immediately, that behavior is consistent with short-path ensemble interpretations.",
        "",
        "## Limitations",
        "",
        "- The checkpoint is lightweight and may be trained on FakeData fallback.",
        "- Accuracy values should not be used as final CIFAR-10 evidence unless rerun with real CIFAR-10 and longer training.",
        "- Results support cautious interpretation only.",
        "",
    ]
    Path("docs/short_path_ensemble_experiment.md").write_text("\n".join(text), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run residual branch lesion study.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", choices=["auto", "cifar10", "fake"], default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--val-size", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--drop-ratios", default="0,0.1,0.3,0.5,0.7")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args(argv)
    run_lesion(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

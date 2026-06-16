"""Run CIFAR-10 checkpoint input-gradient structure analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.gradient_structure_common import (
    PLOT_COLORS,
    aggregate_rows,
    append_run_docs,
    build_cifar_model,
    cifar_model_label,
    checkpoint_display_epoch,
    choose_device,
    collect_input_gradient_batch,
    cosine_correlation,
    dataset_batch,
    ensure_subdirs,
    label_structure_score,
    load_cifar10_dataset,
    make_fixed_batches,
    mean_gradient_norm,
    model_specs,
    normalize_epoch_name,
    print_progress,
    read_csv,
    relative_effective_rank,
    save_heatmap,
    set_global_seed,
    sort_epoch_names,
    spatial_acf_from_gradients,
    write_csv,
    write_json,
    write_run_metadata,
)


def final_epoch_from_metrics(metrics_rows: list[dict]) -> int:
    epochs = [int(float(row["epoch"])) for row in metrics_rows if row.get("epoch", "").strip()]
    return max(epochs) if epochs else 0


def metrics_lookup(metrics_rows: list[dict]) -> dict[tuple[str, int], dict]:
    lookup = {}
    for row in metrics_rows:
        if not row.get("model") or not row.get("epoch"):
            continue
        lookup[(row["model"], int(float(row["epoch"])))] = row
    return lookup


def load_checkpoint_if_needed(model, checkpoint_path: Path, device: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()


def aggregate_structure_rows(rows: list[dict], batch_size: int, num_batches: int) -> list[dict]:
    output = []
    groups = sorted(
        {(row["model"], int(row["depth"]), str(row["epoch"]), int(row["seed"])) for row in rows},
        key=lambda item: (item[0], item[1], int(item[2]) if item[2].isdigit() else 10**9, item[3]),
    )
    for model, depth, epoch, seed in groups:
        group = [
            row
            for row in rows
            if row["model"] == model and int(row["depth"]) == depth and str(row["epoch"]) == epoch and int(row["seed"]) == seed
        ]
        output.append(
            {
                "model": model,
                "depth": depth,
                "epoch": epoch,
                "seed": seed,
                "batch_size": batch_size,
                "num_minibatches": num_batches,
                "gradient_type": "per_sample_input_gradient",
                "train_loss": group[0].get("train_loss", ""),
                "val_loss": group[0].get("val_loss", ""),
                "val_accuracy": group[0].get("val_accuracy", ""),
                "relative_effective_rank_mean": float(np.mean([float(row["relative_effective_rank"]) for row in group])),
                "relative_effective_rank_std": float(np.std([float(row["relative_effective_rank"]) for row in group])),
                "mean_gradient_norm_mean": float(np.mean([float(row["mean_gradient_norm"]) for row in group])),
                "mean_gradient_norm_std": float(np.std([float(row["mean_gradient_norm"]) for row in group])),
                "spatial_acf_lag1": float(np.mean([float(row["spatial_acf_lag1"]) for row in group])),
                "spatial_acf_lag3": float(np.mean([float(row["spatial_acf_lag3"]) for row in group])),
                "spatial_acf_lag5": float(np.mean([float(row["spatial_acf_lag5"]) for row in group])),
                "correlation_matrix_structure_score": float(np.mean([float(row["correlation_matrix_structure_score"]) for row in group])),
            }
        )
    return output


def plot_metric_vs_epoch(path: Path, rows: list[dict], value_key: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted({row["model"] for row in rows})
    for model in models:
        group = [row for row in rows if row["model"] == model]
        group.sort(key=lambda row: int(row["epoch"]) if str(row["epoch"]).isdigit() else 10**9)
        slug = model.lower().replace(" ", "_").replace("=", "")
        ax.errorbar(
            [int(row["epoch"]) if str(row["epoch"]).isdigit() else 0 for row in group],
            [float(row[f"{value_key}_mean"]) for row in group],
            yerr=[float(row[f"{value_key}_std"]) for row in group],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=model,
            color=PLOT_COLORS.get(slug),
        )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spatial_acf_evolution(path: Path, acf_rows: list[dict], model: str, max_lag: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rows = [row for row in acf_rows if row["model"] == model]
    epochs = sorted({str(row["epoch"]) for row in rows}, key=lambda value: int(value) if value.isdigit() else 10**9)
    for epoch in epochs:
        group = [row for row in rows if str(row["epoch"]) == epoch]
        means = [float(np.mean([float(row[f"lag_{lag}"]) for row in group])) for lag in range(max_lag + 1)]
        ax.plot(range(max_lag + 1), means, marker="o", linewidth=1.3, label=f"epoch {epoch}")
    ax.set_title(f"CIFAR-10 Checkpoint Spatial ACF {model}")
    ax.set_xlabel("spatial lag")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_accuracy_gradient_relation(path: Path, rows: list[dict]) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    models = sorted({row["model"] for row in rows})
    for model in models:
        group = [row for row in rows if row["model"] == model]
        group.sort(key=lambda row: int(row["epoch"]) if str(row["epoch"]).isdigit() else 10**9)
        epochs = [int(row["epoch"]) if str(row["epoch"]).isdigit() else 0 for row in group]
        accuracy = [float(row["val_accuracy"]) if row.get("val_accuracy", "") != "" else np.nan for row in group]
        rank = [float(row["relative_effective_rank_mean"]) for row in group]
        ax1.plot(epochs, accuracy, marker="o", linestyle="-", linewidth=1.4, label=f"{model} val acc")
        ax2.plot(epochs, rank, marker="s", linestyle="--", linewidth=1.2, label=f"{model} rel rank")
    ax1.set_title("CIFAR-10 Accuracy vs Gradient Structure")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy")
    ax2.set_ylabel("relative effective rank")
    ax1.grid(True, alpha=0.25)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=56)
    parser.add_argument("--checkpoints", nargs="+", default=["0", "final"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--data-root", default="data/cifar10_verified")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/full_training/checkpoints"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/full_training/tables/identity_full_training_metrics.csv"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-spatial-lag", type=int, default=15)
    parser.add_argument("--include-scaled-ablation", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/cifar10_checkpoint_gradient_evolution"))
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    paths = ensure_subdirs(args.output_root)
    specs = model_specs(args.include_scaled_ablation)
    dataset = load_cifar10_dataset(args.data_root, train=False)
    fixed_batches = {str(seed): make_fixed_batches(len(dataset), args.batch_size, args.num_batches, seed) for seed in args.seeds}
    write_json(
        paths["metadata"] / "fixed_minibatch_indices.json",
        {
            "dataset": "CIFAR10",
            "split": "test",
            "root": args.data_root,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "indices_by_seed": fixed_batches,
        },
    )
    metrics_rows = read_csv(args.metrics_csv)
    final_epoch = final_epoch_from_metrics(metrics_rows)
    metrics_by_model_epoch = metrics_lookup(metrics_rows)
    checkpoint_names = sort_epoch_names({normalize_epoch_name(item) for item in args.checkpoints})

    batch_rows: list[dict] = []
    acf_rows: list[dict] = []
    figures: list[Path] = []

    for seed in args.seeds:
        batches = fixed_batches[str(seed)]
        for spec in specs:
            model_label = cifar_model_label(spec, args.depth)
            for checkpoint_name in checkpoint_names:
                display_epoch = checkpoint_display_epoch(checkpoint_name, final_epoch=final_epoch)
                print_progress(f"cifar checkpoint seed={seed} epoch={checkpoint_name} model={model_label}")
                set_global_seed(seed)
                model = build_cifar_model(spec, args.depth).to(device)
                model.eval()
                checkpoint_path = ""
                if normalize_epoch_name(checkpoint_name) == "final":
                    checkpoint_path_obj = args.checkpoint_root / f"{model_label}.pt"
                    if not checkpoint_path_obj.exists():
                        print_progress(f"missing checkpoint, skipping {checkpoint_path_obj}")
                        continue
                    checkpoint_path = str(checkpoint_path_obj)
                    load_checkpoint_if_needed(model, checkpoint_path_obj, device)
                metric_row = metrics_by_model_epoch.get((model_label, display_epoch), {})

                losses = []
                accuracies = []
                raw_epoch = normalize_epoch_name(checkpoint_name)
                for batch_index, indices in enumerate(batches):
                    inputs, targets = dataset_batch(dataset, indices)
                    result = collect_input_gradient_batch(model, inputs, targets, device)
                    losses.append(float(result["loss"]))
                    accuracies.append(float(result["accuracy"]))
                    gradient_matrix = result["gradient_matrix"]
                    gradients = result["gradients"]
                    labels = targets.numpy()
                    correlation = cosine_correlation(gradient_matrix)
                    rer = relative_effective_rank(gradient_matrix, seed + display_epoch * 100 + batch_index)
                    mean_norm = mean_gradient_norm(gradient_matrix)
                    spatial_acf = spatial_acf_from_gradients(gradients, max_lag=args.max_spatial_lag)
                    structure_score = label_structure_score(correlation, labels)
                    raw_prefix = f"{spec.slug}_depth{args.depth}_epoch{raw_epoch}_seed{seed}_batch{batch_index}"
                    np.save(paths["raw"] / f"gradient_matrix_{raw_prefix}.npy", gradient_matrix)
                    np.save(paths["raw"] / f"correlation_matrix_{raw_prefix}.npy", correlation)
                    np.save(paths["raw"] / f"gradient_tensor_{raw_prefix}.npy", gradients)
                    if batch_index == 0 and seed == args.seeds[0] and spec.slug in {"plain", "resnet", "preact"} and raw_epoch in {"0", "final"}:
                        suffix = {"plain": "plain", "resnet": "resnet", "preact": "preact"}[spec.slug]
                        fig_path = paths["figures"] / f"fig_cifar10_ckpt_corr_{suffix}_epoch{raw_epoch}.png"
                        save_heatmap(fig_path, correlation, f"CIFAR-10 Checkpoint Correlation {spec.label} epoch {raw_epoch}", vmin=-1, vmax=1)
                        figures.append(fig_path)
                    val_loss = metric_row.get("val_loss", "")
                    val_accuracy = metric_row.get("val_accuracy", "")
                    train_loss = metric_row.get("train_loss", "")
                    if raw_epoch == "0":
                        val_loss = float(np.mean(losses))
                        val_accuracy = float(np.mean(accuracies))
                    batch_rows.append(
                        {
                            "model": spec.label,
                            "model_label": model_label,
                            "model_slug": spec.slug,
                            "depth": args.depth,
                            "epoch": str(display_epoch),
                            "checkpoint_name": raw_epoch,
                            "seed": seed,
                            "batch_index": batch_index,
                            "batch_size": len(indices),
                            "num_minibatches": len(batches),
                            "gradient_type": "per_sample_input_gradient",
                            "checkpoint_path": checkpoint_path,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                            "relative_effective_rank": rer,
                            "mean_gradient_norm": mean_norm,
                            "spatial_acf_lag1": float(spatial_acf[1]) if len(spatial_acf) > 1 else 0.0,
                            "spatial_acf_lag3": float(spatial_acf[3]) if len(spatial_acf) > 3 else 0.0,
                            "spatial_acf_lag5": float(spatial_acf[5]) if len(spatial_acf) > 5 else 0.0,
                            "correlation_matrix_structure_score": structure_score,
                            "fixed_batch_loss": float(result["loss"]),
                            "fixed_batch_accuracy": float(result["accuracy"]),
                        }
                    )
                    acf_row = {
                        "model": spec.label,
                        "model_label": model_label,
                        "model_slug": spec.slug,
                        "depth": args.depth,
                        "epoch": str(display_epoch),
                        "checkpoint_name": raw_epoch,
                        "seed": seed,
                        "batch_index": batch_index,
                    }
                    acf_row.update({f"lag_{lag}": float(spatial_acf[lag]) for lag in range(args.max_spatial_lag + 1)})
                    acf_rows.append(acf_row)
                if raw_epoch == "0" and losses:
                    overall_loss = float(np.mean(losses))
                    overall_accuracy = float(np.mean(accuracies))
                    for row in batch_rows:
                        if (
                            row["model_label"] == model_label
                            and row["epoch"] == str(display_epoch)
                            and int(row["seed"]) == seed
                        ):
                            row["val_loss"] = overall_loss
                            row["val_accuracy"] = overall_accuracy
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    summary_rows = aggregate_structure_rows(batch_rows, args.batch_size, args.num_batches)
    rank_rows = aggregate_rows(batch_rows, ["model", "epoch", "seed"], ["relative_effective_rank"])
    norm_rows = aggregate_rows(batch_rows, ["model", "epoch", "seed"], ["mean_gradient_norm"])

    table_summary = paths["tables"] / "table_cifar10_checkpoint_gradient_structure_summary.csv"
    table_rank = paths["tables"] / "table_cifar10_checkpoint_relative_effective_rank.csv"
    table_norm = paths["tables"] / "table_cifar10_checkpoint_mean_gradient_norm.csv"
    table_relation = paths["tables"] / "table_cifar10_checkpoint_accuracy_gradient_relation.csv"
    write_csv(table_summary, summary_rows)
    write_csv(table_rank, rank_rows)
    write_csv(table_norm, norm_rows)
    write_csv(table_relation, summary_rows)

    rank_fig = paths["figures"] / "fig_cifar10_ckpt_relative_effective_rank_vs_epoch.png"
    norm_fig = paths["figures"] / "fig_cifar10_ckpt_mean_gradient_norm_vs_epoch.png"
    relation_fig = paths["figures"] / "fig_cifar10_ckpt_accuracy_vs_gradient_structure.png"
    plot_metric_vs_epoch(rank_fig, rank_rows, "relative_effective_rank", "CIFAR-10 Checkpoint Relative Effective Rank vs Epoch", "relative effective rank")
    plot_metric_vs_epoch(norm_fig, norm_rows, "mean_gradient_norm", "CIFAR-10 Checkpoint Mean Gradient Norm vs Epoch", "mean gradient norm")
    plot_accuracy_gradient_relation(relation_fig, summary_rows)
    figures.extend([rank_fig, norm_fig, relation_fig])

    for suffix, label in [("plain", "PlainNet"), ("resnet", "Standard ResNet"), ("preact", "PreAct ResNet")]:
        fig_path = paths["figures"] / f"fig_cifar10_ckpt_spatial_acf_{suffix}.png"
        plot_spatial_acf_evolution(fig_path, acf_rows, label, args.max_spatial_lag)
        figures.append(fig_path)

    command = " ".join([Path(sys.executable).name, *sys.argv])
    write_run_metadata(
        paths["metadata"] / "run_metadata.json",
        command,
        {
            "experiment": "cifar10_checkpoint_gradient_evolution",
            "data_root": args.data_root,
            "checkpoint_root": str(args.checkpoint_root),
            "metrics_csv": str(args.metrics_csv),
            "depth": args.depth,
            "checkpoints": checkpoint_names,
            "final_epoch_from_metrics": final_epoch,
            "seeds": args.seeds,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "device": device,
            "models": [spec.__dict__ for spec in specs],
        },
    )
    append_run_docs(
        "CIFAR-10 Checkpoint Gradient Evolution",
        command,
        args.output_root,
        figures,
        [table_summary, table_rank, table_norm, table_relation],
        [
            "CIFAR-10 checkpoint analysis used fixed real CIFAR-10 test minibatches and per-sample input gradients.",
            "The run compares a fresh epoch 0 initialization with the existing full_training final checkpoints; intermediate epoch checkpoints were not present in the source checkpoint directory.",
            "Gradient analysis is read-only and does not update model parameters.",
        ],
        dataset="CIFAR10",
    )
    print_progress(f"completed CIFAR-10 checkpoint outputs under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

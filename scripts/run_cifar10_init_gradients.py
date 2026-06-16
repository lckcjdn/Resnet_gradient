"""Run CIFAR-10 input-gradient structure analysis at initialization."""

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
    print_progress,
    relative_effective_rank,
    save_heatmap,
    set_global_seed,
    spatial_acf_from_gradients,
    write_csv,
    write_json,
    write_run_metadata,
)


def aggregate_structure_rows(rows: list[dict], batch_size: int, num_batches: int) -> list[dict]:
    output = []
    groups = sorted({(row["model"], int(row["depth"]), int(row["seed"])) for row in rows}, key=lambda item: (item[0], item[1], item[2]))
    for model, depth, seed in groups:
        group = [row for row in rows if row["model"] == model and int(row["depth"]) == depth and int(row["seed"]) == seed]
        output.append(
            {
                "model": model,
                "depth": depth,
                "seed": seed,
                "batch_size": batch_size,
                "num_minibatches": num_batches,
                "gradient_type": "per_sample_input_gradient",
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


def plot_metric_vs_depth(path: Path, rows: list[dict], value_key: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted({row["model"] for row in rows})
    for model in models:
        group = [row for row in rows if row["model"] == model]
        group.sort(key=lambda row: int(row["depth"]))
        slug = model.lower().replace(" ", "_").replace("=", "")
        ax.errorbar(
            [int(row["depth"]) for row in group],
            [float(row[f"{value_key}_mean"]) for row in group],
            yerr=[float(row[f"{value_key}_std"]) for row in group],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=model,
            color=PLOT_COLORS.get(slug),
        )
    ax.set_title(title)
    ax.set_xlabel("depth")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spatial_acf(path: Path, acf_rows: list[dict], model: str, depth: int, max_lag: int) -> None:
    group = [row for row in acf_rows if row["model"] == model and int(row["depth"]) == depth]
    if not group:
        return
    means = []
    for lag in range(max_lag + 1):
        means.append(float(np.mean([float(row[f"lag_{lag}"]) for row in group])))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(max_lag + 1), means, marker="o", linewidth=1.5)
    ax.set_title(f"CIFAR-10 Init Spatial ACF {model} depth={depth}")
    ax.set_xlabel("spatial lag")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=[20, 56])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--data-root", default="data/cifar10_verified")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-spatial-lag", type=int, default=15)
    parser.add_argument("--include-scaled-ablation", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/cifar10_init_gradient_analysis"))
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

    batch_rows: list[dict] = []
    acf_rows: list[dict] = []
    figures: list[Path] = []
    max_depth = max(args.depths)

    for seed in args.seeds:
        batches = fixed_batches[str(seed)]
        for depth in args.depths:
            for spec in specs:
                print_progress(f"cifar init seed={seed} depth={depth} model={spec.label}")
                set_global_seed(seed)
                model = build_cifar_model(spec, depth).to(device)
                model.eval()
                for batch_index, indices in enumerate(batches):
                    inputs, targets = dataset_batch(dataset, indices)
                    result = collect_input_gradient_batch(model, inputs, targets, device)
                    gradient_matrix = result["gradient_matrix"]
                    gradients = result["gradients"]
                    labels = targets.numpy()
                    correlation = cosine_correlation(gradient_matrix)
                    rer = relative_effective_rank(gradient_matrix, seed + depth * 100 + batch_index)
                    mean_norm = mean_gradient_norm(gradient_matrix)
                    spatial_acf = spatial_acf_from_gradients(gradients, max_lag=args.max_spatial_lag)
                    structure_score = label_structure_score(correlation, labels)
                    raw_prefix = f"{spec.slug}_depth{depth}_seed{seed}_batch{batch_index}"
                    np.save(paths["raw"] / f"gradient_matrix_{raw_prefix}.npy", gradient_matrix)
                    np.save(paths["raw"] / f"correlation_matrix_{raw_prefix}.npy", correlation)
                    np.save(paths["raw"] / f"gradient_tensor_{raw_prefix}.npy", gradients)
                    if batch_index == 0 and seed == args.seeds[0] and depth == max_depth and spec.slug in {"plain", "resnet", "preact"}:
                        suffix = {"plain": "plain", "resnet": "resnet", "preact": "preact"}[spec.slug]
                        fig_path = paths["figures"] / f"fig_cifar10_init_corr_{suffix}{depth}.png"
                        save_heatmap(fig_path, correlation, f"CIFAR-10 Init Gradient Correlation {spec.label} depth={depth}", vmin=-1, vmax=1)
                        figures.append(fig_path)
                        order = np.argsort(labels)
                        sorted_corr = correlation[np.ix_(order, order)]
                        sorted_path = paths["figures"] / f"fig_cifar10_init_corr_label_sorted_{suffix}{depth}.png"
                        save_heatmap(sorted_path, sorted_corr, f"CIFAR-10 Label-sorted Correlation {spec.label} depth={depth}", vmin=-1, vmax=1)
                        figures.append(sorted_path)
                    batch_rows.append(
                        {
                            "model": spec.label,
                            "model_slug": spec.slug,
                            "depth": depth,
                            "seed": seed,
                            "batch_index": batch_index,
                            "batch_size": len(indices),
                            "num_minibatches": len(batches),
                            "gradient_type": "per_sample_input_gradient",
                            "relative_effective_rank": rer,
                            "mean_gradient_norm": mean_norm,
                            "spatial_acf_lag1": float(spatial_acf[1]) if len(spatial_acf) > 1 else 0.0,
                            "spatial_acf_lag3": float(spatial_acf[3]) if len(spatial_acf) > 3 else 0.0,
                            "spatial_acf_lag5": float(spatial_acf[5]) if len(spatial_acf) > 5 else 0.0,
                            "correlation_matrix_structure_score": structure_score,
                            "loss": float(result["loss"]),
                            "accuracy": float(result["accuracy"]),
                        }
                    )
                    acf_row = {
                        "model": spec.label,
                        "model_slug": spec.slug,
                        "depth": depth,
                        "seed": seed,
                        "batch_index": batch_index,
                    }
                    acf_row.update({f"lag_{lag}": float(spatial_acf[lag]) for lag in range(args.max_spatial_lag + 1)})
                    acf_rows.append(acf_row)
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    summary_rows = aggregate_structure_rows(batch_rows, args.batch_size, args.num_batches)
    rank_rows = aggregate_rows(batch_rows, ["model", "depth", "seed"], ["relative_effective_rank"])
    norm_rows = aggregate_rows(batch_rows, ["model", "depth", "seed"], ["mean_gradient_norm"])

    table_summary = paths["tables"] / "table_cifar10_init_gradient_structure_summary.csv"
    table_rank = paths["tables"] / "table_cifar10_init_relative_effective_rank.csv"
    table_norm = paths["tables"] / "table_cifar10_init_mean_gradient_norm.csv"
    table_acf = paths["tables"] / "table_cifar10_init_spatial_acf_summary.csv"
    write_csv(table_summary, summary_rows)
    write_csv(table_rank, rank_rows)
    write_csv(table_norm, norm_rows)
    write_csv(table_acf, acf_rows)

    rank_fig = paths["figures"] / "fig_cifar10_init_relative_effective_rank_vs_depth.png"
    norm_fig = paths["figures"] / "fig_cifar10_init_mean_gradient_norm_vs_depth.png"
    plot_metric_vs_depth(rank_fig, rank_rows, "relative_effective_rank", "CIFAR-10 Init Relative Effective Rank vs Depth", "relative effective rank")
    plot_metric_vs_depth(norm_fig, norm_rows, "mean_gradient_norm", "CIFAR-10 Init Mean Gradient Norm vs Depth", "mean gradient norm")
    figures.extend([rank_fig, norm_fig])

    for slug, suffix, label in [
        ("plain", "plain", "PlainNet"),
        ("resnet", "resnet", "Standard ResNet"),
        ("preact", "preact", "PreAct ResNet"),
    ]:
        fig_path = paths["figures"] / f"fig_cifar10_init_spatial_acf_{suffix}{max_depth}.png"
        plot_spatial_acf(fig_path, acf_rows, label, max_depth, args.max_spatial_lag)
        figures.append(fig_path)

    command = " ".join([Path(sys.executable).name, *sys.argv])
    write_run_metadata(
        paths["metadata"] / "run_metadata.json",
        command,
        {
            "experiment": "cifar10_init_gradient_analysis",
            "data_root": args.data_root,
            "depths": args.depths,
            "seeds": args.seeds,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "device": device,
            "models": [spec.__dict__ for spec in specs],
            "note": "No optimizer updates are performed.",
        },
    )
    append_run_docs(
        "CIFAR-10 Init Gradient Analysis",
        command,
        args.output_root,
        figures,
        [table_summary, table_rank, table_norm, table_acf],
        [
            "CIFAR-10 initialization analysis used real CIFAR-10 test samples and fixed minibatch indices.",
            "Gradient analysis was read-only: models were in eval mode and no optimizer step was called.",
            "This run used the runtime-limited setting from the plan: seed 0, depths 20 and 56, batch size 64, and 5 minibatches unless rerun with broader arguments.",
        ],
        dataset="CIFAR10",
    )
    print_progress(f"completed CIFAR-10 init outputs under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

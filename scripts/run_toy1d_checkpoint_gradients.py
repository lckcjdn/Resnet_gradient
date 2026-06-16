"""Run Toy 1D checkpoint gradient-evolution analysis."""

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
from torch import nn, optim

from scripts.gradient_structure_common import (
    PLOT_COLORS,
    acf_1d,
    append_run_docs,
    choose_device,
    covariance_from_vector,
    ensure_subdirs,
    gradient_summary,
    model_specs,
    normalize_epoch_name,
    print_progress,
    save_heatmap,
    save_line_plot,
    set_global_seed,
    sort_epoch_names,
    write_csv,
    write_run_metadata,
)
from scripts.run_toy1d_init_gradients import build_toy_model, input_gradient


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(3.0 * x) + 0.3 * torch.cos(7.0 * x)


def make_regression_data(train_points: int, val_points: int, device: str):
    x_train = torch.linspace(-2.0, 2.0, train_points, device=device).view(-1, 1)
    x_val = torch.linspace(-2.0, 2.0, val_points, device=device).view(-1, 1)
    return x_train, target_function(x_train), x_val, target_function(x_val)


def evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(x), y).item())


def save_checkpoint(path: Path, model: nn.Module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def analyze_checkpoint(
    *,
    model: nn.Module,
    spec,
    depth: int,
    epoch_name: str,
    seed: int,
    grid: torch.Tensor,
    device: str,
    max_lag: int,
    train_loss: float,
    val_loss: float,
    paths: dict[str, Path],
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    gradient = input_gradient(model, grid, device)
    covariance = covariance_from_vector(gradient)
    acf = acf_1d(gradient, max_lag)
    safe_epoch = normalize_epoch_name(epoch_name)
    np.save(paths["raw"] / f"gradient_vector_{spec.slug}_depth{depth}_epoch{safe_epoch}_seed{seed}.npy", gradient)
    np.save(paths["raw"] / f"gradient_matrix_{spec.slug}_depth{depth}_epoch{safe_epoch}_seed{seed}.npy", gradient.reshape(1, -1))
    np.save(paths["raw"] / f"correlation_matrix_{spec.slug}_depth{depth}_epoch{safe_epoch}_seed{seed}.npy", covariance)
    summary = gradient_summary(
        model=spec.label,
        depth=depth,
        seed=seed,
        gradient=gradient,
        acf=acf,
        covariance_matrix=covariance,
        extra={
            "epoch": safe_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "shortcut_lambda": spec.shortcut_lambda,
            "gradient_type": "df_dx",
        },
    )
    acf_row = {
        "model": spec.label,
        "depth": depth,
        "epoch": safe_epoch,
        "seed": seed,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    acf_row.update({f"lag_{lag}": float(acf[lag]) for lag in range(max_lag + 1)})
    return summary, acf_row, gradient, covariance, acf


def plot_gradient_evolution(root: Path, x_grid: np.ndarray, gradients: dict[tuple[str, str], np.ndarray], spec, epoch_names: list[str]) -> Path:
    path = root / "figures" / f"fig_toy1d_ckpt_gradient_evolution_{spec.slug}.png"
    series = []
    cmap = plt.get_cmap("viridis", len(epoch_names))
    for index, epoch_name in enumerate(epoch_names):
        key = (spec.slug, normalize_epoch_name(epoch_name))
        if key in gradients:
            series.append((f"epoch {normalize_epoch_name(epoch_name)}", gradients[key], cmap(index)))
    save_line_plot(path, x_grid, series, f"Toy 1D Gradient Evolution {spec.label}", "input x", "df/dx")
    return path


def plot_acf_evolution(root: Path, acf_rows: list[dict], spec, max_lag: int) -> Path:
    path = root / "figures" / f"fig_toy1d_ckpt_acf_evolution_{spec.slug}.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rows = [row for row in acf_rows if row["model"] == spec.label]
    for row in rows:
        values = [float(row[f"lag_{lag}"]) for lag in range(max_lag + 1)]
        ax.plot(range(max_lag + 1), values, marker="o", linewidth=1.2, label=f"epoch {row['epoch']}")
    ax.set_title(f"Toy 1D ACF Evolution {spec.label}")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_acf_score_vs_epoch(root: Path, summary_rows: list[dict]) -> Path:
    path = root / "figures" / "fig_toy1d_ckpt_acf_score_vs_epoch.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = sorted({row["model"] for row in summary_rows})
    for model in models:
        rows = [row for row in summary_rows if row["model"] == model]
        rows.sort(key=lambda row: int(row["epoch"]) if str(row["epoch"]).isdigit() else 10**9)
        ax.plot(
            [int(row["epoch"]) if str(row["epoch"]).isdigit() else max(0, len(rows) - 1) for row in rows],
            [float(row["acf_area_under_curve"]) for row in rows],
            marker="o",
            linewidth=1.5,
            label=model,
        )
    ax.set_title("Toy 1D ACF Area vs Epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ACF area under curve")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-epochs", nargs="+", default=["0", "1", "5", "10", "final"])
    parser.add_argument("--train-points", type=int, default=1024)
    parser.add_argument("--val-points", type=int, default=256)
    parser.add_argument("--grid-points", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--include-scaled-ablation", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/toy1d_checkpoint_gradient_evolution"))
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    paths = ensure_subdirs(args.output_root)
    specs = model_specs(args.include_scaled_ablation)
    requested_epochs = sort_epoch_names({normalize_epoch_name(item) for item in args.checkpoint_epochs})
    grid = torch.linspace(-2.0, 2.0, args.grid_points).view(-1, 1)
    x_grid = grid.reshape(-1).numpy()

    summary_rows: list[dict] = []
    acf_rows: list[dict] = []
    gradient_cache: dict[tuple[str, str], np.ndarray] = {}
    figures: list[Path] = []

    for seed in args.seeds:
        for spec in specs:
            print_progress(f"toy checkpoint seed={seed} depth={args.depth} model={spec.label}")
            set_global_seed(seed)
            model = build_toy_model(spec.family, args.depth, args.width, spec.shortcut_lambda).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()
            x_train, y_train, x_val, y_val = make_regression_data(args.train_points, args.val_points, device)

            def capture(epoch_name: str) -> None:
                train_loss = evaluate_loss(model, x_train, y_train)
                val_loss = evaluate_loss(model, x_val, y_val)
                save_checkpoint(
                    paths["checkpoints"] / f"{spec.slug}_depth{args.depth}_epoch{normalize_epoch_name(epoch_name)}_seed{seed}.pt",
                    model,
                    {
                        "model": spec.label,
                        "depth": args.depth,
                        "seed": seed,
                        "epoch": normalize_epoch_name(epoch_name),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                )
                summary, acf_row, gradient, covariance, _acf = analyze_checkpoint(
                    model=model,
                    spec=spec,
                    depth=args.depth,
                    epoch_name=epoch_name,
                    seed=seed,
                    grid=grid,
                    device=device,
                    max_lag=args.max_lag,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    paths=paths,
                )
                summary_rows.append(summary)
                acf_rows.append(acf_row)
                gradient_cache[(spec.slug, normalize_epoch_name(epoch_name))] = gradient
                if spec.slug in {"plain", "resnet", "preact"} and normalize_epoch_name(epoch_name) in {"0", "final"}:
                    cov_path = paths["figures"] / f"fig_toy1d_ckpt_cov_{spec.slug}_epoch{normalize_epoch_name(epoch_name)}.png"
                    save_heatmap(
                        cov_path,
                        covariance,
                        f"Toy 1D Checkpoint Covariance {spec.label} epoch {normalize_epoch_name(epoch_name)}",
                        "input index",
                        "input index",
                    )
                    figures.append(cov_path)

            if "0" in requested_epochs:
                capture("0")

            for epoch in range(1, args.epochs + 1):
                model.train()
                permutation = torch.randperm(args.train_points, device=device)
                for start in range(0, args.train_points, args.batch_size):
                    batch_index = permutation[start : start + args.batch_size]
                    inputs = x_train[batch_index]
                    targets = y_train[batch_index]
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
                if str(epoch) in requested_epochs:
                    capture(str(epoch))

            if "final" in requested_epochs and str(args.epochs) != "final":
                capture("final")

            figures.append(plot_gradient_evolution(args.output_root, x_grid, gradient_cache, spec, requested_epochs))
            figures.append(plot_acf_evolution(args.output_root, acf_rows, spec, args.max_lag))

    table_stats = paths["tables"] / "table_toy1d_checkpoint_gradient_stats.csv"
    table_acf = paths["tables"] / "table_toy1d_checkpoint_acf_summary.csv"
    write_csv(table_stats, summary_rows)
    write_csv(table_acf, acf_rows)
    figures.append(plot_acf_score_vs_epoch(args.output_root, summary_rows))

    command = " ".join([Path(sys.executable).name, *sys.argv])
    write_run_metadata(
        paths["metadata"] / "run_metadata.json",
        command,
        {
            "experiment": "toy1d_checkpoint_gradient_evolution",
            "depth": args.depth,
            "seeds": args.seeds,
            "width": args.width,
            "epochs": args.epochs,
            "checkpoint_epochs": requested_epochs,
            "device": device,
            "models": [spec.__dict__ for spec in specs],
        },
    )
    append_run_docs(
        "Toy 1D Checkpoint Gradient Evolution",
        command,
        args.output_root,
        figures,
        [table_stats, table_acf],
        [
            "Toy 1D checkpoint gradients were computed after saving epoch 0 before optimizer updates.",
            "Training uses a synthetic regression target y = sin(3x) + 0.3 cos(7x).",
            "Gradient analysis is read-only and uses eval mode for every saved checkpoint.",
        ],
        dataset="synthetic Toy 1D regression",
    )
    print_progress(f"completed toy checkpoint outputs under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

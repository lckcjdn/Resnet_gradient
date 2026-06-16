"""Run Toy 1D gradient-structure analysis at random initialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from scripts.gradient_structure_common import (
    PLOT_COLORS,
    acf_1d,
    append_run_docs,
    choose_device,
    covariance_from_vector,
    ensure_subdirs,
    gradient_summary,
    model_specs,
    print_progress,
    save_heatmap,
    save_line_plot,
    set_global_seed,
    write_csv,
    write_run_metadata,
)


class PlainMLP(nn.Module):
    def __init__(self, depth: int, width: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, shortcut_lambda: float = 1.0, preact: bool = False):
        super().__init__()
        self.shortcut_lambda = float(shortcut_lambda)
        self.preact = preact
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preact:
            residual = self.fc2(torch.relu(self.fc1(torch.relu(x))))
            return self.shortcut_lambda * x + residual
        residual = self.fc2(torch.relu(self.fc1(x)))
        return torch.relu(self.shortcut_lambda * x + residual)


class ResidualMLP(nn.Module):
    def __init__(self, depth: int, width: int, shortcut_lambda: float = 1.0, preact: bool = False):
        super().__init__()
        self.input = nn.Linear(1, width)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(width, shortcut_lambda=shortcut_lambda, preact=preact) for _ in range(depth)]
        )
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        if not any(getattr(block, "preact", False) for block in self.blocks):
            x = torch.relu(x)
        x = self.blocks(x)
        return self.output(torch.relu(x))


def build_toy_model(family: str, depth: int, width: int, shortcut_lambda: float) -> nn.Module:
    if family == "PlainNet":
        return PlainMLP(depth=depth, width=width)
    if family == "ResNetV1":
        return ResidualMLP(depth=depth, width=width, shortcut_lambda=1.0, preact=False)
    if family == "PreActResNet":
        return ResidualMLP(depth=depth, width=width, shortcut_lambda=1.0, preact=True)
    if family == "ScaledShortcutResNet":
        return ResidualMLP(depth=depth, width=width, shortcut_lambda=shortcut_lambda, preact=False)
    raise ValueError(f"Unsupported toy model family: {family}")


def input_gradient(model: nn.Module, x_values: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    x = x_values.to(device).detach().clone().requires_grad_(True)
    y = model(x).reshape(-1)
    grad = torch.autograd.grad(y.sum(), x, retain_graph=False, create_graph=False)[0]
    return grad.detach().cpu().reshape(-1).numpy()


def plot_gradient_curves(root: Path, x_grid: np.ndarray, gradients: dict[tuple[str, int, int], np.ndarray], depths: list[int], seeds: list[int], specs) -> list[Path]:
    figures = []
    seed = seeds[0]
    for depth in depths:
        series = []
        for spec in specs:
            key = (spec.slug, depth, seed)
            if key in gradients:
                series.append((spec.label, gradients[key], PLOT_COLORS.get(spec.slug, None)))
        path = root / "figures" / f"fig_toy1d_init_gradient_curve_depth{depth}.png"
        save_line_plot(
            path,
            x_grid,
            series,
            f"Toy 1D Init Gradient Curve depth={depth} seed={seed}",
            "input x",
            "df/dx",
        )
        figures.append(path)
    return figures


def plot_separated_gradient_curves(
    root: Path,
    x_grid: np.ndarray,
    gradients: dict[tuple[str, int, int], np.ndarray],
    depths: list[int],
    seeds: list[int],
) -> list[Path]:
    figures = []
    seed = seeds[0]
    for depth in depths:
        plain = gradients.get(("plain", depth, seed))
        if plain is not None:
            plain_path = root / "figures" / f"fig_toy1d_init_gradient_curve_plain_only_depth{depth}.png"
            save_line_plot(
                plain_path,
                x_grid,
                [("PlainNet", plain, PLOT_COLORS.get("plain", "#333333"))],
                f"Toy 1D Init PlainNet Gradient Curve depth={depth} seed={seed}",
                "input x",
                "df/dx",
            )
            figures.append(plain_path)

        residual_series = []
        resnet = gradients.get(("resnet", depth, seed))
        scaled = gradients.get(("scaled_l1p0", depth, seed))
        if resnet is not None and scaled is not None and np.max(np.abs(resnet - scaled)) <= 1e-12:
            residual_series.append(
                (
                    "Standard ResNet = ScaledShortcut lambda=1.0",
                    resnet,
                    PLOT_COLORS.get("resnet", "#1f77b4"),
                )
            )
        else:
            if resnet is not None:
                residual_series.append(("Standard ResNet", resnet, PLOT_COLORS.get("resnet", "#1f77b4")))
            if scaled is not None:
                residual_series.append(
                    (
                        "ScaledShortcut lambda=1.0",
                        scaled,
                        PLOT_COLORS.get("scaled_l1p0", "#ff7f0e"),
                    )
                )
        preact = gradients.get(("preact", depth, seed))
        if preact is not None:
            residual_series.append(("PreAct ResNet", preact, PLOT_COLORS.get("preact", "#2ca02c")))
        if residual_series:
            residual_path = root / "figures" / f"fig_toy1d_init_gradient_curve_residual_only_depth{depth}.png"
            save_line_plot(
                residual_path,
                x_grid,
                residual_series,
                f"Toy 1D Init Residual-family Gradient Curve depth={depth} seed={seed}",
                "input x",
                "df/dx",
            )
            figures.append(residual_path)
    return figures


def plot_acf_by_depth(root: Path, acf_rows: list[dict], max_lag: int) -> Path:
    path = root / "figures" / "fig_toy1d_init_acf_by_depth.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    depths = sorted({int(row["depth"]) for row in acf_rows})
    selected_depths = depths if len(depths) <= 3 else [depths[0], depths[len(depths) // 2], depths[-1]]
    for axis, depth in zip(axes, selected_depths):
        rows = [row for row in acf_rows if int(row["depth"]) == depth]
        for row in rows:
            values = [float(row[f"lag_{lag}"]) for lag in range(max_lag + 1)]
            axis.plot(range(max_lag + 1), values, marker="o", linewidth=1.2, label=row["model"])
        axis.set_title(f"depth={depth}")
        axis.set_xlabel("lag")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("ACF")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Toy 1D Init ACF by Depth")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_noise_reference(root: Path, points: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(points)
    brown = np.cumsum(rng.standard_normal(points))
    white = (white - white.mean()) / (white.std() + 1e-12)
    brown = (brown - brown.mean()) / (brown.std() + 1e-12)
    x_axis = np.linspace(-2.0, 2.0, points)
    path = root / "figures" / "fig_toy1d_noise_reference.png"
    save_line_plot(
        path,
        x_axis,
        [("white noise", white, "#333333"), ("brown noise", brown, "#1f77b4")],
        "Toy 1D Noise Reference",
        "input x",
        "normalized value",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 24, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--include-scaled-ablation", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/toy1d_init_gradient_analysis"))
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    paths = ensure_subdirs(args.output_root)
    specs = model_specs(args.include_scaled_ablation)
    x_values = torch.linspace(-2.0, 2.0, args.points).view(-1, 1)
    x_grid = x_values.reshape(-1).numpy()

    gradients: dict[tuple[str, int, int], np.ndarray] = {}
    summary_rows: list[dict] = []
    acf_rows: list[dict] = []

    for seed in args.seeds:
        for depth in args.depths:
            for spec in specs:
                print_progress(f"toy init seed={seed} depth={depth} model={spec.label}")
                set_global_seed(seed)
                model = build_toy_model(spec.family, depth, args.width, spec.shortcut_lambda).to(device)
                gradient = input_gradient(model, x_values, device)
                covariance = covariance_from_vector(gradient)
                acf = acf_1d(gradient, args.max_lag)
                key = (spec.slug, depth, seed)
                gradients[key] = gradient
                np.save(paths["raw"] / f"gradient_vector_{spec.slug}_depth{depth}_seed{seed}.npy", gradient)
                np.save(paths["raw"] / f"correlation_matrix_{spec.slug}_depth{depth}_seed{seed}.npy", covariance)
                np.save(paths["raw"] / f"gradient_matrix_{spec.slug}_depth{depth}_seed{seed}.npy", gradient.reshape(1, -1))
                summary_rows.append(
                    gradient_summary(
                        model=spec.label,
                        depth=depth,
                        seed=seed,
                        gradient=gradient,
                        acf=acf,
                        covariance_matrix=covariance,
                        extra={"shortcut_lambda": spec.shortcut_lambda, "training_status": "epoch0_init"},
                    )
                )
                acf_row = {"model": spec.label, "depth": depth, "seed": seed}
                acf_row.update({f"lag_{lag}": float(acf[lag]) for lag in range(args.max_lag + 1)})
                acf_rows.append(acf_row)

    table_stats = paths["tables"] / "table_toy1d_init_gradient_stats.csv"
    table_acf = paths["tables"] / "table_toy1d_init_acf_summary.csv"
    write_csv(table_stats, summary_rows)
    write_csv(table_acf, acf_rows)

    figures = plot_gradient_curves(args.output_root, x_grid, gradients, args.depths, args.seeds, specs)
    figures.extend(plot_separated_gradient_curves(args.output_root, x_grid, gradients, args.depths, args.seeds))
    heatmap_depth = 50 if 50 in args.depths else max(args.depths)
    for slug, filename, title in [
        ("plain", f"fig_toy1d_init_cov_plain_depth{heatmap_depth}.png", "PlainNet"),
        ("resnet", f"fig_toy1d_init_cov_resnet_depth{heatmap_depth}.png", "Standard ResNet"),
        ("preact", f"fig_toy1d_init_cov_preact_depth{heatmap_depth}.png", "PreAct ResNet"),
    ]:
        covariance = np.load(paths["raw"] / f"correlation_matrix_{slug}_depth{heatmap_depth}_seed{args.seeds[0]}.npy")
        fig_path = paths["figures"] / filename
        save_heatmap(fig_path, covariance, f"Toy 1D Init Covariance {title} depth={heatmap_depth}", "input index", "input index")
        figures.append(fig_path)
    figures.append(plot_acf_by_depth(args.output_root, acf_rows, args.max_lag))
    figures.append(plot_noise_reference(args.output_root, args.points, args.seeds[0]))

    command = " ".join([Path(sys.executable).name, *sys.argv])
    write_run_metadata(
        paths["metadata"] / "run_metadata.json",
        command,
        {
            "experiment": "toy1d_init_gradient_analysis",
            "depths": args.depths,
            "seeds": args.seeds,
            "width": args.width,
            "points": args.points,
            "device": device,
            "models": [spec.__dict__ for spec in specs],
        },
    )
    append_run_docs(
        "Toy 1D Init Gradient Analysis",
        command,
        args.output_root,
        figures,
        [table_stats, table_acf],
        [
            "Toy 1D initialization gradients were computed in eval mode with no optimizer updates.",
            "The run stores raw gradient vectors, one-row gradient matrices, covariance matrices, ACF values, and summary metrics.",
            "PlainNet-only and residual-family-only gradient-curve figures are generated because deep PlainNet gradients can be orders of magnitude smaller than residual-family gradients.",
            "Standard ResNet and ScaledShortcut lambda=1.0 are identical in this toy implementation, so separated residual figures combine them into one labeled curve when their raw gradients match exactly.",
            "Only seed 0 should be interpreted as a quick-run setting unless the script is rerun with more seeds.",
        ],
        dataset="synthetic Toy 1D grid",
    )
    print_progress(f"completed toy init outputs under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

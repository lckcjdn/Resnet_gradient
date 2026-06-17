"""Reproduce a Figure-2-style Toy 1D ACF comparison at initialization only."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.gradient_structure_common import (
    append_run_docs,
    choose_device,
    ensure_subdirs,
    print_progress,
    save_artifact_manifest,
    set_global_seed,
    write_csv,
    write_run_metadata,
)
from src.analysis.acf import autocorrelation_1d, normalized_noise, sequence_diagnostics
from src.models.toy1d_feedforward import Toy1DFeedForward
from src.models.toy1d_resnet import Toy1DResNet


DEPTH_COLORS = {
    2: "#1f77b4",
    4: "#ff7f0e",
    10: "#2ca02c",
    24: "#9467bd",
    50: "#d62728",
}
MODEL_COLORS = {
    "feedforward": "#333333",
    "resnet_beta_1.0": "#1f77b4",
    "resnet_beta_0.1": "#2ca02c",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 10, 24, 50])
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--x-min", type=float, default=-2.0)
    parser.add_argument("--x-max", type=float, default=2.0)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy1d_fig2_reproduction"))
    parser.add_argument("--no-mean-center-feedforward", action="store_true")
    parser.add_argument("--no-mean-center-resnet", action="store_true")
    parser.add_argument("--skip-docs", action="store_true")
    return parser.parse_args(argv)


def resolve_seeds(runs: int, seeds: list[int] | None) -> list[int]:
    if seeds is None:
        return list(range(runs))
    if len(seeds) != runs:
        print_progress(f"runs={runs} but {len(seeds)} seeds were provided; using the explicit seed list")
    return list(seeds)


def beta_label(beta: float | None) -> str:
    return "" if beta is None else f"{beta:.1f}"


def depth_label(depth: int | None) -> str:
    return "" if depth is None else str(depth)


def build_model(
    model_type: str,
    depth: int,
    width: int,
    beta: float | None,
    mean_center_feedforward: bool,
    mean_center_resnet: bool,
) -> nn.Module:
    if model_type == "feedforward":
        return Toy1DFeedForward(depth=depth, width=width, mean_center=mean_center_feedforward)
    if model_type == "resnet":
        if beta is None:
            raise ValueError("ResNet beta must be provided")
        return Toy1DResNet(depth=depth, width=width, beta=beta, mean_center=mean_center_resnet)
    raise ValueError(f"Unsupported model_type: {model_type}")


def input_gradient(model: nn.Module, x_values: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    x = x_values.to(device).detach().clone().requires_grad_(True)
    y = model(x).reshape(-1)
    gradient = torch.autograd.grad(y.sum(), x, retain_graph=False, create_graph=False)[0]
    return gradient.detach().cpu().reshape(-1).numpy()


def add_acf_rows(
    rows: list[dict],
    run_metrics: list[dict],
    *,
    model_type: str,
    beta: float | None,
    depth: int | None,
    seed: int,
    values: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    diagnostics = sequence_diagnostics(values)
    acf = autocorrelation_1d(values, max_lag)
    metric_row = {
        "model_type": model_type,
        "beta": beta_label(beta),
        "depth": depth_label(depth),
        "run_seed": seed,
        "gradient_norm": diagnostics["gradient_norm"],
        "gradient_std": diagnostics["gradient_std"],
        "collapse_flag": diagnostics["collapse_flag"],
    }
    run_metrics.append(metric_row)
    for lag, value in enumerate(acf):
        rows.append(
            {
                **metric_row,
                "lag": lag,
                "acf_value": float(value) if not math.isnan(float(value)) else np.nan,
            }
        )
    return acf


def aggregate_acf(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    group_cols = ["model_type", "beta", "depth", "lag"]
    for key, group in frame.groupby(group_cols, sort=False, dropna=False):
        model_type, beta, depth, lag = key
        valid = group[(~group["collapse_flag"]) & group["acf_value"].notna()]
        collapsed_runs = int(group.loc[group["collapse_flag"], "run_seed"].nunique())
        valid_runs = int(valid["run_seed"].nunique())
        output.append(
            {
                "model_type": model_type,
                "beta": beta,
                "depth": depth,
                "lag": int(lag),
                "acf_mean": float(valid["acf_value"].mean()) if valid_runs else np.nan,
                "acf_std": float(valid["acf_value"].std(ddof=0)) if valid_runs else np.nan,
                "valid_runs": valid_runs,
                "collapsed_runs": collapsed_runs,
            }
        )
    return output


def aggregate_collapse(run_metrics: list[dict]) -> list[dict]:
    frame = pd.DataFrame(run_metrics)
    output = []
    group_cols = ["model_type", "beta", "depth"]
    for key, group in frame.groupby(group_cols, sort=False, dropna=False):
        model_type, beta, depth = key
        valid = group[~group["collapse_flag"]]
        output.append(
            {
                "model_type": model_type,
                "beta": beta,
                "depth": depth,
                "gradient_norm_mean": float(group["gradient_norm"].mean()),
                "gradient_norm_std": float(group["gradient_norm"].std(ddof=0)),
                "gradient_std_mean": float(group["gradient_std"].mean()),
                "gradient_std_std": float(group["gradient_std"].std(ddof=0)),
                "collapsed_runs": int(group.loc[group["collapse_flag"], "run_seed"].nunique()),
                "valid_runs": int(valid["run_seed"].nunique()),
            }
        )
    return output


def lookup_series(mean_frame: pd.DataFrame, model_type: str, beta: str, depth: str, max_lag: int) -> np.ndarray:
    subset = mean_frame[
        (mean_frame["model_type"] == model_type)
        & (mean_frame["beta"] == beta)
        & (mean_frame["depth"] == depth)
    ].sort_values("lag")
    values = np.full(max_lag + 1, np.nan, dtype=np.float64)
    for row in subset.to_dict("records"):
        lag = int(row["lag"])
        if lag <= max_lag:
            values[lag] = float(row["acf_mean"])
    return values


def plot_main_acf(path: Path, mean_rows: list[dict], depths: Iterable[int], max_lag: int) -> None:
    frame = pd.DataFrame(mean_rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=True)
    panels = [
        ("Feedforward nets", "feedforward", "", axes[0, 0]),
        ("ResNets beta=1.0", "resnet", "1.0", axes[0, 1]),
        ("ResNets beta=0.1", "resnet", "0.1", axes[1, 0]),
    ]
    lags = np.arange(max_lag + 1)
    for title, model_type, beta, axis in panels:
        for depth in depths:
            values = lookup_series(frame, model_type, beta, str(depth), max_lag)
            if np.all(np.isnan(values)):
                continue
            axis.plot(lags, values, marker="o", linewidth=1.5, markersize=3.5, color=DEPTH_COLORS.get(depth), label=f"depth={depth}")
        axis.set_title(title)
        axis.set_xlabel("Lag")
        axis.set_ylabel("Autocorrelation")
        axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    noise_axis = axes[1, 1]
    for label, model_type, color in [
        ("White noise", "white_noise", "#555555"),
        ("Brown noise", "brown_noise", "#8c564b"),
    ]:
        values = lookup_series(frame, model_type, "", "", max_lag)
        noise_axis.plot(lags, values, marker="o", linewidth=1.7, markersize=3.5, color=color, label=label)
    noise_axis.set_title("White and Brown noise")
    noise_axis.set_xlabel("Lag")
    noise_axis.set_ylabel("Autocorrelation")
    noise_axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    noise_axis.grid(True, alpha=0.25)
    noise_axis.legend(fontsize=8)
    fig.suptitle("Toy 1D Initialization Gradient ACF")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gradient_examples(
    path: Path,
    x_grid: np.ndarray,
    examples: dict[str, np.ndarray],
) -> None:
    ordered = [
        ("Feedforward depth=50", "feedforward"),
        ("ResNet beta=1.0 depth=50", "resnet_beta_1.0"),
        ("ResNet beta=0.1 depth=50", "resnet_beta_0.1"),
        ("Brown noise", "brown_noise"),
        ("White noise", "white_noise"),
    ]
    fig, axes = plt.subplots(len(ordered), 1, figsize=(8.5, 8.5), sharex=True)
    for axis, (title, key) in zip(axes, ordered):
        values = examples[key]
        axis.plot(x_grid, values, linewidth=1.2, color=MODEL_COLORS.get(key, "#555555"))
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("df/dx" if "noise" not in key else "value")
        axis.grid(True, alpha=0.22)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_depth_diagnostic(path: Path, diagnostics: list[dict], value_key: str, ylabel: str) -> None:
    frame = pd.DataFrame(diagnostics)
    frame = frame[frame["model_type"].isin(["feedforward", "resnet"]) & (frame["depth"] != "")]
    series_specs = [
        ("Feedforward", "feedforward", "", "#333333"),
        ("ResNet beta=1.0", "resnet", "1.0", "#1f77b4"),
        ("ResNet beta=0.1", "resnet", "0.1", "#2ca02c"),
    ]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for label, model_type, beta, color in series_specs:
        subset = frame[(frame["model_type"] == model_type) & (frame["beta"] == beta)].copy()
        if subset.empty:
            continue
        subset["depth_int"] = subset["depth"].astype(int)
        subset = subset.sort_values("depth_int")
        axis.plot(subset["depth_int"], subset[value_key], marker="o", linewidth=1.5, color=color, label=label)
    axis.set_xlabel("Depth")
    axis.set_ylabel(ylabel)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def acf_value(mean_rows: list[dict], model_type: str, beta: str, depth: str, lag: int) -> float:
    for row in mean_rows:
        if (
            row["model_type"] == model_type
            and row["beta"] == beta
            and row["depth"] == depth
            and int(row["lag"]) == lag
        ):
            return float(row["acf_mean"]) if not pd.isna(row["acf_mean"]) else float("nan")
    return float("nan")


def write_experiment_doc(
    *,
    command: str,
    output_dir: Path,
    figures: list[Path],
    tables: list[Path],
    diagnostics: list[dict],
    mean_rows: list[dict],
    depths: list[int],
    seeds: list[int],
    width: int,
    num_points: int,
    x_min: float,
    x_max: float,
    max_lag: int,
    mean_center_feedforward: bool,
    mean_center_resnet: bool,
) -> list[str]:
    collapsed = [
        row
        for row in diagnostics
        if row["model_type"] not in {"white_noise", "brown_noise"} and int(row["collapsed_runs"]) > 0
    ]
    collapse_note = (
        "No model-depth setting had collapsed runs under the 1e-8 norm/std rule."
        if not collapsed
        else "Collapsed settings: "
        + "; ".join(
            f"{row['model_type']} beta={row['beta'] or 'n/a'} depth={row['depth']} "
            f"collapsed={row['collapsed_runs']}"
            for row in collapsed
        )
        + "."
    )
    depth_for_comparison = "50" if 50 in depths else str(max(depths))
    feed_lag5 = acf_value(mean_rows, "feedforward", "", depth_for_comparison, min(5, max_lag))
    res1_lag5 = acf_value(mean_rows, "resnet", "1.0", depth_for_comparison, min(5, max_lag))
    res01_lag5 = acf_value(mean_rows, "resnet", "0.1", depth_for_comparison, min(5, max_lag))
    feed_vs_res = (
        f"At depth {depth_for_comparison}, lag-{min(5, max_lag)} mean ACF was "
        f"feedforward={feed_lag5:.4g}, ResNet beta=1.0={res1_lag5:.4g}, "
        f"and ResNet beta=0.1={res01_lag5:.4g}."
    )
    beta_note = (
        "The beta=0.1 ResNet preserved stronger local ACF than beta=1.0 at the comparison depth."
        if not math.isnan(res01_lag5) and not math.isnan(res1_lag5) and res01_lag5 > res1_lag5
        else "The beta=0.1 ResNet did not exceed beta=1.0 on the lag-5 comparison metric in this run."
    )
    feed_note = (
        "The feedforward ACF decayed faster than the beta=1.0 ResNet at the comparison depth."
        if not math.isnan(feed_lag5) and not math.isnan(res1_lag5) and feed_lag5 < res1_lag5
        else "The feedforward lag-5 ACF did not fall below the beta=1.0 ResNet in this run, so that comparison should be read cautiously."
    )
    notes = [
        "This is an epoch-0 initialization-only analysis: no checkpoints, training loop, or optimizer step is used.",
        f"Twenty-run averaging used seeds {seeds[0]} to {seeds[-1]}." if len(seeds) == 20 else f"Averaging used {len(seeds)} explicit seed(s): {seeds}.",
        collapse_note,
        feed_vs_res,
        feed_note,
        beta_note,
        "The implementation is reproduction-inspired rather than exact: model layers use per-sample feature centering/normalization to keep df(x_i)/dx_i pointwise well-defined.",
    ]
    figure_lines = "\n".join(f"- `{path}`" for path in figures)
    table_lines = "\n".join(f"- `{path}`" for path in tables)
    doc = f"""# Toy 1D Figure 2 Reproduction

## Experiment Objective

This experiment reproduces a Figure-2-style autocorrelation-function comparison for Toy 1D input gradients. It compares feedforward ReLU MLPs, residual MLPs with beta=1.0, residual MLPs with beta=0.1, and white/Brown-noise references.

## Relation to Paper Figure 2

The target is inspired by Figure 2 from "The Shattered Gradients Problem: If ResNets are the answer, then what is the question?". The implementation is not an exact architectural reproduction; it keeps the same Toy 1D ACF question and reports initialization-only gradients with explicit collapse diagnostics.

## Initialization-Only Rule

No model is trained. The script initializes each model from scratch for every seed, computes `df/dx` on the fixed Toy 1D grid, and never calls `optimizer.step()`. Saved checkpoints are not loaded or used.

## Model Settings

- Input grid: `{num_points}` points on `[{x_min}, {x_max}]`.
- Depths: `{depths}`.
- Width: `{width}` hidden units.
- Seeds: `{seeds}`.
- Max lag: `{max_lag}`.
- Feedforward mean-centering: `{mean_center_feedforward}`.
- ResNet residual-branch feature centering/normalization: `{mean_center_resnet}`.
- Collapse rule: `gradient_norm < 1e-8` or `gradient_std < 1e-8`.

## ACF Formula

For gradient sequence `g`, let `z = g - mean(g)`. For lag `k`:

```text
ACF(k) = sum_i z_i * z_(i+k) / sum_i z_i^2
```

`ACF(0)` is fixed at 1. If the denominator is numerically near zero, the run is marked as `gradient_collapse` and nonzero-lag ACF values are left uninterpreted.

## Random Run Averaging

Each model-depth setting is run once per seed. Mean and standard deviation ACF curves are computed over valid non-collapsed runs only. Collapsed runs remain in the per-run summary and diagnostics tables.

## Generated Figures

{figure_lines}

## Generated Tables

{table_lines}

## Preliminary Interpretation

- {collapse_note}
- {feed_vs_res}
- {feed_note}
- {beta_note}
- Rapid ACF decay indicates weaker local gradient correlation; slow decay indicates stronger gradient structure.

## Limitations

- This is a reproduction-inspired analysis rather than an exact reproduction of every paper detail.
- The Toy 1D networks use per-sample feature centering/normalization so each grid point keeps a clear pointwise derivative.
- Results should be interpreted together with collapse diagnostics; near-zero gradients are labeled collapse rather than shattered gradients.
- The experiment does not evaluate training dynamics or checkpoint evolution.

## Command

```bash
{command}
```
"""
    (ROOT / "docs" / "toy1d_fig2_reproduction.md").write_text(doc, encoding="utf-8")
    append_run_docs(
        "Toy 1D Figure 2 Reproduction",
        command,
        output_dir,
        figures,
        tables,
        notes,
        dataset="synthetic Toy 1D grid",
    )
    return notes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = resolve_seeds(args.runs, args.seeds)
    device = choose_device(args.device)
    paths = ensure_subdirs(args.output_dir, names=("figures", "tables", "metadata"))
    mean_center_feedforward = not args.no_mean_center_feedforward
    mean_center_resnet = not args.no_mean_center_resnet

    x_values = torch.linspace(args.x_min, args.x_max, args.num_points).view(-1, 1)
    x_grid = x_values.reshape(-1).numpy()
    acf_rows: list[dict] = []
    run_metrics: list[dict] = []
    examples: dict[str, np.ndarray] = {}
    example_depth = 50 if 50 in args.depths else max(args.depths)
    example_seed = seeds[0]

    model_settings = [
        ("feedforward", None),
        ("resnet", 1.0),
        ("resnet", 0.1),
    ]
    for model_type, beta in model_settings:
        for depth in args.depths:
            for seed in seeds:
                print_progress(
                    f"toy fig2 init model={model_type} beta={beta_label(beta) or 'n/a'} depth={depth} seed={seed}"
                )
                set_global_seed(seed)
                model = build_model(
                    model_type,
                    depth,
                    args.width,
                    beta,
                    mean_center_feedforward,
                    mean_center_resnet,
                ).to(device)
                gradient = input_gradient(model, x_values, device)
                add_acf_rows(
                    acf_rows,
                    run_metrics,
                    model_type=model_type,
                    beta=beta,
                    depth=depth,
                    seed=seed,
                    values=gradient,
                    max_lag=args.max_lag,
                )
                if depth == example_depth and seed == example_seed:
                    key = "feedforward" if model_type == "feedforward" else f"resnet_beta_{beta:.1f}"
                    examples[key] = gradient

    for seed in seeds:
        for kind, model_type in [("white", "white_noise"), ("brown", "brown_noise")]:
            values = normalized_noise(kind, args.num_points, seed)
            add_acf_rows(
                acf_rows,
                run_metrics,
                model_type=model_type,
                beta=None,
                depth=None,
                seed=seed,
                values=values,
                max_lag=args.max_lag,
            )
            if seed == example_seed:
                examples[model_type] = values

    mean_rows = aggregate_acf(acf_rows)
    diagnostics = aggregate_collapse(run_metrics)

    table_acf_summary = paths["tables"] / "table_toy1d_acf_summary.csv"
    table_mean_std = paths["tables"] / "table_toy1d_acf_mean_std.csv"
    table_diagnostics = paths["tables"] / "table_toy1d_collapse_diagnostics.csv"
    write_csv(
        table_acf_summary,
        acf_rows,
        [
            "model_type",
            "beta",
            "depth",
            "run_seed",
            "lag",
            "acf_value",
            "gradient_norm",
            "gradient_std",
            "collapse_flag",
        ],
    )
    write_csv(
        table_mean_std,
        mean_rows,
        ["model_type", "beta", "depth", "lag", "acf_mean", "acf_std", "valid_runs", "collapsed_runs"],
    )
    write_csv(
        table_diagnostics,
        diagnostics,
        [
            "model_type",
            "beta",
            "depth",
            "gradient_norm_mean",
            "gradient_norm_std",
            "gradient_std_mean",
            "gradient_std_std",
            "collapsed_runs",
            "valid_runs",
        ],
    )

    fig_main = paths["figures"] / "fig_toy1d_acf_figure2_reproduction.png"
    fig_examples = paths["figures"] / "fig_toy1d_gradient_examples_depth50.png"
    fig_norm = paths["figures"] / "fig_toy1d_gradient_norm_by_depth.png"
    fig_std = paths["figures"] / "fig_toy1d_gradient_std_by_depth.png"
    plot_main_acf(fig_main, mean_rows, args.depths, args.max_lag)
    plot_gradient_examples(fig_examples, x_grid, examples)
    plot_depth_diagnostic(fig_norm, diagnostics, "gradient_norm_mean", "mean gradient L2 norm")
    plot_depth_diagnostic(fig_std, diagnostics, "gradient_std_mean", "mean gradient std")
    figures = [fig_main, fig_examples, fig_norm, fig_std]
    tables = [table_acf_summary, table_mean_std, table_diagnostics]

    command = " ".join([Path(sys.executable).name, *sys.argv])
    write_run_metadata(
        paths["metadata"] / "run_metadata.json",
        command,
        {
            "experiment": "toy1d_fig2_reproduction",
            "depths": args.depths,
            "width": args.width,
            "num_points": args.num_points,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "max_lag": args.max_lag,
            "runs_requested": args.runs,
            "seeds": seeds,
            "device": device,
            "mean_center_feedforward": mean_center_feedforward,
            "mean_center_resnet": mean_center_resnet,
            "initialization_only": True,
            "uses_checkpoints": False,
            "uses_optimizer_step": False,
        },
    )
    manifest = save_artifact_manifest(args.output_dir)
    tables.append(manifest)

    if not args.skip_docs:
        write_experiment_doc(
            command=command,
            output_dir=args.output_dir,
            figures=figures,
            tables=tables,
            diagnostics=diagnostics,
            mean_rows=mean_rows,
            depths=args.depths,
            seeds=seeds,
            width=args.width,
            num_points=args.num_points,
            x_min=args.x_min,
            x_max=args.x_max,
            max_lag=args.max_lag,
            mean_center_feedforward=mean_center_feedforward,
            mean_center_resnet=mean_center_resnet,
        )

    print_progress(f"completed Toy 1D Figure 2 reproduction under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

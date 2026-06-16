"""Run PlainNet initialization and normalization diagnostics on CIFAR-10."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.gradient_structure_common import (
    GRADIENT_COLLAPSE_THRESHOLD,
    acf_1d,
    append_run_docs,
    choose_device,
    cosine_correlation,
    dataset_batch,
    diagnostic_allows_shattering,
    ensure_subdirs,
    gradient_diagnostic_label,
    label_structure_score,
    load_cifar10_dataset,
    make_fixed_batches,
    print_progress,
    relative_effective_rank,
    save_artifact_manifest,
    save_heatmap,
    set_global_seed,
    spatial_acf_from_gradients,
    write_csv,
    write_json,
    write_run_metadata,
)


@dataclass(frozen=True)
class PlainNetAblationSpec:
    label: str
    slug: str
    init_scheme: str
    uses_bn: bool
    calibrate_bn: bool = False
    main_shattered_setting: bool = False


ABLATION_SPECS = [
    PlainNetAblationSpec(
        "PlainNet-DefaultInit",
        "plainnet_default_init",
        "default",
        uses_bn=True,
        calibrate_bn=False,
    ),
    PlainNetAblationSpec(
        "PlainNet-XavierInit",
        "plainnet_xavier_init",
        "xavier",
        uses_bn=False,
    ),
    PlainNetAblationSpec(
        "PlainNet-HeInit",
        "plainnet_he_init",
        "he",
        uses_bn=False,
    ),
    PlainNetAblationSpec(
        "PlainNet-HeInit-BN",
        "plainnet_he_init_bn",
        "he",
        uses_bn=True,
        calibrate_bn=True,
        main_shattered_setting=True,
    ),
    PlainNetAblationSpec(
        "PlainNet-OrthogonalInit",
        "plainnet_orthogonal_init",
        "orthogonal",
        uses_bn=False,
    ),
]


def blocks_per_stage(depth: int) -> int:
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR PlainNet depth should satisfy depth = 6n + 2.")
    return (depth - 2) // 6


class AblationPlainBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, uses_bn: bool):
        super().__init__()
        self.uses_bn = uses_bn
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not uses_bn)
        self.bn1 = nn.BatchNorm2d(out_channels) if uses_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not uses_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if uses_bn else nn.Identity()

    def forward(self, x: torch.Tensor, activations: list[tuple[str, torch.Tensor]] | None = None, prefix: str = "") -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        if activations is not None:
            activations.append((f"{prefix}.relu1", x))
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=False)
        if activations is not None:
            activations.append((f"{prefix}.relu2", x))
        return x


class AblationPlainNet(nn.Module):
    def __init__(self, depth: int = 56, num_classes: int = 10, uses_bn: bool = False):
        super().__init__()
        blocks = blocks_per_stage(depth)
        self.depth = depth
        self.num_classes = num_classes
        self.uses_bn = uses_bn
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=not uses_bn)
        self.bn1 = nn.BatchNorm2d(16) if uses_bn else nn.Identity()
        self.stage1 = self._make_stage(16, blocks, stride=1)
        self.stage2 = self._make_stage(32, blocks, stride=2)
        self.stage3 = self._make_stage(64, blocks, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.ModuleList:
        layers = nn.ModuleList([AblationPlainBlock(self.in_channels, out_channels, stride=stride, uses_bn=self.uses_bn)])
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(AblationPlainBlock(self.in_channels, out_channels, stride=1, uses_bn=self.uses_bn))
        return layers

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        activations: list[tuple[str, torch.Tensor]] | None = [] if return_activations else None
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        if activations is not None:
            activations.append(("stem.relu", x))
        for stage_index, stage in enumerate((self.stage1, self.stage2, self.stage3), start=1):
            for block_index, block in enumerate(stage):
                x = block(x, activations=activations, prefix=f"stage{stage_index}.{block_index}")
        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        if return_activations:
            return logits, activations
        return logits


def apply_init(model: nn.Module, scheme: str) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if scheme == "default":
                continue
            if scheme == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif scheme == "he":
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif scheme == "orthogonal":
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            else:
                raise ValueError(f"Unknown init scheme: {scheme}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            module.running_mean.zero_()
            module.running_var.fill_(1.0)


def build_variant(spec: PlainNetAblationSpec, depth: int) -> nn.Module:
    model = AblationPlainNet(depth=depth, num_classes=10, uses_bn=spec.uses_bn)
    apply_init(model, spec.init_scheme)
    return model


def calibrate_batch_norm(model: nn.Module, dataset, batches: list[list[int]], device: str) -> None:
    if not any(isinstance(module, nn.BatchNorm2d) for module in model.modules()):
        return
    model.train()
    with torch.no_grad():
        for indices in batches:
            inputs, _ = dataset_batch(dataset, indices)
            model(inputs.to(device))
    model.eval()


def analyze_batch(model: AblationPlainNet, inputs: torch.Tensor, targets: torch.Tensor, device: str):
    model.eval()
    inputs = inputs.to(device).detach().clone().requires_grad_(True)
    targets = targets.to(device)
    model.zero_grad(set_to_none=True)
    logits, activations = model(inputs, return_activations=True)
    losses = F.cross_entropy(logits, targets, reduction="none")
    input_grad = torch.autograd.grad(losses.sum(), inputs, retain_graph=True, create_graph=False)[0]
    model.zero_grad(set_to_none=True)
    losses.mean().backward()
    layer_grads = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not name.endswith("weight"):
            continue
        layer_grads.append(
            {
                "layer_name": name,
                "layerwise_gradient_norm": float(parameter.grad.detach().norm().item()),
                "layerwise_gradient_std": float(parameter.grad.detach().std(unbiased=False).item()),
            }
        )
    activation_rows = []
    for index, (name, activation) in enumerate(activations):
        values = activation.detach()
        activation_rows.append(
            {
                "layer_index": index,
                "layer_name": name,
                "activation_mean": float(values.mean().item()),
                "activation_std": float(values.std(unbiased=False).item()),
                "zero_activation_ratio": float((values == 0).float().mean().item()),
            }
        )
    return {
        "logits": logits.detach().cpu().numpy(),
        "input_gradients": input_grad.detach().cpu().numpy(),
        "gradient_matrix": input_grad.detach().flatten(1).cpu().numpy(),
        "activation_rows": activation_rows,
        "layer_gradient_rows": layer_grads,
        "loss": float(losses.mean().item()),
    }


def aggregate_layer_rows(rows: list[dict], keys: list[str], value_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key_values, group_rows in groups.items():
        out = {key: value for key, value in zip(keys, key_values)}
        for value_key in value_keys:
            values = [float(row[value_key]) for row in group_rows]
            out[value_key] = float(np.mean(values))
            out[f"{value_key}_std_over_batches"] = float(np.std(values))
        output.append(out)
    return output


def gradient_vector_acf(gradient_matrix: np.ndarray, max_lag: int) -> np.ndarray:
    sample_acfs = []
    for row in gradient_matrix:
        sample_acfs.append(acf_1d(row, max_lag))
    return np.mean(np.stack(sample_acfs, axis=0), axis=0)


def masked_cosine_correlation(gradient_matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    norms = np.linalg.norm(gradient_matrix.astype(np.float64), axis=1)
    valid = norms >= threshold
    matrix = np.full((gradient_matrix.shape[0], gradient_matrix.shape[0]), np.nan, dtype=np.float64)
    if np.any(valid):
        valid_matrix = gradient_matrix[valid]
        matrix[np.ix_(valid, valid)] = cosine_correlation(valid_matrix)
    return matrix, float(np.mean(valid))


def save_masked_heatmap(path: Path, matrix: np.ndarray, title: str, vmin: float = -1.0, vmax: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("sample index")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary_bars(path: Path, rows: list[dict], metric: str, ylabel: str, title: str, log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels = [row["variant"] for row in rows]
    values = [float(row[metric]) for row in rows]
    colors = ["#d62728" if row["diagnostic_label"] == "gradient_collapse" else "#1f77b4" for row in rows]
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(GRADIENT_COLLAPSE_THRESHOLD, color="#333333", linestyle="--", linewidth=1.0, label="collapse threshold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_layer_heatmap(path: Path, rows: list[dict], value_key: str, title: str, colorbar_label: str) -> None:
    variants = [spec.label for spec in ABLATION_SPECS]
    layer_names = sorted({row["layer_name"] for row in rows}, key=lambda name: rows[[r["layer_name"] for r in rows].index(name)].get("layer_index", 0))
    matrix = np.full((len(variants), len(layer_names)), np.nan)
    lookup = {(row["variant"], row["layer_name"]): float(row[value_key]) for row in rows}
    for i, variant in enumerate(variants):
        for j, layer_name in enumerate(layer_names):
            matrix[i, j] = lookup.get((variant, layer_name), np.nan)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("activation/parameter layer")
    ax.set_ylabel("variant")
    tick_step = max(1, len(layer_names) // 12)
    ax.set_xticks(range(0, len(layer_names), tick_step))
    ax.set_xticklabels([layer_names[i] for i in range(0, len(layer_names), tick_step)], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=8)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_acf(path: Path, rows: list[dict], max_lag: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for row in rows:
        values = [float(row[f"acf_lag{lag}"]) for lag in range(max_lag + 1)]
        linestyle = "--" if row["diagnostic_label"] == "gradient_collapse" else "-"
        ax.plot(range(max_lag + 1), values, marker="o", linewidth=1.3, linestyle=linestyle, label=row["variant"])
    ax.set_title("PlainNet Init Gradient ACF")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=56)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--data-root", default="data/cifar10_verified")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--output-root", type=Path, default=Path("results/plainnet_init_ablation"))
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    paths = ensure_subdirs(args.output_root, ("figures", "tables", "raw", "metadata"))
    collapse_root = args.output_root / "vanishing_collapse_analysis"
    shattered_root = args.output_root / "shattered_gradient_analysis"
    collapse_paths = ensure_subdirs(collapse_root, ("figures", "tables", "raw"))
    shattered_paths = ensure_subdirs(shattered_root, ("figures", "tables", "raw"))

    dataset = load_cifar10_dataset(args.data_root, train=False)
    fixed_batches = make_fixed_batches(len(dataset), args.batch_size, args.num_batches, args.seed)
    write_json(
        paths["metadata"] / "fixed_minibatch_indices.json",
        {
            "dataset": "CIFAR10",
            "split": "test",
            "root": args.data_root,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "indices": fixed_batches,
        },
    )

    summary_rows: list[dict] = []
    activation_rows: list[dict] = []
    layer_gradient_rows: list[dict] = []
    output_rows: list[dict] = []
    acf_rows: list[dict] = []
    figures: list[Path] = []

    for spec in ABLATION_SPECS:
        print_progress(f"plainnet ablation seed={args.seed} depth={args.depth} variant={spec.label}")
        set_global_seed(args.seed)
        model = build_variant(spec, args.depth).to(device)
        if spec.calibrate_bn:
            calibrate_batch_norm(model, dataset, fixed_batches, device)
        model.eval()

        gradient_matrices = []
        gradient_tensors = []
        logits_all = []
        targets_all = []
        losses = []
        for batch_index, indices in enumerate(fixed_batches):
            inputs, targets = dataset_batch(dataset, indices)
            result = analyze_batch(model, inputs, targets, device)
            gradient_matrices.append(result["gradient_matrix"])
            gradient_tensors.append(result["input_gradients"])
            logits_all.append(result["logits"])
            targets_all.append(targets.numpy())
            losses.append(float(result["loss"]))
            for row in result["activation_rows"]:
                activation_rows.append(
                    {
                        "variant": spec.label,
                        "variant_slug": spec.slug,
                        "batch_index": batch_index,
                        **row,
                    }
                )
            for row in result["layer_gradient_rows"]:
                layer_gradient_rows.append(
                    {
                        "variant": spec.label,
                        "variant_slug": spec.slug,
                        "batch_index": batch_index,
                        **row,
                    }
                )

        gradient_matrix = np.concatenate(gradient_matrices, axis=0)
        gradient_tensor = np.concatenate(gradient_tensors, axis=0)
        logits = np.concatenate(logits_all, axis=0)
        labels = np.concatenate(targets_all, axis=0)
        sample_norms = np.linalg.norm(gradient_matrix.astype(np.float64), axis=1)
        gradient_std = float(np.std(gradient_matrix))
        input_gradient_norm_mean = float(np.mean(sample_norms))
        input_gradient_norm_std = float(np.std(sample_norms))
        diagnostic_label = gradient_diagnostic_label(input_gradient_norm_mean, gradient_std)
        correlation, valid_correlation_fraction = masked_cosine_correlation(gradient_matrix, GRADIENT_COLLAPSE_THRESHOLD)
        centered_gradient_matrix = gradient_matrix - np.mean(gradient_matrix, axis=1, keepdims=True)
        centered_correlation, centered_valid_fraction = masked_cosine_correlation(centered_gradient_matrix, GRADIENT_COLLAPSE_THRESHOLD)
        spatial_acf = spatial_acf_from_gradients(gradient_tensor, max_lag=args.max_lag)
        vector_acf = gradient_vector_acf(gradient_matrix, args.max_lag)
        rer = relative_effective_rank(gradient_matrix, args.seed)
        structure_score = label_structure_score(np.nan_to_num(correlation, nan=0.0), labels)

        np.save(paths["raw"] / f"gradient_matrix_{spec.slug}.npy", gradient_matrix)
        np.save(paths["raw"] / f"gradient_tensor_{spec.slug}.npy", gradient_tensor)
        np.save(paths["raw"] / f"correlation_matrix_{spec.slug}.npy", correlation)
        np.save(paths["raw"] / f"centered_correlation_matrix_{spec.slug}.npy", centered_correlation)
        target_paths = shattered_paths if diagnostic_allows_shattering(diagnostic_label) else collapse_paths
        np.save(target_paths["raw"] / f"correlation_matrix_{spec.slug}.npy", correlation)
        np.save(target_paths["raw"] / f"gradient_matrix_{spec.slug}.npy", gradient_matrix)

        corr_path = target_paths["figures"] / f"fig_gradient_correlation_{spec.slug}.png"
        save_masked_heatmap(
            corr_path,
            correlation,
            f"{spec.label} correlation ({diagnostic_label})",
        )
        figures.append(corr_path)
        centered_corr_path = target_paths["figures"] / f"fig_centered_gradient_correlation_{spec.slug}.png"
        save_masked_heatmap(
            centered_corr_path,
            centered_correlation,
            f"{spec.label} centered correlation ({diagnostic_label})",
        )
        figures.append(centered_corr_path)

        summary_rows.append(
            {
                "variant": spec.label,
                "variant_slug": spec.slug,
                "depth": args.depth,
                "seed": args.seed,
                "init_scheme": spec.init_scheme,
                "uses_bn": spec.uses_bn,
                "bn_calibrated": spec.calibrate_bn,
                "main_shattered_plainnet_setting": spec.main_shattered_setting,
                "output_mean": float(np.mean(logits)),
                "output_std": float(np.std(logits)),
                "loss": float(np.mean(losses)),
                "input_gradient_norm_mean": input_gradient_norm_mean,
                "input_gradient_norm_std": input_gradient_norm_std,
                "input_gradient_norm_min": float(np.min(sample_norms)),
                "input_gradient_norm_max": float(np.max(sample_norms)),
                "gradient_mean": float(np.mean(gradient_matrix)),
                "gradient_std": gradient_std,
                "relative_effective_rank": rer,
                "spatial_acf_lag1": float(spatial_acf[1]) if len(spatial_acf) > 1 else 0.0,
                "spatial_acf_lag3": float(spatial_acf[3]) if len(spatial_acf) > 3 else 0.0,
                "spatial_acf_lag5": float(spatial_acf[5]) if len(spatial_acf) > 5 else 0.0,
                "vector_acf_lag1": float(vector_acf[1]) if len(vector_acf) > 1 else 0.0,
                "vector_acf_lag5": float(vector_acf[5]) if len(vector_acf) > 5 else 0.0,
                "vector_acf_lag10": float(vector_acf[10]) if len(vector_acf) > 10 else 0.0,
                "correlation_matrix_structure_score": structure_score,
                "valid_correlation_fraction": valid_correlation_fraction,
                "centered_valid_correlation_fraction": centered_valid_fraction,
                "diagnostic_threshold": GRADIENT_COLLAPSE_THRESHOLD,
                "diagnostic_label": diagnostic_label,
                "interpretation_bucket": "shattered-gradient analysis" if diagnostic_allows_shattering(diagnostic_label) else "vanishing/collapse analysis",
            }
        )
        acf_row = {"variant": spec.label, "variant_slug": spec.slug, "diagnostic_label": diagnostic_label}
        acf_row.update({f"acf_lag{lag}": float(spatial_acf[lag]) for lag in range(args.max_lag + 1)})
        acf_rows.append(acf_row)
        output_rows.append(
            {
                "variant": spec.label,
                "output_mean": float(np.mean(logits)),
                "output_std": float(np.std(logits)),
            }
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    activation_aggregate = aggregate_layer_rows(
        activation_rows,
        ["variant", "variant_slug", "layer_index", "layer_name"],
        ["activation_mean", "activation_std", "zero_activation_ratio"],
    )
    layer_gradient_aggregate = aggregate_layer_rows(
        layer_gradient_rows,
        ["variant", "variant_slug", "layer_name"],
        ["layerwise_gradient_norm", "layerwise_gradient_std"],
    )
    collapse_rows = [row for row in summary_rows if row["diagnostic_label"] == "gradient_collapse"]
    shattered_rows = [row for row in summary_rows if row["diagnostic_label"] != "gradient_collapse"]
    main_rows = [row for row in summary_rows if row["main_shattered_plainnet_setting"]]

    table_summary = paths["tables"] / "table_plainnet_init_ablation_summary.csv"
    table_activation = paths["tables"] / "table_plainnet_init_ablation_activation_stats.csv"
    table_layer_grad = paths["tables"] / "table_plainnet_init_ablation_layerwise_gradient_norms.csv"
    table_acf = paths["tables"] / "table_plainnet_init_ablation_acf.csv"
    table_output = paths["tables"] / "table_plainnet_init_ablation_output_stats.csv"
    write_csv(table_summary, summary_rows)
    write_csv(table_activation, activation_aggregate)
    write_csv(table_layer_grad, layer_gradient_aggregate)
    write_csv(table_acf, acf_rows)
    write_csv(table_output, output_rows)

    table_collapse = collapse_paths["tables"] / "table_plainnet_gradient_collapse_summary.csv"
    table_shattered = shattered_paths["tables"] / "table_plainnet_shattered_gradient_candidates.csv"
    table_main = shattered_paths["tables"] / "table_main_plainnet_shattered_setting.csv"
    write_csv(table_collapse, collapse_rows)
    write_csv(table_shattered, shattered_rows)
    write_csv(table_main, main_rows)

    grad_norm_fig = paths["figures"] / "fig_plainnet_ablation_input_gradient_norm.png"
    grad_std_fig = paths["figures"] / "fig_plainnet_ablation_gradient_std.png"
    rank_fig = paths["figures"] / "fig_plainnet_ablation_relative_effective_rank.png"
    output_fig = paths["figures"] / "fig_plainnet_ablation_output_std.png"
    zero_fig = paths["figures"] / "fig_plainnet_ablation_activation_zero_ratio.png"
    layer_grad_fig = paths["figures"] / "fig_plainnet_ablation_layerwise_gradient_norm.png"
    acf_fig = paths["figures"] / "fig_plainnet_ablation_spatial_acf.png"
    plot_summary_bars(grad_norm_fig, summary_rows, "input_gradient_norm_mean", "mean sample input-gradient norm", "PlainNet Init Input Gradient Norm", log_scale=True)
    plot_summary_bars(grad_std_fig, summary_rows, "gradient_std", "input-gradient std", "PlainNet Init Gradient Std", log_scale=True)
    plot_summary_bars(rank_fig, summary_rows, "relative_effective_rank", "relative effective rank", "PlainNet Init Relative Effective Rank")
    plot_summary_bars(output_fig, summary_rows, "output_std", "output std", "PlainNet Init Output Std")
    plot_layer_heatmap(zero_fig, activation_aggregate, "zero_activation_ratio", "PlainNet Init Zero Activation Ratio", "zero activation ratio")
    plot_layer_heatmap(layer_grad_fig, layer_gradient_aggregate, "layerwise_gradient_norm", "PlainNet Init Layer-wise Gradient Norm", "gradient norm")
    plot_acf(acf_fig, acf_rows, args.max_lag)
    figures.extend([grad_norm_fig, grad_std_fig, rank_fig, output_fig, zero_fig, layer_grad_fig, acf_fig])

    for root in (args.output_root, collapse_root, shattered_root):
        save_artifact_manifest(root)

    command = " ".join([Path(sys.executable).name, *sys.argv])
    write_run_metadata(
        paths["metadata"] / "run_metadata.json",
        command,
        {
            "experiment": "plainnet_init_ablation",
            "data_root": args.data_root,
            "depth": args.depth,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "device": device,
            "diagnostic_rule": (
                "if input_gradient_norm_mean < 1e-8 or gradient_std < 1e-8, "
                "label as gradient_collapse rather than shattered_gradient_candidate"
            ),
            "variants": [spec.__dict__ for spec in ABLATION_SPECS],
        },
    )
    write_json(
        args.output_root / "plainnet_init_ablation_summary.json",
        {
            "summary_table": str(table_summary),
            "collapse_table": str(table_collapse),
            "shattered_table": str(table_shattered),
            "main_plainnet_setting": "PlainNet-HeInit-BN",
            "diagnostic_rule": "mean gradient norm < 1e-8 or gradient std < 1e-8 => gradient_collapse",
            "collapse_variants": [row["variant"] for row in collapse_rows],
            "shattered_candidate_variants": [row["variant"] for row in shattered_rows],
        },
    )
    append_run_docs(
        "PlainNet Init Ablation Diagnostics",
        command,
        args.output_root,
        figures,
        [
            table_summary,
            table_activation,
            table_layer_grad,
            table_acf,
            table_output,
            table_collapse,
            table_shattered,
            table_main,
        ],
        [
            "PlainNet initialization/normalization variants were diagnosed before making any shattered-gradient claim.",
            "The diagnostic rule is explicit: if mean sample input-gradient norm < 1e-8 or input-gradient std < 1e-8, the result is labeled gradient_collapse.",
            "Only non-collapsed variants are placed in the shattered-gradient candidate table; collapsed variants are separated under vanishing_collapse_analysis.",
            "PlainNet-HeInit-BN is recorded as the main PlainNet setting for future shattered-gradient comparisons with ResNet.",
        ],
        dataset="CIFAR10",
    )
    print_progress(f"completed PlainNet init ablation under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

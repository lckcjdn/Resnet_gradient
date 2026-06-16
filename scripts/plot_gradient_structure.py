"""Index and validate generated gradient-structure artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gradient_structure_common import save_artifact_manifest, write_csv


EXPECTED_OUTPUTS = {
    "toy1d_init_gradient_analysis": [
        "figures/fig_toy1d_init_gradient_curve_depth50.png",
        "figures/fig_toy1d_init_gradient_curve_plain_only_depth50.png",
        "figures/fig_toy1d_init_gradient_curve_residual_only_depth50.png",
        "figures/fig_toy1d_init_cov_plain_depth50.png",
        "figures/fig_toy1d_init_acf_by_depth.png",
        "tables/table_toy1d_init_gradient_stats.csv",
        "tables/table_toy1d_init_acf_summary.csv",
    ],
    "toy1d_checkpoint_gradient_evolution": [
        "figures/fig_toy1d_ckpt_gradient_evolution_plain.png",
        "figures/fig_toy1d_ckpt_cov_plain_epoch0.png",
        "figures/fig_toy1d_ckpt_acf_score_vs_epoch.png",
        "tables/table_toy1d_checkpoint_gradient_stats.csv",
        "tables/table_toy1d_checkpoint_acf_summary.csv",
    ],
    "cifar10_init_gradient_analysis": [
        "figures/fig_cifar10_init_corr_label_sorted_plain56.png",
        "figures/fig_cifar10_init_relative_effective_rank_vs_depth.png",
        "figures/fig_cifar10_init_mean_gradient_norm_vs_depth.png",
        "tables/table_cifar10_init_gradient_structure_summary.csv",
        "tables/table_cifar10_init_relative_effective_rank.csv",
    ],
    "cifar10_checkpoint_gradient_evolution": [
        "figures/fig_cifar10_ckpt_corr_plain_epoch0.png",
        "figures/fig_cifar10_ckpt_corr_plain_epochfinal.png",
        "figures/fig_cifar10_ckpt_relative_effective_rank_vs_epoch.png",
        "figures/fig_cifar10_ckpt_accuracy_vs_gradient_structure.png",
        "tables/table_cifar10_checkpoint_gradient_structure_summary.csv",
        "tables/table_cifar10_checkpoint_accuracy_gradient_relation.csv",
    ],
    "plainnet_init_ablation": [
        "README.md",
        "figures/fig_plainnet_ablation_input_gradient_norm.png",
        "figures/fig_plainnet_ablation_activation_zero_ratio.png",
        "tables/table_plainnet_init_ablation_summary.csv",
        "tables/table_plainnet_init_ablation_activation_stats.csv",
        "vanishing_collapse_analysis/tables/table_plainnet_gradient_collapse_summary.csv",
        "shattered_gradient_analysis/tables/table_plainnet_shattered_gradient_candidates.csv",
        "shattered_gradient_analysis/tables/table_main_plainnet_shattered_setting.csv",
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/gradient_structure_artifact_index.csv"))
    args = parser.parse_args(argv)

    rows = []
    missing = []
    for folder, expected_paths in EXPECTED_OUTPUTS.items():
        root = args.results_root / folder
        if root.exists():
            save_artifact_manifest(root)
        for relative in expected_paths:
            path = root / relative
            exists = path.exists()
            rows.append(
                {
                    "experiment": folder,
                    "expected_artifact": str(path),
                    "exists": exists,
                    "bytes": path.stat().st_size if exists else 0,
                }
            )
            if not exists:
                missing.append(path)
    write_csv(args.output, rows, ["experiment", "expected_artifact", "exists", "bytes"])
    if missing:
        print("Missing expected artifacts:")
        for path in missing:
            print(path)
        return 1
    print(f"Artifact index written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

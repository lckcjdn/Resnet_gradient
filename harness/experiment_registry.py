"""Experiment registry for planned runs and outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config: str
    expected_figures: tuple[str, ...]
    expected_tables: tuple[str, ...]
    status: str = "planned"


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "plain_resnet_preact": ExperimentSpec(
        name="PlainNet vs ResNet vs PreAct ResNet",
        config="configs/preact_resnet56.yaml",
        expected_figures=(
            "fig01_training_loss_plain_resnet_preact.png",
            "fig02_test_accuracy_plain_resnet_preact.png",
            "fig03_layerwise_gradient_norm_epoch_last.png",
            "fig04_gradient_heatmap_plain_resnet_preact.png",
        ),
        expected_tables=("table01_final_performance.csv", "table02_gradient_stability.csv"),
    ),
    "lambda_ablation": ExperimentSpec(
        name="Scaled shortcut lambda ablation",
        config="configs/scaled_lambda_10.yaml",
        expected_figures=(
            "fig05_lambda_ablation_loss.png",
            "fig06_lambda_ablation_gradient_stability.png",
            "fig07_lambda_ablation_accuracy.png",
        ),
        expected_tables=("table03_lambda_ablation.csv",),
    ),
    "lesion": ExperimentSpec(
        name="Residual branch lesion study",
        config="configs/lesion.yaml",
        expected_figures=(
            "fig08_lesion_accuracy_curve.png",
            "fig09_lesion_loss_curve.png",
            "fig10_lesion_sensitivity_by_block.png",
        ),
        expected_tables=("table04_lesion_summary.csv",),
    ),
}


def list_experiments() -> list[ExperimentSpec]:
    return list(EXPERIMENTS.values())

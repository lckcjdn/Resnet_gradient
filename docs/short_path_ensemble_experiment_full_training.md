# Short-path Ensemble Lesion Experiment

## Objective

Evaluate whether a trained ResNet remains partially functional when selected residual branches are removed.

## Theory Background

Residual blocks contain a shortcut path and a residual branch. Lesion masks evaluate `y = x + m_l * F(x)`, where `m_l = 0` disables a residual branch while preserving the shortcut.

## Lesion Method

- Checkpoint used: `results\full_training\checkpoints\PreActResNet-56.pt`
- Drop ratios: 0%, 10%, 30%, 50%, 70%
- Random Drop uses multiple seeds when requested.
- Early Drop removes branches near the input side.
- Late Drop removes branches near the output side.

## Generated Figures

- `results\full_training\figures\fig_lesion_accuracy_vs_drop_ratio.png`
- `results\full_training\figures\fig_lesion_accuracy_drop_vs_drop_ratio.png`
- `results\full_training\figures\fig_lesion_random_vs_early_vs_late.png`
- `results\full_training\figures\fig_active_blocks_vs_accuracy.png`
- `results\full_training\figures\fig_lesion_heatmap.png`

## Generated Tables

- `results\full_training\tables\table_lesion_accuracy.csv`
- `results\full_training\tables\table_lesion_drop_strategy_comparison.csv`
- `results\full_training\tables\table_active_blocks_summary.csv`

## Conceptual PlainNet Comparison

PlainNet cannot remove intermediate layers without breaking the only forward path. In contrast, ResNet can disable residual branches while shortcuts still preserve an information path.

## Preliminary Interpretation

The generated tables and figures show the mechanics of residual branch lesion evaluation. If accuracy declines gradually rather than failing immediately, that behavior is consistent with short-path ensemble interpretations.

## Limitations

- This run used real CIFAR-10 validation samples, but the checkpoint was still trained with a lightweight budget.
- Accuracy values should not be used as final CIFAR-10 evidence unless rerun with real CIFAR-10 and longer training.
- Results support cautious interpretation only.

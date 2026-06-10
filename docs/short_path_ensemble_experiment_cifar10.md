# Short-path Ensemble Lesion Experiment

## Objective

Evaluate whether a trained ResNet remains partially functional when selected residual branches are removed.

## Theory Background

Residual blocks contain a shortcut path and a residual branch. Lesion masks evaluate `y = x + m_l * F(x)`, where `m_l = 0` disables a residual branch while preserving the shortcut.

## Lesion Method

- Checkpoint used: `results\identity_mapping_cifar10\checkpoints\PreActResNet-56.pt`
- Drop ratios: 0%, 10%, 30%, 50%, 70%
- Random Drop uses multiple seeds when requested.
- Early Drop removes branches near the input side.
- Late Drop removes branches near the output side.

## Generated Figures

- `results\lesion_study_cifar10\figures\fig_lesion_accuracy_vs_drop_ratio.png`
- `results\lesion_study_cifar10\figures\fig_lesion_accuracy_drop_vs_drop_ratio.png`
- `results\lesion_study_cifar10\figures\fig_lesion_random_vs_early_vs_late.png`
- `results\lesion_study_cifar10\figures\fig_active_blocks_vs_accuracy.png`
- `results\lesion_study_cifar10\figures\fig_lesion_heatmap.png`

## Generated Tables

- `results\lesion_study_cifar10\tables\table_lesion_accuracy.csv`
- `results\lesion_study_cifar10\tables\table_lesion_drop_strategy_comparison.csv`
- `results\lesion_study_cifar10\tables\table_active_blocks_summary.csv`

## Conceptual PlainNet Comparison

PlainNet cannot remove intermediate layers without breaking the only forward path. In contrast, ResNet can disable residual branches while shortcuts still preserve an information path.

## Preliminary Interpretation

The CIFAR-10 lesion run shows a meaningful degradation trend. The unlesioned PreAct ResNet-56 checkpoint reached 18.36% accuracy on the selected validation subset. Random residual-branch removal declined from 18.36% at 0% drop to 16.15% at 10%, 11.85% at 30%, 9.57% at 50%, and 9.11% at 70%. Early-drop and late-drop both degraded performance, with late-drop showing higher sensitivity at 10% and 30% in this run.

The model remained partially functional under branch removal because the shortcut path still preserved a forward route. This behavior is consistent with the short-path ensemble interpretation, but the checkpoint quality is modest, so the result should be presented as preliminary evidence rather than a final quantitative claim.

## Limitations

- This run used real CIFAR-10 validation samples, but the checkpoint was trained with a lightweight budget.
- Accuracy values should not be used as final CIFAR-10 evidence unless rerun with longer training and multiple seeds.
- Results support cautious interpretation only.

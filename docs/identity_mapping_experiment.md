# Identity Mapping Experiment

## Objective

Evaluate whether shortcuts close to identity mapping are associated with more stable optimization and gradient propagation.

## Theory Background

For a residual block `x_{l+1} = x_l + F_l(x_l)`, the identity term creates a direct gradient component. Scaled shortcuts test `x_{l+1} = lambda * x_l + F_l(x_l)`.

## Compared Models

- PlainNet-56
- Standard ResNet-56
- PreAct ResNet-56
- ScaledShortcut ResNet-56 with lambda values 0.5, 0.9, 1.0, and 1.1

## Training Settings

- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite identity --dataset auto --epochs 1 --train-size 112 --val-size 56 --batch-size 28 --learning-rate 0.01 --device cpu --torch-threads 2`
- Dataset: FakeData
- Dataset note: CIFAR-10 unavailable; using FakeData fallback: Dataset not found or corrupted. You can use download=True to download it

## Generated Figures

- `results\identity_mapping\figures\fig_identity_loss_curve.png`
- `results\identity_mapping\figures\fig_identity_accuracy_curve.png`
- `results\identity_mapping\figures\fig_identity_layerwise_grad_norm.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_all.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_plainnet.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_resnet.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_preact_resnet.png`
- `results\identity_mapping\figures\fig_identity_lambda_ablation_grad_ratio.png`

## Generated Tables

- `results\identity_mapping\tables\table_identity_model_comparison.csv`
- `results\identity_mapping\tables\table_identity_gradient_stability.csv`
- `results\identity_mapping\tables\table_identity_lambda_ablation.csv`

## Preliminary Interpretation

The run provides saved evidence for comparing optimization and gradient statistics. Because the setting is lightweight, wording should remain cautious: results may support or be consistent with the identity shortcut hypothesis, but they do not prove it generally.

## Limitations

- Small dataset subset and short training budget.
- FakeData fallback, if used, validates pipeline behavior rather than CIFAR-10 scientific trends.
- Single seed unless rerun with additional seeds.

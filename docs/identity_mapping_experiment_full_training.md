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

- Command: `python -m harness.run_suite --suite identity --dataset cifar10 --data-root data\cifar10_verified --epochs 30 --train-size 50000 --val-size 10000 --batch-size 64 --num-workers 2 --learning-rate 0.025 --seed 0 --device cuda --torch-threads 2 --log-interval 100 --output-tag full_training --output-root results\full_training`
- Dataset: CIFAR10
- Dataset note: none

## Generated Figures

- `results\full_training\figures\fig_identity_loss_curve.png`
- `results\full_training\figures\fig_identity_accuracy_curve.png`
- `results\full_training\figures\fig_identity_layerwise_grad_norm.png`
- `results\full_training\figures\fig_identity_gradient_heatmap_all.png`
- `results\full_training\figures\fig_identity_gradient_heatmap_plainnet.png`
- `results\full_training\figures\fig_identity_gradient_heatmap_resnet.png`
- `results\full_training\figures\fig_identity_gradient_heatmap_preact_resnet.png`
- `results\full_training\figures\fig_identity_lambda_ablation_grad_ratio.png`

## Generated Tables

- `results\full_training\tables\table_identity_full_training_model_comparison.csv`
- `results\full_training\tables\table_identity_full_training_gradient_stability.csv`
- `results\full_training\tables\table_identity_full_training_lambda_ablation.csv`

## Preliminary Interpretation

The run provides saved evidence for comparing optimization and gradient statistics. Because the setting is lightweight, wording should remain cautious: results may support or be consistent with the identity shortcut hypothesis, but they do not prove it generally.

## Limitations

- Small dataset subset and short training budget.
- This run used real CIFAR-10 data, but still only a subset and a short training budget.
- Single seed unless rerun with additional seeds.

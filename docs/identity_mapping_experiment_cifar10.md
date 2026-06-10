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

- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite identity --dataset cifar10 --download --output-tag cifar10 --epochs 3 --train-size 2048 --val-size 512 --batch-size 128 --learning-rate 0.05 --device auto --torch-threads 2`
- Dataset: CIFAR10
- Dataset note: none

## Generated Figures

- `results\identity_mapping_cifar10\figures\fig_identity_loss_curve.png`
- `results\identity_mapping_cifar10\figures\fig_identity_accuracy_curve.png`
- `results\identity_mapping_cifar10\figures\fig_identity_layerwise_grad_norm.png`
- `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_all.png`
- `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_plainnet.png`
- `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_resnet.png`
- `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_preact_resnet.png`
- `results\identity_mapping_cifar10\figures\fig_identity_lambda_ablation_grad_ratio.png`

## Generated Tables

- `results\identity_mapping_cifar10\tables\table_identity_cifar10_model_comparison.csv`
- `results\identity_mapping_cifar10\tables\table_identity_cifar10_gradient_stability.csv`
- `results\identity_mapping_cifar10\tables\table_identity_cifar10_lambda_ablation.csv`

## Preliminary Interpretation

The CIFAR-10 run provides real-data evidence that the residual variants optimize more easily than the PlainNet baseline in this lightweight setting. PlainNet-56 ended at 11.33% validation accuracy, while ResNet-56 reached 21.68% and PreAct ResNet-56 reached 18.36%. The PlainNet final shallow-to-deep gradient ratio was 0.055, much more extreme than the residual variants, which is consistent with less stable gradient propagation.

For the scaled shortcut ablation, lambda values near identity performed better than lambda values farther away in this run: lambda=0.9 reached the best validation accuracy among scaled variants, lambda=1.0 remained competitive, and lambda=1.1 showed a large first-epoch training loss. This supports the identity-mapping hypothesis cautiously, but the run is too small to claim a general ordering among all lambda values.

## Limitations

- Small dataset subset and short training budget.
- This run used real CIFAR-10 data, but only a subset and three epochs.
- Single seed unless rerun with additional seeds.

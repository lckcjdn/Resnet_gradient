# identity_mapping_full_training Log

- Started: 2026-06-13T17:16:10
- Command: `python -m harness.run_suite --suite identity --dataset cifar10 --data-root data\cifar10_verified --epochs 30 --train-size 50000 --val-size 10000 --batch-size 64 --num-workers 2 --learning-rate 0.025 --seed 0 --device cuda --torch-threads 2 --log-interval 100 --output-tag full_training --output-root results\full_training`
- Python: `E:\python\envs\resnet-gradient-path-study\python.exe`
- Device: `cuda`
- Dataset: `CIFAR10`
- Train size: 50000
- Validation size: 10000
- Fallback note: none
- Epochs: 30
- Batch size: 64
- Learning rate: 0.025

## Outputs

- `results/full_training/figures/fig_identity_loss_curve.png`
- `results/full_training/figures/fig_identity_accuracy_curve.png`
- `results/full_training/figures/fig_identity_layerwise_grad_norm.png`
- `results/full_training/figures/fig_identity_gradient_heatmap_all.png`
- `results/full_training/tables/identity_full_training_metrics.csv`
- `results/full_training/tables/identity_full_training_model_comparison.csv`
- `results/full_training/tables/identity_full_training_gradient_stability.csv`
- `results/full_training/tables/identity_full_training_status.csv`
- `results/full_training/gradients/identity_full_training_gradient_stats.csv`

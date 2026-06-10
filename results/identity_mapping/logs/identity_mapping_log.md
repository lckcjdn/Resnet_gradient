# identity_mapping Log

- Started: 2026-06-10T23:19:53
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite identity --dataset auto --epochs 1 --train-size 112 --val-size 56 --batch-size 28 --learning-rate 0.01 --device cpu --torch-threads 2`
- Python: `E:\python\envs\resnet-gradient-path-study\python.exe`
- Device: `cpu`
- Dataset: `FakeData`
- Train size: 112
- Validation size: 56
- Fallback note: CIFAR-10 unavailable; using FakeData fallback: Dataset not found or corrupted. You can use download=True to download it
- Epochs: 1
- Batch size: 28
- Learning rate: 0.01

## Outputs

- `results/identity_mapping/figures/fig_identity_loss_curve.png`
- `results/identity_mapping/figures/fig_identity_accuracy_curve.png`
- `results/identity_mapping/figures/fig_identity_layerwise_grad_norm.png`
- `results/identity_mapping/figures/fig_identity_gradient_heatmap_all.png`
- `results/identity_mapping/tables/identity_metrics.csv`
- `results/identity_mapping/tables/identity_model_comparison.csv`
- `results/identity_mapping/tables/identity_gradient_stability.csv`
- `results/identity_mapping/tables/identity_status.csv`
- `results/identity_mapping/gradients/identity_gradient_stats.csv`

# smoke_test Log

- Started: 2026-06-10T23:17:12
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset fake --epochs 1 --train-size 96 --val-size 48 --batch-size 24 --learning-rate 0.01 --device cpu --torch-threads 2`
- Python: `E:\python\envs\resnet-gradient-path-study\python.exe`
- Device: `cpu`
- Dataset: `FakeData`
- Train size: 96
- Validation size: 48
- Fallback note: FakeData explicitly requested.
- Epochs: 1
- Batch size: 24
- Learning rate: 0.01

## Outputs

- `results/figures/smoke_loss_curve.png`
- `results/figures/smoke_accuracy_curve.png`
- `results/figures/smoke_layerwise_grad_norm.png`
- `results/figures/smoke_gradient_heatmap.png`
- `results/tables/smoke_metrics.csv`
- `results/tables/smoke_model_comparison.csv`
- `results/tables/smoke_gradient_stability.csv`
- `results/tables/smoke_status.csv`
- `results/gradients/smoke_gradient_stats.csv`

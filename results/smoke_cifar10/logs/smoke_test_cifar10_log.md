# smoke_test_cifar10 Log

- Started: 2026-06-10T23:48:44
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset cifar10 --download --output-tag cifar10 --epochs 2 --train-size 512 --val-size 256 --batch-size 64 --learning-rate 0.05 --device auto --torch-threads 2`
- Python: `E:\python\envs\resnet-gradient-path-study\python.exe`
- Device: `cuda`
- Dataset: `CIFAR10`
- Train size: 512
- Validation size: 256
- Fallback note: none
- Epochs: 2
- Batch size: 64
- Learning rate: 0.05

## Outputs

- `results/smoke_cifar10/figures/smoke_loss_curve.png`
- `results/smoke_cifar10/figures/smoke_accuracy_curve.png`
- `results/smoke_cifar10/figures/smoke_layerwise_grad_norm.png`
- `results/smoke_cifar10/figures/smoke_gradient_heatmap.png`
- `results/smoke_cifar10/tables/smoke_cifar10_metrics.csv`
- `results/smoke_cifar10/tables/smoke_cifar10_model_comparison.csv`
- `results/smoke_cifar10/tables/smoke_cifar10_gradient_stability.csv`
- `results/smoke_cifar10/tables/smoke_cifar10_status.csv`
- `results/smoke_cifar10/gradients/smoke_cifar10_gradient_stats.csv`

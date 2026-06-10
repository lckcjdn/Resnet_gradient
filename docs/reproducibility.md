# Reproducibility

## Smoke Test Command

Run inside the project conda environment:

```bash
conda activate resnet-gradient-path-study
D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset fake --epochs 1 --train-size 96 --val-size 48 --batch-size 24 --learning-rate 0.01 --device cpu --torch-threads 2
```

The smoke test uses a small subset and may fall back to `torchvision.datasets.FakeData` when CIFAR-10 is unavailable.

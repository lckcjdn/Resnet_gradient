# Lesion Study Log

- Started: 2026-06-10T23:25:45
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\lesion_study.py --dataset auto --val-size 72 --batch-size 24 --device cpu --torch-threads 2 --drop-ratios 0,0.1,0.3,0.5,0.7 --random-seeds 0,1,2`
- Checkpoint: `results\identity_mapping\checkpoints\PreActResNet-56.pt`
- Model: PreActResNet-56
- Dataset: FakeData
- Dataset note: CIFAR-10 unavailable; using FakeData fallback: Dataset not found or corrupted. You can use download=True to download it
- Baseline accuracy: 0.1111111111111111
- Total residual blocks: 27
- Drop ratios: [0.0, 0.1, 0.3, 0.5, 0.7]
- Random seeds: [0, 1, 2]

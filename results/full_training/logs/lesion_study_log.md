# Lesion Study Log

- Started: 2026-06-14T01:14:20
- Command: `python -m harness.lesion_study --checkpoint results\full_training\checkpoints\PreActResNet-56.pt --dataset cifar10 --data-root data\cifar10_verified --val-size 10000 --batch-size 64 --num-workers 2 --device cuda --seed 0 --random-seeds 0,1,2 --drop-ratios 0,0.1,0.3,0.5,0.7 --torch-threads 2 --output-tag full_training --output-root results\full_training`
- Checkpoint: `results\full_training\checkpoints\PreActResNet-56.pt`
- Model: PreActResNet-56
- Dataset: CIFAR10
- Dataset note: none
- Baseline accuracy: 0.7988
- Total residual blocks: 27
- Drop ratios: [0.0, 0.1, 0.3, 0.5, 0.7]
- Random seeds: [0, 1, 2]

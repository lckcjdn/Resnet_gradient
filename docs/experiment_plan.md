# Experiment Plan

## Goal

Run controlled experiments that test whether identity shortcuts are associated with more stable gradient propagation and whether residual branch lesions show short-path ensemble-like behavior.

## Dataset

Primary dataset: CIFAR-10.

Optional datasets: CIFAR-100 or Fashion-MNIST if time allows.

## Shared Training Settings

- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 5e-4
- Batch size: 128
- Initial learning rate: 0.1
- Scheduler: MultiStepLR or cosine annealing
- Seeds: 0, 1, 2 for final runs

## Experiments

1. PlainNet-56 vs standard ResNet-56 vs PreAct ResNet-56.
2. Scaled shortcut ablation with lambda values 0.5, 0.9, 1.0, and 1.1.
3. Residual branch lesion study with random, early, late, and interval drops.
4. Optional active block number analysis.

## Initialization Boundary

This framework initialization does not run full experiments. It only creates the scaffold and minimal sanity-check code.

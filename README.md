# ResNet Gradient Path Study

This project is a reproducible course framework for studying two ResNet mechanisms:

1. Identity shortcuts as direct signal and gradient paths.
2. Short-path ensemble behavior under residual branch lesion tests.

The goal is not to maximize CIFAR-10 accuracy. The goal is to run controlled comparisons between PlainNet, standard ResNet, PreAct ResNet, scaled shortcuts, and lesion settings, then document the evidence carefully.

## Quick Start

```bash
conda env create -f environment.yml
conda activate resnet-gradient-path-study
python -m harness.sanity_check
```

If the conda package channels are unavailable on this machine, create the environment with an available local conda source and then install missing packages inside that environment. The experiments in this repository were run through `conda run -n resnet-gradient-path-study ...`.

No full experiment should be run during initialization. Use the sanity check first, then validate the smoke experiment before launching longer runs.

## Project Layout

- `.codex/`: Codex project rules, skills, and execution harness notes.
- `configs/`: YAML experiment configurations.
- `docs/`: Theory notes, experiment plan, logs, conclusions, and report outline.
- `src/`: Data, model, training, analysis, and plotting modules.
- `harness/`: Experiment orchestration and artifact management.
- `scripts/`: Thin command-line wrappers for common workflows.
- `results/`: Logs, gradients, figures, tables, checkpoints, and run folders.

## Minimal Validation

```bash
python -m harness.sanity_check
python -m compileall src harness scripts
```

The sanity check creates missing result directories and, when PyTorch is installed, runs a tiny forward/backward pass for each model family.

## Experiment Commands

```bash
python -m harness.run_suite --suite smoke --dataset fake --epochs 1 --train-size 96 --val-size 48 --batch-size 24 --learning-rate 0.01 --device cpu --torch-threads 2
python -m harness.run_suite --suite identity --dataset auto --epochs 1 --train-size 112 --val-size 56 --batch-size 28 --learning-rate 0.01 --device cpu --torch-threads 2
python -m harness.lesion_study --dataset auto --val-size 72 --batch-size 24 --device cpu --torch-threads 2 --drop-ratios 0,0.1,0.3,0.5,0.7 --random-seeds 0,1,2
```

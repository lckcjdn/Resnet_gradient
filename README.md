# ResNet Gradient Path Study

This project is a reproducible course framework for studying two ResNet mechanisms:

1. Identity shortcuts as direct signal and gradient paths.
2. Short-path ensemble behavior under residual branch lesion tests.

The goal is not to maximize CIFAR-10 accuracy. The goal is to run controlled comparisons between PlainNet, standard ResNet, PreAct ResNet, scaled shortcuts, and lesion settings, then document the evidence carefully.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m harness.sanity_check
```

No full experiment should be run during initialization. Use the sanity check first, then implement and validate one small training path before launching longer runs.

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

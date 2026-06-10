# Process Conclusions

This document records intermediate conclusions during the project.
All conclusions should be written carefully and grounded in generated figures/tables.

## Conclusion Style Rules

Use:

- "The result suggests..."
- "The result is consistent with..."
- "The figure provides evidence that..."
- "The current run does not fully support..."

Avoid:

- "This completely proves..."
- "ResNet always..."
- "Gradient vanishing is fully solved..."

---

## Experiment 1: PlainNet vs ResNet vs PreAct ResNet

### Current Evidence

Not run yet.

### Interpretation

Pending generated figures and tables.

### Limitations

No experiment results have been collected during initialization.

### Next Actions

Implement the training loop and run a quick one-epoch smoke test before full training.

---

## Experiment 2: Scaled Shortcut Ablation

### Current Evidence

Not run yet.

### Interpretation

Pending lambda ablation results.

### Limitations

No experiment results have been collected during initialization.

### Next Actions

Verify that shortcut lambda is recorded in configs, metrics, and run summaries.

---

## Experiment 3: Residual Branch Lesion

### Current Evidence

Not run yet.

### Interpretation

Pending lesion study results.

### Limitations

No trained checkpoint is available yet.

### Next Actions

Implement residual branch masks and evaluation-only lesion runs after baseline training.

---

## Smoke Test Observations

The smoke test exercises model construction, one short training loop, evaluation, gradient recording, figure generation, and table generation. These outputs validate the pipeline mechanics; they should not be used as scientific evidence about ResNet behavior.

## Identity Mapping Experiment Observations

This lightweight identity-mapping run produced comparable metrics and layer-wise gradient statistics for PlainNet, standard ResNet, PreAct ResNet, and scaled shortcut variants. Because the run is intentionally small, conclusions should be treated as preliminary pipeline evidence.

## Lesion Study Observations

The lesion study verifies that residual branch masks can be applied without breaking the forward path. Because this run uses a lightweight checkpoint and may use FakeData fallback, the observed accuracy trend should be interpreted as pipeline evidence and a preliminary illustration rather than a final scientific result.

## CIFAR-10 Real-data Rerun Summary

See `docs/cifar10_rerun_analysis.md` for the full real-data analysis and dataset provenance table.

### Data Source Separation

- FakeData results live under `results/`, `results/identity_mapping/`, and `results/lesion_study/`. These validate the pipeline mechanics only.
- CIFAR-10 results live under `results/smoke_cifar10/`, `results/identity_mapping_cifar10/`, and `results/lesion_study_cifar10/`. These are the current real-data experiment results.

### Round 1: CIFAR-10 Smoke Test

The smoke test confirms that the end-to-end pipeline works on real CIFAR-10 data. A fairness issue was found and fixed: each model now receives a fresh same-seed DataLoader, so model comparisons are less affected by shuffled data order. The corrected smoke run produced loss/accuracy curves, gradient figures, tables, and 72 gradient-stat rows.

### Round 2: CIFAR-10 Identity Mapping

The CIFAR-10 identity-mapping run is consistent with the expected direction. PlainNet-56 reached 11.33% validation accuracy, while ResNet-56 reached 21.68% and PreAct ResNet-56 reached 18.36%. PlainNet-56 also had a much more extreme final shallow-to-deep gradient ratio of 0.055. Scaled shortcut variants near identity behaved better than more distorted shortcuts in this lightweight run, though lambda=0.9 outperformed lambda=1.0 on validation accuracy, so the conclusion should remain cautious.

### Round 3: CIFAR-10 Lesion Study

The CIFAR-10 lesion run used `results/identity_mapping_cifar10/checkpoints/PreActResNet-56.pt`. Random-drop accuracy declined from 18.36% at 0% drop to 16.15%, 11.85%, 9.57%, and 9.11% at 10%, 30%, 50%, and 70% drop. This gradual degradation is consistent with the short-path ensemble interpretation. Late-drop was more sensitive than early-drop at lower drop ratios in this run.

### Limitations

- CIFAR-10 runs are real-data runs, but still lightweight subset experiments.
- The checkpoint quality is modest; longer training and multiple seeds are needed before making stronger claims.
- Current conclusions should use "supports", "suggests", and "is consistent with", not stronger proof language.

## Smoke Test (cifar10) Observations

The smoke test exercises model construction, one short training loop, evaluation, gradient recording, figure generation, and table generation. These outputs validate the pipeline mechanics; they should not be used as scientific evidence about ResNet behavior.

## Smoke Test (cifar10) Observations

The smoke test exercises model construction, one short training loop, evaluation, gradient recording, figure generation, and table generation. These outputs validate the pipeline mechanics; they should not be used as scientific evidence about ResNet behavior.

## Identity Mapping Experiment (cifar10) Observations

This lightweight identity-mapping run produced comparable metrics and layer-wise gradient statistics for PlainNet, standard ResNet, PreAct ResNet, and scaled shortcut variants. Because the run is intentionally small, conclusions should be treated as preliminary pipeline evidence.

## Lesion Study Observations

The lesion study verifies that residual branch masks can be applied without breaking the forward path. Because this run uses a lightweight checkpoint and may use FakeData fallback, the observed accuracy trend should be interpreted as pipeline evidence and a preliminary illustration rather than a final scientific result.

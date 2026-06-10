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

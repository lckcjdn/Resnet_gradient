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

## Identity Mapping Experiment (full_training) Observations

This lightweight identity-mapping run produced comparable metrics and layer-wise gradient statistics for PlainNet, standard ResNet, PreAct ResNet, and scaled shortcut variants. Because the run is intentionally small, conclusions should be treated as preliminary pipeline evidence.

## Lesion Study Observations

The lesion study verifies that residual branch masks can be applied without breaking the forward path. Because this run uses a lightweight checkpoint and may use FakeData fallback, the observed accuracy trend should be interpreted as pipeline evidence and a preliminary illustration rather than a final scientific result.

## Toy 1D Init Gradient Analysis Observations

Toy 1D initialization gradients were computed in eval mode with no optimizer updates.
The scripts can emit raw gradient vectors, one-row gradient matrices, covariance matrices, ACF values, and summary metrics; the organized Git-ready results retain the figures, tables, metadata, and manifests while omitting raw `.npy` caches.
Only seed 0 should be interpreted as a quick-run setting unless the script is rerun with more seeds.

## Toy 1D Checkpoint Gradient Evolution Observations

Toy 1D checkpoint gradients were computed after saving epoch 0 before optimizer updates.
Training uses a synthetic regression target y = sin(3x) + 0.3 cos(7x).
Gradient analysis is read-only and uses eval mode for every saved checkpoint.

## CIFAR-10 Init Gradient Analysis Observations

CIFAR-10 initialization analysis used real CIFAR-10 test samples and fixed minibatch indices.
Gradient analysis was read-only: models were in eval mode and no optimizer step was called.
This run used the runtime-limited setting from the plan: seed 0, depths 20 and 56, batch size 64, and 5 minibatches unless rerun with broader arguments.

## CIFAR-10 Checkpoint Gradient Evolution Observations

CIFAR-10 checkpoint analysis used fixed real CIFAR-10 test minibatches and per-sample input gradients.
The run compares a fresh epoch 0 initialization with the existing full_training final checkpoints; intermediate epoch checkpoints were not present in the source checkpoint directory.
Gradient analysis is read-only and does not update model parameters.

## Gradient Structure Plan Concrete Findings

These findings come from the plan-specific outputs under `results/toy1d_init_gradient_analysis/`, `results/toy1d_checkpoint_gradient_evolution/`, `results/cifar10_init_gradient_analysis/`, and `results/cifar10_checkpoint_gradient_evolution/`.

- Toy 1D initialization: at depth 50, PlainNet gradients nearly collapsed (`gradient_std` about `1.02e-20`, `acf_lag1` about `2.50e-26`), while Standard ResNet / ScaledShortcut lambda=1.0 kept much larger gradient variation (`gradient_std` about `0.405`) and high local ACF (`acf_lag1` about `0.984`). PreAct ResNet also retained structure (`acf_lag1` about `0.928`).
- Toy 1D plotting note: PlainNet depth 24/50 gradients are orders of magnitude smaller than residual-family gradients, so they should be inspected with the PlainNet-only figures. Standard ResNet and ScaledShortcut lambda=1.0 have identical raw Toy 1D gradient vectors in the current implementation, so their lines overlap exactly in combined plots.
- Toy 1D checkpoint training: PlainNet did not fit the synthetic regression target in the 10-epoch quick run (`val_loss` stayed about `0.566`) and its ACF stayed near zero. ResNet, PreAct ResNet, and ScaledShortcut lambda=1.0 reached `val_loss` around `0.009-0.010` with high ACF, suggesting the residual-style toy models preserved a usable gradient structure during this run.
- CIFAR-10 initialization: on real CIFAR-10 fixed minibatches, PlainNet-56 again showed near-zero input gradients (`mean_gradient_norm_mean` about `3.44e-24`, spatial `acf_lag1` about `1.17e-34`). ResNet-56 / ScaledShortcut lambda=1.0 and PreAct ResNet-56 retained nonzero input-gradient structure (`mean_gradient_norm_mean` about `8.46e-4`, spatial `acf_lag1` about `0.268`).
- CIFAR-10 checkpoint evolution: after full_training final checkpoints, PlainNet-56 recovered nonzero gradient structure and reached `val_accuracy=0.7137`, but ResNet-56, PreAct ResNet-56, and ScaledShortcut lambda=1.0 had higher final validation accuracy (`0.8077`, `0.7988`, and `0.8220` respectively) with strong mean input-gradient norms.

Interpretation should remain cautious: in this implementation the deepest PlainNet often shows gradient collapse, which is related to but not identical to white-noise-like shattering. The CIFAR analyses used the plan's runtime-limited setting: seed 0, batch size 64, and 5 fixed minibatches. The checkpoint experiment used epoch 0 plus existing final full_training checkpoints because intermediate full_training checkpoints were not available; model checkpoint files are excluded from the organized Git-ready results and should be regenerated when needed.

## PlainNet Init Ablation Diagnostics Observations

PlainNet initialization/normalization variants were diagnosed before making any shattered-gradient claim.
The diagnostic rule is explicit: if mean sample input-gradient norm < 1e-8 or input-gradient std < 1e-8, the result is labeled gradient_collapse.
Only non-collapsed variants are placed in the shattered-gradient candidate table; collapsed variants are separated under vanishing_collapse_analysis.
PlainNet-HeInit-BN is recorded as the main PlainNet setting for future shattered-gradient comparisons with ResNet.

### Collapse vs Shattered-Gradient Distinction

- A near-zero gradient correlation matrix is not sufficient evidence for shattered gradients. If the input gradients are numerically near zero, cosine similarity is degenerate and the correct label is gradient collapse/vanishing.
- The diagnostic rule for these results is: `input_gradient_norm_mean < 1e-8` or `gradient_std < 1e-8` implies `gradient_collapse`.
- In `results/plainnet_init_ablation/`, `PlainNet-DefaultInit` was labeled `gradient_collapse` (`input_gradient_norm_mean` about `2.58e-23`, `gradient_std=0.0`). `PlainNet-XavierInit` was also labeled `gradient_collapse` (`input_gradient_norm_mean` about `4.22e-11`, `gradient_std` about `7.75e-13`).
- `PlainNet-HeInit`, `PlainNet-HeInit-BN`, and `PlainNet-OrthogonalInit` passed the non-collapse rule and are stored as shattered-gradient candidates. Among them, `PlainNet-HeInit-BN` is the main PlainNet setting for future ResNet comparisons because it combines He initialization with calibrated batch normalization and has a robust nonzero input-gradient norm (`input_gradient_norm_mean` about `2.13`, `gradient_std` about `3.95e-2`).
- Final conclusions about shattered gradients should therefore use `PlainNet-HeInit-BN` or another non-collapsed PlainNet variant. The previous default PlainNet epoch-0 CIFAR-10 correlation image should be described as gradient collapse, not as white-noise-like shattering.

## Toy 1D Figure 2 Reproduction Observations

This is an epoch-0 initialization-only analysis: no checkpoints, training loop, or optimizer step is used.
Twenty-run averaging used seeds 0 to 19.
No model-depth setting had collapsed runs under the 1e-8 norm/std rule.
At depth 50, lag-5 mean ACF was feedforward=-0.05423, ResNet beta=1.0=0.6047, and ResNet beta=0.1=0.8404.
The feedforward ACF decayed faster than the beta=1.0 ResNet at the comparison depth.
The beta=0.1 ResNet preserved stronger local ACF than beta=1.0 at the comparison depth.
The implementation is reproduction-inspired rather than exact: model layers use per-sample feature centering/normalization to keep df(x_i)/dx_i pointwise well-defined.

# PlainNet Initialization Ablation

This directory diagnoses whether PlainNet epoch-0 CIFAR-10 gradients are usable for shattered-gradient analysis or should be labeled as gradient collapse.

## Diagnostic Rule

If either condition is true, the result is labeled `gradient_collapse`:

- `input_gradient_norm_mean < 1e-8`
- `gradient_std < 1e-8`

Collapsed variants are separated under `vanishing_collapse_analysis/`. Non-collapsed variants are placed under `shattered_gradient_analysis/`.

## Variants

- `PlainNet-DefaultInit`: default PyTorch initialization with the repo-style PlainNet BN layers left uncalibrated at initialization.
- `PlainNet-XavierInit`: no-BN PlainNet with Xavier initialization.
- `PlainNet-HeInit`: no-BN PlainNet with He initialization.
- `PlainNet-HeInit-BN`: He initialization with calibrated BN statistics; this is the main PlainNet setting for future shattered-gradient comparisons with ResNet.
- `PlainNet-OrthogonalInit`: no-BN PlainNet with orthogonal initialization.

## Summary

- `PlainNet-DefaultInit` and `PlainNet-XavierInit` are classified as `gradient_collapse`.
- `PlainNet-HeInit`, `PlainNet-HeInit-BN`, and `PlainNet-OrthogonalInit` are non-collapsed shattered-gradient candidates.
- The default PlainNet CIFAR-10 epoch-0 correlation image should not be claimed as shattered gradients; it reflects near-zero input gradients.

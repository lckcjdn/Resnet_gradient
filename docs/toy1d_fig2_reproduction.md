# Toy 1D Figure 2 Reproduction

## Experiment Objective

This experiment reproduces a Figure-2-style autocorrelation-function comparison for Toy 1D input gradients. It compares feedforward ReLU MLPs, residual MLPs with beta=1.0, residual MLPs with beta=0.1, and white/Brown-noise references.

## Relation to Paper Figure 2

The target is inspired by Figure 2 from "The Shattered Gradients Problem: If ResNets are the answer, then what is the question?". The implementation is not an exact architectural reproduction; it keeps the same Toy 1D ACF question and reports initialization-only gradients with explicit collapse diagnostics.

## Initialization-Only Rule

No model is trained. The script initializes each model from scratch for every seed, computes `df/dx` on the fixed Toy 1D grid, and never calls `optimizer.step()`. Saved checkpoints are not loaded or used.

## Model Settings

- Input grid: `256` points on `[-2.0, 2.0]`.
- Depths: `[2, 4, 10, 24, 50]`.
- Width: `200` hidden units.
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`.
- Max lag: `15`.
- Feedforward mean-centering: `True`.
- ResNet residual-branch feature centering/normalization: `True`.
- Collapse rule: `gradient_norm < 1e-8` or `gradient_std < 1e-8`.

## ACF Formula

For gradient sequence `g`, let `z = g - mean(g)`. For lag `k`:

```text
ACF(k) = sum_i z_i * z_(i+k) / sum_i z_i^2
```

`ACF(0)` is fixed at 1. If the denominator is numerically near zero, the run is marked as `gradient_collapse` and nonzero-lag ACF values are left uninterpreted.

## Random Run Averaging

Each model-depth setting is run once per seed. Mean and standard deviation ACF curves are computed over valid non-collapsed runs only. Collapsed runs remain in the per-run summary and diagnostics tables.

## Generated Figures

- `results\toy1d_fig2_reproduction\figures\fig_toy1d_acf_figure2_reproduction.png`
- `results\toy1d_fig2_reproduction\figures\fig_toy1d_gradient_examples_depth50.png`
- `results\toy1d_fig2_reproduction\figures\fig_toy1d_gradient_norm_by_depth.png`
- `results\toy1d_fig2_reproduction\figures\fig_toy1d_gradient_std_by_depth.png`

## Generated Tables

- `results\toy1d_fig2_reproduction\tables\table_toy1d_acf_summary.csv`
- `results\toy1d_fig2_reproduction\tables\table_toy1d_acf_mean_std.csv`
- `results\toy1d_fig2_reproduction\tables\table_toy1d_collapse_diagnostics.csv`
- `results\toy1d_fig2_reproduction\artifact_manifest.csv`

## Preliminary Interpretation

- No model-depth setting had collapsed runs under the 1e-8 norm/std rule.
- At depth 50, lag-5 mean ACF was feedforward=-0.05423, ResNet beta=1.0=0.6047, and ResNet beta=0.1=0.8404.
- The feedforward ACF decayed faster than the beta=1.0 ResNet at the comparison depth.
- The beta=0.1 ResNet preserved stronger local ACF than beta=1.0 at the comparison depth.
- Rapid ACF decay indicates weaker local gradient correlation; slow decay indicates stronger gradient structure.

## Limitations

- This is a reproduction-inspired analysis rather than an exact reproduction of every paper detail.
- The Toy 1D networks use per-sample feature centering/normalization so each grid point keeps a clear pointwise derivative.
- Results should be interpreted together with collapse diagnostics; near-zero gradients are labeled collapse rather than shattered gradients.
- The experiment does not evaluate training dynamics or checkpoint evolution.

## Command

```bash
python.exe scripts/run_toy1d_fig2_reproduction.py --depths 2 4 10 24 50 --width 200 --num-points 256 --x-min -2 --x-max 2 --max-lag 15 --runs 20 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --output-dir results/toy1d_fig2_reproduction
```

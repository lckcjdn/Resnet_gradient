# CIFAR-10 Rerun Analysis

## Dataset Provenance

The earlier round used `torchvision.datasets.FakeData` because CIFAR-10 was not available locally. The rerun downloaded CIFAR-10 successfully and saved real-data outputs in separate directories so fake-data and real-data artifacts are not mixed.

| Data source | Purpose | Result directories |
|---|---|---|
| FakeData | Initial pipeline validation | `results/`, `results/identity_mapping/`, `results/lesion_study/` |
| CIFAR-10 | Real-data rerun | `results/smoke_cifar10/`, `results/identity_mapping_cifar10/`, `results/lesion_study_cifar10/` |

## Environment And Download

- Environment: `conda run -n resnet-gradient-path-study ...`
- CIFAR-10 path: `data/cifar-10-batches-py/`
- Download verification: 50,000 training samples and 10,000 test samples.
- Device used by experiment runners: CUDA.

## Round 1: Smoke Test On CIFAR-10

Command:

```bash
conda run -n resnet-gradient-path-study python -m harness.run_suite --suite smoke --dataset cifar10 --download --output-tag cifar10 --epochs 2 --train-size 512 --val-size 256 --batch-size 64 --learning-rate 0.05 --device auto --torch-threads 2
```

Generated artifacts:

- `results/smoke_cifar10/figures/smoke_loss_curve.png`
- `results/smoke_cifar10/figures/smoke_accuracy_curve.png`
- `results/smoke_cifar10/figures/smoke_layerwise_grad_norm.png`
- `results/smoke_cifar10/figures/smoke_gradient_heatmap.png`
- `results/smoke_cifar10/tables/smoke_cifar10_model_comparison.csv`
- `results/smoke_cifar10/gradients/smoke_cifar10_gradient_stats.csv`

Analysis:

- The smoke test used real CIFAR-10 and completed all four 20-layer model variants.
- Training loss decreased over two epochs for the tested models.
- Gradient recording worked, producing 72 gradient-stat rows.
- A fairness issue was found after the first CIFAR-10 smoke attempt: model comparisons were not guaranteed to see the same shuffled data order. This was improved by seeding the DataLoader generator and recreating loaders per model. The retained `results/smoke_cifar10/` files are from the corrected rerun.

## Round 2: Identity Mapping On CIFAR-10

Command:

```bash
conda run -n resnet-gradient-path-study python -m harness.run_suite --suite identity --dataset cifar10 --download --output-tag cifar10 --epochs 3 --train-size 2048 --val-size 512 --batch-size 128 --learning-rate 0.05 --device auto --torch-threads 2
```

Generated artifacts:

- `results/identity_mapping_cifar10/figures/fig_identity_loss_curve.png`
- `results/identity_mapping_cifar10/figures/fig_identity_accuracy_curve.png`
- `results/identity_mapping_cifar10/figures/fig_identity_layerwise_grad_norm.png`
- `results/identity_mapping_cifar10/figures/fig_identity_gradient_heatmap_plainnet.png`
- `results/identity_mapping_cifar10/figures/fig_identity_gradient_heatmap_resnet.png`
- `results/identity_mapping_cifar10/figures/fig_identity_gradient_heatmap_preact_resnet.png`
- `results/identity_mapping_cifar10/figures/fig_identity_lambda_ablation_grad_ratio.png`
- `results/identity_mapping_cifar10/tables/table_identity_cifar10_model_comparison.csv`
- `results/identity_mapping_cifar10/tables/table_identity_cifar10_gradient_stability.csv`
- `results/identity_mapping_cifar10/tables/table_identity_cifar10_lambda_ablation.csv`

Heatmap redraw note: the identity heatmaps now use a shared `log10_grad_norm` color scale from `-2.9018` to `-0.0602`, computed from `results/identity_mapping_cifar10/gradients/identity_cifar10_gradient_stats.csv`.

Analysis:

- PlainNet-56 remained near chance-level validation accuracy at 11.33%.
- ResNet-56 reached 21.68% validation accuracy, and PreAct ResNet-56 reached 18.36%.
- PlainNet-56 had a final shallow-to-deep gradient ratio of 0.055, indicating a much more uneven layer-wise gradient distribution than the residual variants in this run.
- ScaledShortcut lambda=0.9 and lambda=1.0 behaved better than lambda=0.5 and lambda=1.1 on the lightweight CIFAR-10 subset, broadly consistent with the identity-shortcut hypothesis.
- The lambda ordering is not definitive: lambda=0.9 outperformed lambda=1.0 on validation accuracy in this small run, so the correct conclusion is that shortcuts near identity looked better than more distorted shortcuts, not that lambda=1.0 always wins.

## Round 3: Lesion Study On CIFAR-10

Command:

```bash
conda run -n resnet-gradient-path-study python -m harness.lesion_study --dataset cifar10 --download --output-tag cifar10 --val-size 512 --batch-size 128 --device auto --torch-threads 2 --drop-ratios 0,0.1,0.3,0.5,0.7 --random-seeds 0,1,2
```

Checkpoint:

```text
results/identity_mapping_cifar10/checkpoints/PreActResNet-56.pt
```

Generated artifacts:

- `results/lesion_study_cifar10/figures/fig_lesion_accuracy_vs_drop_ratio.png`
- `results/lesion_study_cifar10/figures/fig_lesion_accuracy_drop_vs_drop_ratio.png`
- `results/lesion_study_cifar10/figures/fig_lesion_random_vs_early_vs_late.png`
- `results/lesion_study_cifar10/figures/fig_active_blocks_vs_accuracy.png`
- `results/lesion_study_cifar10/figures/fig_lesion_heatmap.png`
- `results/lesion_study_cifar10/tables/table_lesion_accuracy.csv`
- `results/lesion_study_cifar10/tables/table_lesion_drop_strategy_comparison.csv`
- `results/lesion_study_cifar10/tables/table_active_blocks_summary.csv`
- `results/lesion_study_cifar10/masks/`

Analysis:

- Baseline accuracy was 18.36% on the selected CIFAR-10 validation subset.
- Random-drop mean accuracy declined from 18.36% at 0% drop to 16.15%, 11.85%, 9.57%, and 9.11% at 10%, 30%, 50%, and 70% drop.
- Early-drop declined more gradually at low drop ratios than late-drop in this run.
- Late-drop showed stronger sensitivity at 10% and 30%, suggesting later residual branches were more important for this lightly trained checkpoint.
- The model remained partially functional when residual branches were removed, which is consistent with the shortcut path preserving a forward route.

## FakeData vs CIFAR-10 Interpretation

- FakeData results validate the code path only: model construction, training loop, gradient collection, plotting, tables, and lesion masks.
- CIFAR-10 results are the real experimental evidence in this repository.
- The CIFAR-10 runs are still lightweight: small subsets, short training, and mostly single seed. Use cautious language such as "supports", "suggests", and "is consistent with".

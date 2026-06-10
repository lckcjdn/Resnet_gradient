# Expected Figures And Tables

## Figures

| Figure ID | File | Purpose | Status |
|---|---|---|---|
| Fig. 1 | `results/figures/fig01_training_loss_plain_resnet_preact.png` | Compare optimization behavior | planned |
| Fig. 2 | `results/figures/fig02_test_accuracy_plain_resnet_preact.png` | Compare test accuracy curves | planned |
| Fig. 3 | `results/figures/fig03_layerwise_gradient_norm_epoch_last.png` | Compare final layer-wise gradient norms | planned |
| Fig. 4 | `results/figures/fig04_gradient_heatmap_plain_resnet_preact.png` | Show gradient stability over epochs | planned |
| Fig. 5 | `results/figures/fig05_lambda_ablation_loss.png` | Show lambda effect on loss | planned |
| Fig. 6 | `results/figures/fig06_lambda_ablation_gradient_stability.png` | Show lambda effect on gradient stability | planned |
| Fig. 7 | `results/figures/fig07_lambda_ablation_accuracy.png` | Show lambda effect on accuracy | planned |
| Fig. 8 | `results/figures/fig08_lesion_accuracy_curve.png` | Show lesion accuracy trend | planned |
| Fig. 9 | `results/figures/fig09_lesion_loss_curve.png` | Show lesion loss trend | planned |
| Fig. 10 | `results/figures/fig10_lesion_sensitivity_by_block.png` | Show block sensitivity | planned |
| Fig. 11 | `results/figures/fig11_active_blocks_accuracy.png` | Optional active block analysis | planned |

## Tables

| Table ID | CSV | Markdown | Purpose | Status |
|---|---|---|---|---|
| Table 1 | `results/tables/table01_final_performance.csv` | `docs/tables/table01_final_performance.md` | Final performance | planned |
| Table 2 | `results/tables/table02_gradient_stability.csv` | `docs/tables/table02_gradient_stability.md` | Gradient stability | planned |
| Table 3 | `results/tables/table03_lambda_ablation.csv` | `docs/tables/table03_lambda_ablation.md` | Shortcut lambda ablation | planned |
| Table 4 | `results/tables/table04_lesion_summary.csv` | `docs/tables/table04_lesion_summary.md` | Lesion study | planned |
| Table 5 | `results/tables/table05_active_blocks_summary.csv` | `docs/tables/table05_active_blocks_summary.md` | Optional active blocks | planned |

## Generated Artifacts: Smoke Test

- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset fake --epochs 1 --train-size 96 --val-size 48 --batch-size 24 --learning-rate 0.01 --device cpu --torch-threads 2`
- Dataset: FakeData

### Figures
- `results\figures\smoke_loss_curve.png`
- `results\figures\smoke_accuracy_curve.png`
- `results\figures\smoke_layerwise_grad_norm.png`
- `results\figures\smoke_gradient_heatmap.png`

### Tables
- `results\tables\smoke_model_comparison.csv`
- `results\tables\smoke_gradient_stability.csv`
- `results\tables\smoke_status.csv`
- `results\gradients\smoke_gradient_stats.csv`

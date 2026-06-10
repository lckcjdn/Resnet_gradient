# Experiment Log

## Run Template

### Run ID

### Date

### Git Commit

### Config

### Model

### Dataset

### Training Settings

### Status

### Main Results

### Output Files

### Notes

---

## Run: Smoke Test

- Date: 2026-06-10T23:17:19
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset fake --epochs 1 --train-size 96 --val-size 48 --batch-size 24 --learning-rate 0.01 --device cpu --torch-threads 2`
- Dataset: FakeData
- Fallback note: FakeData explicitly requested.
- Log: `results\logs\smoke_test_log.md`
- Gradient stats: `results\gradients\smoke_gradient_stats.csv`
- Figures: `results\figures\smoke_loss_curve.png`, `results\figures\smoke_accuracy_curve.png`, `results\figures\smoke_layerwise_grad_norm.png`, `results\figures\smoke_gradient_heatmap.png`
- Tables: `results\tables\smoke_model_comparison.csv`, `results\tables\smoke_gradient_stability.csv`, `results\tables\smoke_status.csv`
- Status: completed

---

## Run: Identity Mapping Experiment

- Date: 2026-06-10T23:20:08
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite identity --dataset auto --epochs 1 --train-size 112 --val-size 56 --batch-size 28 --learning-rate 0.01 --device cpu --torch-threads 2`
- Dataset: FakeData
- Fallback note: CIFAR-10 unavailable; using FakeData fallback: Dataset not found or corrupted. You can use download=True to download it
- Log: `results\identity_mapping\logs\identity_mapping_log.md`
- Gradient stats: `results\identity_mapping\gradients\identity_gradient_stats.csv`
- Figures: `results\identity_mapping\figures\fig_identity_loss_curve.png`, `results\identity_mapping\figures\fig_identity_accuracy_curve.png`, `results\identity_mapping\figures\fig_identity_layerwise_grad_norm.png`, `results\identity_mapping\figures\fig_identity_gradient_heatmap_all.png`, `results\identity_mapping\figures\fig_identity_gradient_heatmap_plainnet.png`, `results\identity_mapping\figures\fig_identity_gradient_heatmap_resnet.png`, `results\identity_mapping\figures\fig_identity_gradient_heatmap_preact_resnet.png`, `results\identity_mapping\figures\fig_identity_lambda_ablation_grad_ratio.png`
- Tables: `results\identity_mapping\tables\table_identity_model_comparison.csv`, `results\identity_mapping\tables\table_identity_gradient_stability.csv`, `results\identity_mapping\tables\table_identity_lambda_ablation.csv`
- Status: completed

---

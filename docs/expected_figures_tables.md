
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

## Generated Artifacts: Identity Mapping Experiment

- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite identity --dataset auto --epochs 1 --train-size 112 --val-size 56 --batch-size 28 --learning-rate 0.01 --device cpu --torch-threads 2`
- Dataset: FakeData

### Figures
- `results\identity_mapping\figures\fig_identity_loss_curve.png`
- `results\identity_mapping\figures\fig_identity_accuracy_curve.png`
- `results\identity_mapping\figures\fig_identity_layerwise_grad_norm.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_all.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_plainnet.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_resnet.png`
- `results\identity_mapping\figures\fig_identity_gradient_heatmap_preact_resnet.png`
- `results\identity_mapping\figures\fig_identity_lambda_ablation_grad_ratio.png`

### Tables
- `results\identity_mapping\tables\table_identity_model_comparison.csv`
- `results\identity_mapping\tables\table_identity_gradient_stability.csv`
- `results\identity_mapping\tables\table_identity_lambda_ablation.csv`
- `results\identity_mapping\gradients\identity_gradient_stats.csv`

## Generated Artifacts: Short-path Ensemble Lesion Study

- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\lesion_study.py --dataset auto --val-size 72 --batch-size 24 --device cpu --torch-threads 2 --drop-ratios 0,0.1,0.3,0.5,0.7 --random-seeds 0,1,2`
- Checkpoint: `results\identity_mapping\checkpoints\PreActResNet-56.pt`

### Figures
- `results\lesion_study\figures\fig_lesion_accuracy_vs_drop_ratio.png`
- `results\lesion_study\figures\fig_lesion_accuracy_drop_vs_drop_ratio.png`
- `results\lesion_study\figures\fig_lesion_random_vs_early_vs_late.png`
- `results\lesion_study\figures\fig_active_blocks_vs_accuracy.png`
- `results\lesion_study\figures\fig_lesion_heatmap.png`

### Tables
- `results\lesion_study\tables\table_lesion_accuracy.csv`
- `results\lesion_study\tables\table_lesion_drop_strategy_comparison.csv`
- `results\lesion_study\tables\table_active_blocks_summary.csv`

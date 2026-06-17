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

## Run: Short-path Ensemble Lesion Study

- Date: 2026-06-10T23:25:57
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\lesion_study.py --dataset auto --val-size 72 --batch-size 24 --device cpu --torch-threads 2 --drop-ratios 0,0.1,0.3,0.5,0.7 --random-seeds 0,1,2`
- Checkpoint: `results\identity_mapping\checkpoints\PreActResNet-56.pt`
- Dataset note: CIFAR-10 unavailable; using FakeData fallback: Dataset not found or corrupted. You can use download=True to download it
- Figures: `results\lesion_study\figures\fig_lesion_accuracy_vs_drop_ratio.png`, `results\lesion_study\figures\fig_lesion_accuracy_drop_vs_drop_ratio.png`, `results\lesion_study\figures\fig_lesion_random_vs_early_vs_late.png`, `results\lesion_study\figures\fig_active_blocks_vs_accuracy.png`, `results\lesion_study\figures\fig_lesion_heatmap.png`
- Tables: `results\lesion_study\tables\table_lesion_accuracy.csv`, `results\lesion_study\tables\table_lesion_drop_strategy_comparison.csv`, `results\lesion_study\tables\table_active_blocks_summary.csv`
- Status: completed

---

## Run: Smoke Test (cifar10)

- Date: 2026-06-10T23:43:51
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset cifar10 --download --output-tag cifar10 --epochs 2 --train-size 512 --val-size 256 --batch-size 64 --learning-rate 0.05 --device auto --torch-threads 2`
- Dataset: CIFAR10
- Fallback note: none
- Log: `results\smoke_cifar10\logs\smoke_test_cifar10_log.md`
- Gradient stats: `results\smoke_cifar10\gradients\smoke_cifar10_gradient_stats.csv`
- Figures: `results\smoke_cifar10\figures\smoke_loss_curve.png`, `results\smoke_cifar10\figures\smoke_accuracy_curve.png`, `results\smoke_cifar10\figures\smoke_layerwise_grad_norm.png`, `results\smoke_cifar10\figures\smoke_gradient_heatmap.png`
- Tables: `results\smoke_cifar10\tables\smoke_cifar10_model_comparison.csv`, `results\smoke_cifar10\tables\smoke_cifar10_gradient_stability.csv`, `results\smoke_cifar10\tables\smoke_cifar10_status.csv`
- Status: completed

---

## Run: Smoke Test (cifar10)

- Date: 2026-06-10T23:49:26
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset cifar10 --download --output-tag cifar10 --epochs 2 --train-size 512 --val-size 256 --batch-size 64 --learning-rate 0.05 --device auto --torch-threads 2`
- Dataset: CIFAR10
- Fallback note: none
- Log: `results\smoke_cifar10\logs\smoke_test_cifar10_log.md`
- Gradient stats: `results\smoke_cifar10\gradients\smoke_cifar10_gradient_stats.csv`
- Figures: `results\smoke_cifar10\figures\smoke_loss_curve.png`, `results\smoke_cifar10\figures\smoke_accuracy_curve.png`, `results\smoke_cifar10\figures\smoke_layerwise_grad_norm.png`, `results\smoke_cifar10\figures\smoke_gradient_heatmap.png`
- Tables: `results\smoke_cifar10\tables\smoke_cifar10_model_comparison.csv`, `results\smoke_cifar10\tables\smoke_cifar10_gradient_stability.csv`, `results\smoke_cifar10\tables\smoke_cifar10_status.csv`
- Status: completed

---

## Run: Identity Mapping Experiment (cifar10)

- Date: 2026-06-10T23:56:35
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite identity --dataset cifar10 --download --output-tag cifar10 --epochs 3 --train-size 2048 --val-size 512 --batch-size 128 --learning-rate 0.05 --device auto --torch-threads 2`
- Dataset: CIFAR10
- Fallback note: none
- Log: `results\identity_mapping_cifar10\logs\identity_mapping_cifar10_log.md`
- Gradient stats: `results\identity_mapping_cifar10\gradients\identity_cifar10_gradient_stats.csv`
- Figures: `results\identity_mapping_cifar10\figures\fig_identity_loss_curve.png`, `results\identity_mapping_cifar10\figures\fig_identity_accuracy_curve.png`, `results\identity_mapping_cifar10\figures\fig_identity_layerwise_grad_norm.png`, `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_all.png`, `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_plainnet.png`, `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_resnet.png`, `results\identity_mapping_cifar10\figures\fig_identity_gradient_heatmap_preact_resnet.png`, `results\identity_mapping_cifar10\figures\fig_identity_lambda_ablation_grad_ratio.png`
- Tables: `results\identity_mapping_cifar10\tables\table_identity_cifar10_model_comparison.csv`, `results\identity_mapping_cifar10\tables\table_identity_cifar10_gradient_stability.csv`, `results\identity_mapping_cifar10\tables\table_identity_cifar10_lambda_ablation.csv`
- Status: completed

---

## Run: Short-path Ensemble Lesion Study (cifar10)

- Date: 2026-06-11T00:01:39
- Command: `D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\lesion_study.py --dataset cifar10 --download --output-tag cifar10 --val-size 512 --batch-size 128 --device auto --torch-threads 2 --drop-ratios 0,0.1,0.3,0.5,0.7 --random-seeds 0,1,2`
- Checkpoint: `results\identity_mapping_cifar10\checkpoints\PreActResNet-56.pt`
- Dataset note: none
- Figures: `results\lesion_study_cifar10\figures\fig_lesion_accuracy_vs_drop_ratio.png`, `results\lesion_study_cifar10\figures\fig_lesion_accuracy_drop_vs_drop_ratio.png`, `results\lesion_study_cifar10\figures\fig_lesion_random_vs_early_vs_late.png`, `results\lesion_study_cifar10\figures\fig_active_blocks_vs_accuracy.png`, `results\lesion_study_cifar10\figures\fig_lesion_heatmap.png`
- Tables: `results\lesion_study_cifar10\tables\table_lesion_accuracy.csv`, `results\lesion_study_cifar10\tables\table_lesion_drop_strategy_comparison.csv`, `results\lesion_study_cifar10\tables\table_active_blocks_summary.csv`
- Status: completed

---

## Run: Identity Mapping Experiment (full_training)

- Date: 2026-06-14T01:13:32
- Command: `python -m harness.run_suite --suite identity --dataset cifar10 --data-root data\cifar10_verified --epochs 30 --train-size 50000 --val-size 10000 --batch-size 64 --num-workers 2 --learning-rate 0.025 --seed 0 --device cuda --torch-threads 2 --log-interval 100 --output-tag full_training --output-root results\full_training`
- Dataset: CIFAR10
- Fallback note: none
- Log: `results\full_training\logs\identity_mapping_full_training_log.md`
- Gradient stats: `results\full_training\gradients\identity_full_training_gradient_stats.csv`
- Figures: `results\full_training\figures\fig_identity_loss_curve.png`, `results\full_training\figures\fig_identity_accuracy_curve.png`, `results\full_training\figures\fig_identity_layerwise_grad_norm.png`, `results\full_training\figures\fig_identity_gradient_heatmap_all.png`, `results\full_training\figures\fig_identity_gradient_heatmap_plainnet.png`, `results\full_training\figures\fig_identity_gradient_heatmap_resnet.png`, `results\full_training\figures\fig_identity_gradient_heatmap_preact_resnet.png`, `results\full_training\figures\fig_identity_lambda_ablation_grad_ratio.png`
- Tables: `results\full_training\tables\table_identity_full_training_model_comparison.csv`, `results\full_training\tables\table_identity_full_training_gradient_stability.csv`, `results\full_training\tables\table_identity_full_training_lambda_ablation.csv`
- Status: completed

---

## Run: Short-path Ensemble Lesion Study (full_training)

- Date: 2026-06-14T01:18:00
- Command: `python -m harness.lesion_study --checkpoint results\full_training\checkpoints\PreActResNet-56.pt --dataset cifar10 --data-root data\cifar10_verified --val-size 10000 --batch-size 64 --num-workers 2 --device cuda --seed 0 --random-seeds 0,1,2 --drop-ratios 0,0.1,0.3,0.5,0.7 --torch-threads 2 --output-tag full_training --output-root results\full_training`
- Checkpoint: `results\full_training\checkpoints\PreActResNet-56.pt`
- Dataset note: none
- Figures: `results\full_training\figures\fig_lesion_accuracy_vs_drop_ratio.png`, `results\full_training\figures\fig_lesion_accuracy_drop_vs_drop_ratio.png`, `results\full_training\figures\fig_lesion_random_vs_early_vs_late.png`, `results\full_training\figures\fig_active_blocks_vs_accuracy.png`, `results\full_training\figures\fig_lesion_heatmap.png`
- Tables: `results\full_training\tables\table_lesion_accuracy.csv`, `results\full_training\tables\table_lesion_drop_strategy_comparison.csv`, `results\full_training\tables\table_active_blocks_summary.csv`
- Status: completed

---

## Run: Toy 1D Init Gradient Analysis

- Date: 2026-06-16T12:58:18
- Command: `python.exe scripts/run_toy1d_init_gradients.py --depths 2 24 50 --seeds 0 --device auto`
- Dataset: synthetic Toy 1D grid
- Output root: `results\toy1d_init_gradient_analysis`
- Status: completed

### Output Files

- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_init_gradient_curve_depth2.png`
- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_init_gradient_curve_depth24.png`
- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_init_gradient_curve_depth50.png`
- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_init_cov_plain_depth50.png`
- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_init_cov_resnet_depth50.png`
- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_init_cov_preact_depth50.png`
- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_init_acf_by_depth.png`
- `results\toy1d_init_gradient_analysis\figures\fig_toy1d_noise_reference.png`
- `results\toy1d_init_gradient_analysis\tables\table_toy1d_init_gradient_stats.csv`
- `results\toy1d_init_gradient_analysis\tables\table_toy1d_init_acf_summary.csv`

### Notes

- Toy 1D initialization gradients were computed in eval mode with no optimizer updates.
- The scripts can emit raw gradient vectors, one-row gradient matrices, covariance matrices, ACF values, and summary metrics; the organized Git-ready results retain figures, tables, metadata, and manifests while omitting raw `.npy` caches.
- Only seed 0 should be interpreted as a quick-run setting unless the script is rerun with more seeds.

---

## Run: Toy 1D Checkpoint Gradient Evolution

- Date: 2026-06-16T12:59:15
- Command: `python.exe scripts/run_toy1d_checkpoint_gradients.py --depth 50 --epochs 10 --checkpoint-epochs 0 1 5 10 final --seeds 0 --device auto`
- Dataset: synthetic Toy 1D regression
- Output root: `results\toy1d_checkpoint_gradient_evolution`
- Status: completed

### Output Files

- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_cov_plain_epoch0.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_cov_plain_epochfinal.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_gradient_evolution_plain.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_acf_evolution_plain.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_cov_resnet_epoch0.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_cov_resnet_epochfinal.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_gradient_evolution_resnet.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_acf_evolution_resnet.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_cov_preact_epoch0.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_cov_preact_epochfinal.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_gradient_evolution_preact.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_acf_evolution_preact.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_gradient_evolution_scaled_l1p0.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_acf_evolution_scaled_l1p0.png`
- `results\toy1d_checkpoint_gradient_evolution\figures\fig_toy1d_ckpt_acf_score_vs_epoch.png`
- `results\toy1d_checkpoint_gradient_evolution\tables\table_toy1d_checkpoint_gradient_stats.csv`
- `results\toy1d_checkpoint_gradient_evolution\tables\table_toy1d_checkpoint_acf_summary.csv`

### Notes

- Toy 1D checkpoint gradients were computed after saving epoch 0 before optimizer updates.
- Training uses a synthetic regression target y = sin(3x) + 0.3 cos(7x).
- Gradient analysis is read-only and uses eval mode for every saved checkpoint.

---

## Run: CIFAR-10 Init Gradient Analysis

- Date: 2026-06-16T13:00:00
- Command: `python.exe scripts/run_cifar10_init_gradients.py --depths 20 56 --batch-size 64 --num-batches 5 --seeds 0 --data-root data/cifar10_verified --device auto`
- Dataset: CIFAR10
- Output root: `results\cifar10_init_gradient_analysis`
- Status: completed

### Output Files

- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_corr_plain56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_corr_label_sorted_plain56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_corr_resnet56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_corr_label_sorted_resnet56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_corr_preact56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_corr_label_sorted_preact56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_relative_effective_rank_vs_depth.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_mean_gradient_norm_vs_depth.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_spatial_acf_plain56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_spatial_acf_resnet56.png`
- `results\cifar10_init_gradient_analysis\figures\fig_cifar10_init_spatial_acf_preact56.png`
- `results\cifar10_init_gradient_analysis\tables\table_cifar10_init_gradient_structure_summary.csv`
- `results\cifar10_init_gradient_analysis\tables\table_cifar10_init_relative_effective_rank.csv`
- `results\cifar10_init_gradient_analysis\tables\table_cifar10_init_mean_gradient_norm.csv`
- `results\cifar10_init_gradient_analysis\tables\table_cifar10_init_spatial_acf_summary.csv`

### Notes

- CIFAR-10 initialization analysis used real CIFAR-10 test samples and fixed minibatch indices.
- Gradient analysis was read-only: models were in eval mode and no optimizer step was called.
- This run used the runtime-limited setting from the plan: seed 0, depths 20 and 56, batch size 64, and 5 minibatches unless rerun with broader arguments.

---

## Run: CIFAR-10 Checkpoint Gradient Evolution

- Date: 2026-06-16T13:00:46
- Command: `python.exe scripts/run_cifar10_checkpoint_gradients.py --depth 56 --checkpoints 0 final --batch-size 64 --num-batches 5 --seeds 0 --data-root data/cifar10_verified --checkpoint-root results/full_training/checkpoints --metrics-csv results/full_training/tables/identity_full_training_metrics.csv --device auto`
- Dataset: CIFAR10
- Output root: `results\cifar10_checkpoint_gradient_evolution`
- Status: completed

### Output Files

- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_corr_plain_epoch0.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_corr_plain_epochfinal.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_corr_resnet_epoch0.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_corr_resnet_epochfinal.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_corr_preact_epoch0.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_corr_preact_epochfinal.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_relative_effective_rank_vs_epoch.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_mean_gradient_norm_vs_epoch.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_accuracy_vs_gradient_structure.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_spatial_acf_plain.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_spatial_acf_resnet.png`
- `results\cifar10_checkpoint_gradient_evolution\figures\fig_cifar10_ckpt_spatial_acf_preact.png`
- `results\cifar10_checkpoint_gradient_evolution\tables\table_cifar10_checkpoint_gradient_structure_summary.csv`
- `results\cifar10_checkpoint_gradient_evolution\tables\table_cifar10_checkpoint_relative_effective_rank.csv`
- `results\cifar10_checkpoint_gradient_evolution\tables\table_cifar10_checkpoint_mean_gradient_norm.csv`
- `results\cifar10_checkpoint_gradient_evolution\tables\table_cifar10_checkpoint_accuracy_gradient_relation.csv`

### Notes

- CIFAR-10 checkpoint analysis used fixed real CIFAR-10 test minibatches and per-sample input gradients.
- The run compares a fresh epoch 0 initialization with the existing full_training final checkpoints; intermediate epoch checkpoints were not present in the source checkpoint directory.
- Gradient analysis is read-only and does not update model parameters.

---

## Run: PlainNet Init Ablation Diagnostics

- Date: 2026-06-16T13:56:14
- Command: `python.exe scripts\run_plainnet_init_ablation.py --depth 56 --seed 0 --batch-size 64 --num-batches 5 --data-root data/cifar10_verified --device auto`
- Dataset: CIFAR10
- Output root: `results\plainnet_init_ablation`
- Status: completed

### Output Files

- `results\plainnet_init_ablation\vanishing_collapse_analysis\figures\fig_gradient_correlation_plainnet_default_init.png`
- `results\plainnet_init_ablation\vanishing_collapse_analysis\figures\fig_centered_gradient_correlation_plainnet_default_init.png`
- `results\plainnet_init_ablation\vanishing_collapse_analysis\figures\fig_gradient_correlation_plainnet_xavier_init.png`
- `results\plainnet_init_ablation\vanishing_collapse_analysis\figures\fig_centered_gradient_correlation_plainnet_xavier_init.png`
- `results\plainnet_init_ablation\shattered_gradient_analysis\figures\fig_gradient_correlation_plainnet_he_init.png`
- `results\plainnet_init_ablation\shattered_gradient_analysis\figures\fig_centered_gradient_correlation_plainnet_he_init.png`
- `results\plainnet_init_ablation\shattered_gradient_analysis\figures\fig_gradient_correlation_plainnet_he_init_bn.png`
- `results\plainnet_init_ablation\shattered_gradient_analysis\figures\fig_centered_gradient_correlation_plainnet_he_init_bn.png`
- `results\plainnet_init_ablation\shattered_gradient_analysis\figures\fig_gradient_correlation_plainnet_orthogonal_init.png`
- `results\plainnet_init_ablation\shattered_gradient_analysis\figures\fig_centered_gradient_correlation_plainnet_orthogonal_init.png`
- `results\plainnet_init_ablation\figures\fig_plainnet_ablation_input_gradient_norm.png`
- `results\plainnet_init_ablation\figures\fig_plainnet_ablation_gradient_std.png`
- `results\plainnet_init_ablation\figures\fig_plainnet_ablation_relative_effective_rank.png`
- `results\plainnet_init_ablation\figures\fig_plainnet_ablation_output_std.png`
- `results\plainnet_init_ablation\figures\fig_plainnet_ablation_activation_zero_ratio.png`
- `results\plainnet_init_ablation\figures\fig_plainnet_ablation_layerwise_gradient_norm.png`
- `results\plainnet_init_ablation\figures\fig_plainnet_ablation_spatial_acf.png`
- `results\plainnet_init_ablation\tables\table_plainnet_init_ablation_summary.csv`
- `results\plainnet_init_ablation\tables\table_plainnet_init_ablation_activation_stats.csv`
- `results\plainnet_init_ablation\tables\table_plainnet_init_ablation_layerwise_gradient_norms.csv`
- `results\plainnet_init_ablation\tables\table_plainnet_init_ablation_acf.csv`
- `results\plainnet_init_ablation\tables\table_plainnet_init_ablation_output_stats.csv`
- `results\plainnet_init_ablation\vanishing_collapse_analysis\tables\table_plainnet_gradient_collapse_summary.csv`
- `results\plainnet_init_ablation\shattered_gradient_analysis\tables\table_plainnet_shattered_gradient_candidates.csv`
- `results\plainnet_init_ablation\shattered_gradient_analysis\tables\table_main_plainnet_shattered_setting.csv`

### Notes

- PlainNet initialization/normalization variants were diagnosed before making any shattered-gradient claim.
- The diagnostic rule is explicit: if mean sample input-gradient norm < 1e-8 or input-gradient std < 1e-8, the result is labeled gradient_collapse.
- Only non-collapsed variants are placed in the shattered-gradient candidate table; collapsed variants are separated under vanishing_collapse_analysis.
- PlainNet-HeInit-BN is recorded as the main PlainNet setting for future shattered-gradient comparisons with ResNet.

---

## Run: Toy 1D Figure 2 Reproduction

- Date: 2026-06-17T15:58:20
- Command: `python.exe scripts/run_toy1d_fig2_reproduction.py --depths 2 4 10 24 50 --width 200 --num-points 256 --x-min -2 --x-max 2 --max-lag 15 --runs 20 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --output-dir results/toy1d_fig2_reproduction`
- Dataset: synthetic Toy 1D grid
- Output root: `results\toy1d_fig2_reproduction`
- Status: completed

### Output Files

- `results\toy1d_fig2_reproduction\figures\fig_toy1d_acf_figure2_reproduction.png`
- `results\toy1d_fig2_reproduction\figures\fig_toy1d_gradient_examples_depth50.png`
- `results\toy1d_fig2_reproduction\figures\fig_toy1d_gradient_norm_by_depth.png`
- `results\toy1d_fig2_reproduction\figures\fig_toy1d_gradient_std_by_depth.png`
- `results\toy1d_fig2_reproduction\tables\table_toy1d_acf_summary.csv`
- `results\toy1d_fig2_reproduction\tables\table_toy1d_acf_mean_std.csv`
- `results\toy1d_fig2_reproduction\tables\table_toy1d_collapse_diagnostics.csv`
- `results\toy1d_fig2_reproduction\artifact_manifest.csv`

### Notes

- This is an epoch-0 initialization-only analysis: no checkpoints, training loop, or optimizer step is used.
- Twenty-run averaging used seeds 0 to 19.
- No model-depth setting had collapsed runs under the 1e-8 norm/std rule.
- At depth 50, lag-5 mean ACF was feedforward=-0.05423, ResNet beta=1.0=0.6047, and ResNet beta=0.1=0.8404.
- The feedforward ACF decayed faster than the beta=1.0 ResNet at the comparison depth.
- The beta=0.1 ResNet preserved stronger local ACF than beta=1.0 at the comparison depth.
- The implementation is reproduction-inspired rather than exact: model layers use per-sample feature centering/normalization to keep df(x_i)/dx_i pointwise well-defined.

---

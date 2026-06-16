# Results Directory

Experiment artifacts are organized here:

- `logs/`: text logs.
- `checkpoints/`: model checkpoints, ignored by default for `.pt` and `.pth`.
- `gradients/`: gradient statistics CSV files.
- `figures/`: generated PNG figures.
- `tables/`: generated CSV summary tables.
- `runs/`: per-run metrics, summaries, and copied configs.
- `full_training/`: full CIFAR-10 training figures, tables, logs, gradients, and lesion masks. Model checkpoints remain ignored.
- `toy1d_init_gradient_analysis/`: Toy 1D initialization gradient-structure outputs from the Gradient Structure plan.
- `toy1d_checkpoint_gradient_evolution/`: Toy 1D checkpoint gradient-evolution outputs from the Gradient Structure plan.
- `cifar10_init_gradient_analysis/`: CIFAR-10 initialization input-gradient structure outputs from the Gradient Structure plan.
- `cifar10_checkpoint_gradient_evolution/`: CIFAR-10 checkpoint input-gradient evolution outputs from the Gradient Structure plan.
- `plainnet_init_ablation/`: PlainNet initialization and normalization diagnostics separating gradient collapse from shattered-gradient candidates.
- `gradient_structure_plan_summary.md`: concise summary of the Gradient Structure plan execution and limitations.

Large reproducible intermediates are intentionally excluded from Git: `*.npy` raw gradient arrays and `*.pt`/`*.pth` model checkpoints.

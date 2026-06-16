# Full Training Results

This folder is reserved for full CIFAR-10 training artifacts.

- `logs/`: full-training logs and command records.
- `checkpoints/`: saved model checkpoints.
- `gradients/`: gradient statistics CSV files.
- `figures/`: generated figures from the full run.
- `tables/`: summary tables and run metadata.
- `masks/`: residual branch lesion masks generated after full training.

Current status: full training has not been run yet. Existing CIFAR-10 artifacts are subset experiments with a small number of epochs.

Recommended 2GB GPU command on this machine:

```bash
conda run -n resnet-gradient-path-study python -B scripts/run_full_training_gpu_2gb.py
```

This runs full CIFAR-10 identity/shortcut training on CUDA with `batch_size=32`, `num_workers=2`, `learning_rate=0.025`, `epochs=30`, and then runs full-test-set lesion validation unless `--skip-lesion` is passed. Training progress is printed every 100 batches by default.

Use `--dry-run` to check CUDA, dataset paths, and the generated commands without launching training. If CUDA runs out of memory, rerun with `--batch-size 16 --learning-rate 0.0125`. Use `--log-interval 50` for more frequent progress or `--log-interval 0` to disable batch-level printing.

During training, confirm GPU execution from log lines containing `model_device=cuda:0` and `cuda_mem=...`. If GPU utilization stays low but memory is allocated, the run is using CUDA but is likely bottlenecked by CPU data loading or small batch size. Try `--num-workers 4` or, if memory allows, `--batch-size 64 --learning-rate 0.05`.

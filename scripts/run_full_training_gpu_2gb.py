"""Run full CIFAR-10 identity/shortcut experiments on a 2GB CUDA GPU.

Defaults are tuned for the local NVIDIA GeForce MX450:

- batch size 32 to stay within roughly 2GB VRAM
- 2 DataLoader workers plus pinned memory to reduce CPU input stalls
- learning rate 0.025, scaled down from the project plan's 0.1 for batch size 128
- full CIFAR-10 train/test sizes: 50000/10000
- outputs under results/full_training
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_environment(torch_threads: int) -> None:
    """Set conservative CUDA/PyTorch environment variables before torch import."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))


def ensure_output_dirs(output_root: Path) -> None:
    for name in ("logs", "checkpoints", "gradients", "figures", "tables", "masks"):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def ensure_cifar10(data_root: Path, allow_download: bool) -> None:
    expected_batch = data_root / "cifar-10-batches-py" / "data_batch_1"
    if expected_batch.exists():
        return

    archive_path = PROJECT_ROOT / "data" / "cifar-10-python.tar.gz"
    if archive_path.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(data_root)
        return

    if not allow_download:
        raise FileNotFoundError(
            "CIFAR-10 is missing. Put data/cifar-10-python.tar.gz in the project, "
            "or rerun with --allow-download."
        )


def verify_cuda() -> None:
    import torch

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script is configured for GPU training.")

    props = torch.cuda.get_device_properties(0)
    print("cuda_device:", torch.cuda.get_device_name(0))
    print("total_vram_gb:", round(props.total_memory / 1024**3, 2))


def verify_full_cifar10(
    data_root: Path,
    batch_size: int,
    seed: int,
    allow_download: bool,
    num_workers: int,
) -> None:
    from src.data.cifar import build_small_image_loaders

    _, _, info = build_small_image_loaders(
        dataset="cifar10",
        root=str(data_root),
        batch_size=batch_size,
        train_size=50000,
        val_size=10000,
        seed=seed,
        download=allow_download,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(info)
    if info.name != "CIFAR10" or info.train_size != 50000 or info.val_size != 10000:
        raise RuntimeError(f"Expected full CIFAR10 50000/10000, got {info!r}")


def run_identity_training(args: argparse.Namespace) -> None:
    from harness.run_suite import main as run_suite_main

    train_args = [
        "--suite",
        "identity",
        "--dataset",
        "cifar10",
        "--data-root",
        str(args.data_root),
        "--epochs",
        str(args.epochs),
        "--train-size",
        "50000",
        "--val-size",
        "10000",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--torch-threads",
        str(args.torch_threads),
        "--log-interval",
        str(args.log_interval),
        "--output-tag",
        "full_training",
        "--output-root",
        str(args.output_root),
    ]
    if args.allow_download:
        train_args.append("--download")

    print("Training command:")
    print("python -m harness.run_suite " + " ".join(train_args))
    if not args.dry_run:
        original_argv = sys.argv[:]
        sys.argv = ["python", "-m", "harness.run_suite", *train_args]
        try:
            run_suite_main(train_args)
        finally:
            sys.argv = original_argv


def run_lesion_validation(args: argparse.Namespace) -> None:
    from harness.lesion_study import main as lesion_main

    checkpoint = args.output_root / "checkpoints" / "PreActResNet-56.pt"
    lesion_args = [
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        "cifar10",
        "--data-root",
        str(args.data_root),
        "--val-size",
        "10000",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--random-seeds",
        "0,1,2",
        "--drop-ratios",
        "0,0.1,0.3,0.5,0.7",
        "--torch-threads",
        str(args.torch_threads),
        "--output-tag",
        "full_training",
        "--output-root",
        str(args.output_root),
    ]
    if args.allow_download:
        lesion_args.append("--download")

    print("Lesion validation command:")
    print("python -m harness.lesion_study " + " ".join(lesion_args))
    if args.dry_run:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint was not created: {checkpoint}")
    original_argv = sys.argv[:]
    sys.argv = ["python", "-m", "harness.lesion_study", *lesion_args]
    try:
        lesion_main(lesion_args)
    finally:
        sys.argv = original_argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10_verified"))
    parser.add_argument("--output-root", type=Path, default=Path("results/full_training"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-lesion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    configure_environment(args.torch_threads)
    ensure_output_dirs(args.output_root)
    ensure_cifar10(args.data_root, args.allow_download)
    verify_cuda()
    verify_full_cifar10(
        args.data_root,
        args.batch_size,
        args.seed,
        args.allow_download,
        args.num_workers,
    )
    run_identity_training(args)
    if not args.skip_lesion:
        run_lesion_validation(args)
    if args.dry_run:
        print("Dry run complete. No training was launched.")
    else:
        print(f"Full GPU experiment complete. Outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

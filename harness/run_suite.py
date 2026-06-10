"""Experiment suite entry point."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .mini_experiments import run_identity, run_smoke


SUITES = ("smoke", "identity")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a suite of experiments.")
    parser.add_argument("--suite", choices=SUITES, default="smoke")
    parser.add_argument("--dataset", choices=["auto", "cifar10", "fake"], default="auto")
    parser.add_argument("--download", action="store_true", help="Allow CIFAR-10 download before FakeData fallback.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for result directories and table names, e.g. cifar10.",
    )
    args = parser.parse_args(argv)

    if args.suite == "smoke":
        run_smoke(args)
    elif args.suite == "identity":
        run_identity(args)
    else:
        parser.error(f"Unknown suite: {args.suite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

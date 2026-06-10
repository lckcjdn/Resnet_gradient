"""Experiment suite entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .run_experiment import main as run_one


SUITES = {
    "quick": ["configs/plain56.yaml", "configs/resnet56.yaml", "configs/preact_resnet56.yaml"],
    "full": [
        "configs/plain56.yaml",
        "configs/resnet56.yaml",
        "configs/preact_resnet56.yaml",
        "configs/scaled_lambda_05.yaml",
        "configs/scaled_lambda_09.yaml",
        "configs/scaled_lambda_10.yaml",
        "configs/scaled_lambda_11.yaml",
        "configs/lesion.yaml",
    ],
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a suite of experiments.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="quick")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for config in SUITES[args.suite]:
        if not Path(config).exists():
            parser.error(f"Config does not exist: {config}")
        run_args = ["--config", config]
        if args.dry_run:
            run_args.append("--dry-run")
        run_one(run_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

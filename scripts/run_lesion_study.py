"""Placeholder entry point for lesion studies."""

from __future__ import annotations

from harness.run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/lesion.yaml", "--dry-run"]))

"""Artifact path management for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


RESULT_DIRS = [
    "results/logs",
    "results/checkpoints",
    "results/gradients",
    "results/figures",
    "results/tables",
    "results/runs",
]


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    metrics_path: Path
    gradient_path: Path
    summary_path: Path


def ensure_result_dirs(project_root: PathLike = ".") -> None:
    """Create the standard result directories."""
    root = Path(project_root)
    for relative in RESULT_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)


def create_run_id(model_name: str, depth: int, seed: int) -> str:
    """Create a run id like YYYYMMDD-HHMMSS_model_depth_seed0."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = model_name.lower().replace(" ", "_").replace("-", "_")
    return f"{timestamp}_{safe_model}{depth}_seed{seed}"


def prepare_run_paths(
    model_name: str,
    depth: int,
    seed: int,
    project_root: PathLike = ".",
) -> RunPaths:
    """Create run directory paths without launching an experiment."""
    root = Path(project_root)
    ensure_result_dirs(root)
    run_id = create_run_id(model_name, depth, seed)
    run_dir = root / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.csv",
        gradient_path=root / "results" / "gradients" / f"{run_id}_gradient_stats.csv",
        summary_path=run_dir / "run_summary.md",
    )

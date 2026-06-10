"""Documentation update helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Union

from src.analysis.table_writer import csv_to_markdown


PathLike = Union[str, Path]


def append_experiment_log(project_root: PathLike, run_record: dict[str, str]) -> None:
    """Append a run record to docs/experiment_log.md."""
    root = Path(project_root)
    log_path = root / "docs" / "experiment_log.md"
    lines = [
        "",
        f"## Run: {run_record.get('run_id', 'unknown')}",
        "",
        f"- Date: {run_record.get('date', datetime.now().isoformat(timespec='seconds'))}",
        f"- Git commit: {run_record.get('git_commit', 'not recorded')}",
        f"- Config: {run_record.get('config', '')}",
        f"- Model: {run_record.get('model', '')}",
        f"- Dataset: {run_record.get('dataset', '')}",
        f"- Device: {run_record.get('device', '')}",
        f"- Epochs: {run_record.get('epochs', '')}",
        f"- Status: {run_record.get('status', '')}",
        f"- Best test accuracy: {run_record.get('best_test_accuracy', '')}",
        f"- Final train loss: {run_record.get('final_train_loss', '')}",
        f"- Gradient stability ratio: {run_record.get('gradient_stability_ratio', '')}",
        f"- Output figures: {run_record.get('output_figures', '')}",
        f"- Output tables: {run_record.get('output_tables', '')}",
        f"- Notes: {run_record.get('notes', '')}",
        "",
        "---",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def convert_result_table(project_root: PathLike, csv_name: str, markdown_name: str) -> None:
    """Convert a result CSV to docs/tables Markdown."""
    root = Path(project_root)
    csv_to_markdown(root / "results" / "tables" / csv_name, root / "docs" / "tables" / markdown_name)

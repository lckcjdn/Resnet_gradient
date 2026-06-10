"""CSV to Markdown table helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def csv_to_markdown(csv_path: PathLike, markdown_path: PathLike) -> None:
    """Convert a CSV file to a simple Markdown table."""
    csv_path = Path(csv_path)
    markdown_path = Path(markdown_path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        markdown_path.write_text("", encoding="utf-8")
        return

    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

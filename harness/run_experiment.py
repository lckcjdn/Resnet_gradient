"""Single-experiment entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .artifact_manager import ensure_result_dirs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one configured experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths without launching training.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config does not exist: {config_path}")

    ensure_result_dirs(Path.cwd())
    if args.dry_run:
        print(f"Dry run OK: {config_path}")
        return 0

    raise NotImplementedError(
        "Full training is intentionally deferred. Implement trainer integration before running."
    )


if __name__ == "__main__":
    raise SystemExit(main())

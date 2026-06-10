"""Create standard output directories."""

from __future__ import annotations

from harness.artifact_manager import ensure_result_dirs


def main() -> int:
    ensure_result_dirs()
    print("Project directories are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

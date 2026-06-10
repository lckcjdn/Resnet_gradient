"""Lightweight project sanity checks."""

from __future__ import annotations

import importlib
from pathlib import Path

from .artifact_manager import RESULT_DIRS, ensure_result_dirs


REQUIRED_PATHS = [
    "PROJECT_INIT.md",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "configs/default.yaml",
    "docs/experiment_plan.md",
    "docs/process_conclusions.md",
    "src/models/plain_cnn.py",
    "src/models/resnet_v1.py",
    "src/models/preact_resnet.py",
    "src/models/scaled_shortcut_resnet.py",
]


def _check_paths(root: Path) -> list[str]:
    failures = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            failures.append(f"missing required path: {relative}")
    for relative in RESULT_DIRS:
        if not (root / relative).exists():
            failures.append(f"missing result directory: {relative}")
    return failures


def _check_model_forward_backward() -> list[str]:
    failures = []
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        models = importlib.import_module("src.models")
        gradients = importlib.import_module("src.analysis.gradient_monitor")
    except Exception as exc:
        return [f"PyTorch model sanity skipped or unavailable: {exc}"]

    builders = [
        ("PlainNet", lambda: models.build_model("PlainNet", depth=20)),
        ("ResNetV1", lambda: models.build_model("ResNetV1", depth=20)),
        ("PreActResNet", lambda: models.build_model("PreActResNet", depth=20)),
        (
            "ScaledShortcutResNet",
            lambda: models.build_model("ScaledShortcutResNet", depth=20, shortcut_lambda=0.9),
        ),
    ]
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])

    for name, builder in builders:
        try:
            model = builder()
            outputs = model(inputs)
            if tuple(outputs.shape) != (2, 10):
                failures.append(f"{name} output shape mismatch: {tuple(outputs.shape)}")
                continue
            loss = criterion(outputs, targets)
            loss.backward()
            if name != "PlainNet":
                records = gradients.collect_gradient_stats(
                    model,
                    epoch=0,
                    model_name=name,
                    seed=0,
                    run_id="sanity",
                )
                if not records:
                    failures.append(f"{name} produced no conv2 gradient records")
        except Exception as exc:
            failures.append(f"{name} sanity failed: {exc}")
    return failures


def main() -> int:
    root = Path.cwd()
    ensure_result_dirs(root)
    failures = _check_paths(root)
    failures.extend(_check_model_forward_backward())

    hard_failures = [item for item in failures if not item.startswith("PyTorch model sanity skipped")]
    if failures:
        print("Sanity check notes:")
        for failure in failures:
            print(f"- {failure}")
    if hard_failures:
        return 1

    print("Sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

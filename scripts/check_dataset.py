"""Verify that CIFAR-10 is available and record dataset provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gradient_structure_common import append_text, write_json


def verify_cifar10(root: str) -> dict:
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train = datasets.CIFAR10(root=root, train=True, transform=transform, download=False)
    test = datasets.CIFAR10(root=root, train=False, transform=transform, download=False)
    sample_image, sample_label = train[0]
    return {
        "dataset": "CIFAR10",
        "root": root,
        "train_samples": len(train),
        "test_samples": len(test),
        "classes": len(train.classes),
        "class_names": train.classes,
        "image_shape": list(sample_image.shape),
        "first_label": int(sample_label),
        "status": "verified",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cifar10_verified")
    parser.add_argument("--output", type=Path, default=Path("results/cifar10_init_gradient_analysis/metadata/dataset_check.json"))
    args = parser.parse_args(argv)

    info = verify_cifar10(args.data_root)
    write_json(args.output, info)
    append_text(
        ROOT / "docs" / "dataset_log.md",
        "\n".join(
            [
                "",
                "## CIFAR-10 Dataset Verification",
                "",
                f"- Root: `{args.data_root}`",
                f"- Train samples: {info['train_samples']}",
                f"- Test samples: {info['test_samples']}",
                f"- Classes: {info['classes']}",
                f"- Image shape: {info['image_shape']}",
                f"- Metadata: `{args.output}`",
                "- Status: verified",
                "",
            ]
        ),
    )
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download CIFAR-10 if it is missing, then verify the dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_dataset import verify_cifar10
from scripts.gradient_structure_common import append_text, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cifar10_verified")
    parser.add_argument("--output", type=Path, default=Path("results/cifar10_init_gradient_analysis/metadata/dataset_download.json"))
    args = parser.parse_args(argv)

    downloaded = False
    try:
        info = verify_cifar10(args.data_root)
    except Exception:
        from torchvision import datasets, transforms

        datasets.CIFAR10(root=args.data_root, train=True, transform=transforms.ToTensor(), download=True)
        datasets.CIFAR10(root=args.data_root, train=False, transform=transforms.ToTensor(), download=True)
        downloaded = True
        info = verify_cifar10(args.data_root)

    payload = {**info, "download_attempted": downloaded}
    write_json(args.output, payload)
    append_text(
        ROOT / "docs" / "dataset_log.md",
        "\n".join(
            [
                "",
                "## CIFAR-10 Download/Availability Check",
                "",
                f"- Root: `{args.data_root}`",
                f"- Download attempted: {downloaded}",
                f"- Train samples: {info['train_samples']}",
                f"- Test samples: {info['test_samples']}",
                f"- Metadata: `{args.output}`",
                "- Status: verified",
                "",
            ]
        ),
    )
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

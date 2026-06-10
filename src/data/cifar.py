"""CIFAR and FakeData dataloader helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    train_size: int
    val_size: int
    root: str
    fallback_reason: str = ""


def build_cifar10_loaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
    augment: bool = True,
):
    """Create CIFAR-10 train and test loaders.

    Imports are local so project sanity checks can run before optional dependencies
    are installed.
    """
    import torch
    from torchvision import datasets, transforms

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=download)
    test_set = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=download)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_small_image_loaders(
    dataset: str = "auto",
    root: str = "data",
    batch_size: int = 32,
    train_size: int = 128,
    val_size: int = 64,
    seed: int = 0,
    download: bool = False,
    num_workers: int = 0,
):
    """Build small CIFAR-10 loaders or deterministic FakeData fallback loaders."""
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    root_path = Path(root)
    fallback_reason = ""
    use_cifar = dataset.lower() in {"auto", "cifar10", "cifar-10"}

    if use_cifar:
        try:
            train_set = datasets.CIFAR10(
                root=root,
                train=True,
                transform=transform,
                download=download,
            )
            val_set = datasets.CIFAR10(
                root=root,
                train=False,
                transform=transform,
                download=download,
            )
            dataset_name = "CIFAR10"
        except Exception as exc:
            fallback_reason = f"CIFAR-10 unavailable; using FakeData fallback: {exc}"
            train_set = None
            val_set = None
    else:
        fallback_reason = "FakeData explicitly requested."
        train_set = None
        val_set = None

    if train_set is None or val_set is None:
        train_set = datasets.FakeData(
            size=train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=seed * 1000,
        )
        val_set = datasets.FakeData(
            size=val_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=seed * 1000 + 500,
        )
        dataset_name = "FakeData"
    else:
        train_count = min(train_size, len(train_set))
        val_count = min(val_size, len(val_set))
        generator = torch.Generator().manual_seed(seed)
        train_indices = torch.randperm(len(train_set), generator=generator)[:train_count].tolist()
        val_indices = torch.randperm(len(val_set), generator=generator)[:val_count].tolist()
        train_set = torch.utils.data.Subset(train_set, train_indices)
        val_set = torch.utils.data.Subset(val_set, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    info = DatasetInfo(
        name=dataset_name,
        train_size=len(train_set),
        val_size=len(val_set),
        root=str(root_path),
        fallback_reason=fallback_reason,
    )
    return train_loader, val_loader, info

"""CIFAR dataloader helpers."""

from __future__ import annotations


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

"""Scaled shortcut ResNet for lambda ablation."""

from __future__ import annotations

from .resnet_v1 import ResNetCifar


class ScaledShortcutResNet(ResNetCifar):
    """Standard CIFAR ResNet with `lambda * shortcut(x)` in every block."""

    def __init__(
        self,
        depth: int = 56,
        num_classes: int = 10,
        shortcut_lambda: float = 1.0,
    ):
        super().__init__(
            depth=depth,
            num_classes=num_classes,
            shortcut_lambda=shortcut_lambda,
        )


def scaled_shortcut_resnet56(
    num_classes: int = 10,
    shortcut_lambda: float = 1.0,
) -> ScaledShortcutResNet:
    return ScaledShortcutResNet(
        depth=56,
        num_classes=num_classes,
        shortcut_lambda=shortcut_lambda,
    )

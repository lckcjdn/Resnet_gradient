"""Model builders for CIFAR-style PlainNet and ResNet variants."""

from .plain_cnn import PlainNet, plainnet56
from .preact_resnet import PreActResNetCifar, preact_resnet56
from .resnet_v1 import ResNetCifar, resnet56
from .scaled_shortcut_resnet import ScaledShortcutResNet, scaled_shortcut_resnet56
from .toy1d_feedforward import Toy1DFeedForward
from .toy1d_resnet import Toy1DResNet


def build_model(name: str, depth: int = 56, num_classes: int = 10, shortcut_lambda: float = 1.0):
    """Build a model by config name."""
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized in {"plainnet", "plaincnn"}:
        return PlainNet(depth=depth, num_classes=num_classes)
    if normalized in {"resnetv1", "standardresnet", "resnet"}:
        return ResNetCifar(depth=depth, num_classes=num_classes)
    if normalized in {"preactresnet", "preactivationresnet"}:
        return PreActResNetCifar(depth=depth, num_classes=num_classes)
    if normalized in {"scaledshortcutresnet", "scaledresnet"}:
        return ScaledShortcutResNet(
            depth=depth,
            num_classes=num_classes,
            shortcut_lambda=shortcut_lambda,
        )
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "PlainNet",
    "PreActResNetCifar",
    "ResNetCifar",
    "ScaledShortcutResNet",
    "Toy1DFeedForward",
    "Toy1DResNet",
    "build_model",
    "plainnet56",
    "preact_resnet56",
    "resnet56",
    "scaled_shortcut_resnet56",
]

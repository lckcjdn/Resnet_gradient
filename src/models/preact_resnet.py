"""Pre-activation CIFAR ResNet."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .resnet_v1 import _blocks_per_stage


class PreActBlock(nn.Module):
    """BN-ReLU-Conv-BN-ReLU-Conv-Add residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        shortcut_lambda: float = 1.0,
    ):
        super().__init__()
        self.shortcut_lambda = float(shortcut_lambda)
        self.residual_mask = 1.0
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def set_residual_mask(self, value: float) -> None:
        self.residual_mask = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preactivated = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(preactivated) * self.shortcut_lambda
        residual = self.conv1(preactivated)
        residual = self.conv2(F.relu(self.bn2(residual), inplace=True))
        return shortcut + self.residual_mask * residual


class PreActResNetCifar(nn.Module):
    """CIFAR PreAct ResNet with no post-addition activation inside blocks."""

    def __init__(
        self,
        depth: int = 56,
        num_classes: int = 10,
        shortcut_lambda: float = 1.0,
    ):
        super().__init__()
        blocks = _blocks_per_stage(depth)
        self.depth = depth
        self.num_classes = num_classes
        self.shortcut_lambda = float(shortcut_lambda)
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(16, blocks, stride=1)
        self.stage2 = self._make_stage(32, blocks, stride=2)
        self.stage3 = self._make_stage(64, blocks, stride=2)
        self.bn_last = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [
            PreActBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                shortcut_lambda=self.shortcut_lambda,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                PreActBlock(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    shortcut_lambda=self.shortcut_lambda,
                )
            )
        return nn.Sequential(*layers)

    def iter_blocks(self) -> Iterable[PreActBlock]:
        for stage in (self.stage1, self.stage2, self.stage3):
            yield from stage

    def set_block_masks(self, masks: Iterable[float]) -> None:
        for block, mask in zip(self.iter_blocks(), masks):
            block.set_residual_mask(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn_last(x), inplace=True)
        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, 1)
        return self.fc(x)


def preact_resnet56(num_classes: int = 10) -> PreActResNetCifar:
    return PreActResNetCifar(depth=56, num_classes=num_classes)

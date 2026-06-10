"""Plain CIFAR CNN baseline with no shortcut connections."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _blocks_per_stage(depth: int) -> int:
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR depth should satisfy depth = 6n + 2.")
    return (depth - 2) // 6


class PlainBlock(nn.Module):
    """Two Conv-BN-ReLU layers without a residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class PlainNet(nn.Module):
    """CIFAR PlainNet that mirrors the stage layout of CIFAR ResNet."""

    def __init__(self, depth: int = 56, num_classes: int = 10):
        super().__init__()
        blocks = _blocks_per_stage(depth)
        self.depth = depth
        self.num_classes = num_classes
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, blocks, stride=1)
        self.stage2 = self._make_stage(32, blocks, stride=2)
        self.stage3 = self._make_stage(64, blocks, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [PlainBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(PlainBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, 1)
        return self.fc(x)


def plainnet56(num_classes: int = 10) -> PlainNet:
    return PlainNet(depth=56, num_classes=num_classes)

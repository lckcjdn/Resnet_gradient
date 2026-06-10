"""Standard post-activation CIFAR ResNet."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def _blocks_per_stage(depth: int) -> int:
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR ResNet depth should satisfy depth = 6n + 2.")
    return (depth - 2) // 6


class BasicBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN-Add-ReLU residual block."""

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

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def set_residual_mask(self, value: float) -> None:
        self.residual_mask = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.bn1(self.conv1(x)), inplace=True)
        residual = self.bn2(self.conv2(residual))
        shortcut = self.shortcut(x) * self.shortcut_lambda
        out = shortcut + self.residual_mask * residual
        return F.relu(out, inplace=True)


class ResNetCifar(nn.Module):
    """CIFAR ResNet supporting depths such as 20, 32, and 56."""

    block_cls = BasicBlock

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
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, blocks, stride=1)
        self.stage2 = self._make_stage(32, blocks, stride=2)
        self.stage3 = self._make_stage(64, blocks, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [
            self.block_cls(
                self.in_channels,
                out_channels,
                stride=stride,
                shortcut_lambda=self.shortcut_lambda,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                self.block_cls(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    shortcut_lambda=self.shortcut_lambda,
                )
            )
        return nn.Sequential(*layers)

    def iter_blocks(self) -> Iterable[BasicBlock]:
        for stage in (self.stage1, self.stage2, self.stage3):
            yield from stage

    def set_block_masks(self, masks: Iterable[float]) -> None:
        for block, mask in zip(self.iter_blocks(), masks):
            block.set_residual_mask(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet56(num_classes: int = 10) -> ResNetCifar:
    return ResNetCifar(depth=56, num_classes=num_classes)

"""Toy 1D feedforward networks for initialization-gradient ACF experiments."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FeatureCenteredReLU(nn.Module):
    """ReLU followed by per-sample feature centering."""

    def __init__(self, mean_center: bool = True) -> None:
        super().__init__()
        self.mean_center = bool(mean_center)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.relu(x)
        if not self.mean_center:
            return activated
        return activated - activated.mean(dim=1, keepdim=True)


def initialize_he_linear(layer: nn.Linear, *, random_bias: bool = True) -> None:
    """Use approximately 2 / fan_in variance for linear weights."""
    fan_in = layer.weight.shape[1]
    std = math.sqrt(2.0 / float(fan_in))
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        if random_bias:
            nn.init.normal_(layer.bias, mean=0.0, std=std)
        else:
            nn.init.zeros_(layer.bias)


class Toy1DFeedForward(nn.Module):
    """Fully connected ReLU MLP mapping R to R on a fixed 1D grid."""

    def __init__(self, depth: int, width: int = 200, mean_center: bool = True) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Toy1DFeedForward depth must be at least 1")
        self.depth = int(depth)
        self.width = int(width)
        self.mean_center = bool(mean_center)

        hidden_layers = []
        in_features = 1
        for _ in range(self.depth):
            layer = nn.Linear(in_features, self.width)
            initialize_he_linear(layer)
            hidden_layers.append(layer)
            in_features = self.width
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.activation = FeatureCenteredReLU(mean_center=self.mean_center)
        self.output = nn.Linear(self.width, 1)
        initialize_he_linear(self.output, random_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output(out)

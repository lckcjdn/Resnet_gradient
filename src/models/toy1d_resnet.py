"""Toy 1D residual MLPs for initialization-gradient ACF experiments."""

from __future__ import annotations

import torch
from torch import nn

from .toy1d_feedforward import FeatureCenteredReLU, initialize_he_linear


class Toy1DResidualBlock(nn.Module):
    """Residual block with x_{l+1} = x_l + beta * F_l(x_l)."""

    def __init__(self, width: int, beta: float = 1.0, mean_center: bool = True) -> None:
        super().__init__()
        self.beta = float(beta)
        self.mean_center = bool(mean_center)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.activation = FeatureCenteredReLU(mean_center=self.mean_center)
        initialize_he_linear(self.fc1)
        initialize_he_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.norm1(self.fc1(x)))
        residual = self.norm2(self.fc2(residual))
        return x + self.beta * residual


class Toy1DResNet(nn.Module):
    """Fully connected residual MLP mapping R to R."""

    def __init__(
        self,
        depth: int,
        width: int = 200,
        beta: float = 1.0,
        mean_center: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Toy1DResNet depth must be at least 1")
        self.depth = int(depth)
        self.width = int(width)
        self.beta = float(beta)
        self.mean_center = bool(mean_center)

        self.input = nn.Linear(1, self.width)
        initialize_he_linear(self.input)
        self.input_activation = FeatureCenteredReLU(mean_center=self.mean_center)
        self.blocks = nn.ModuleList(
            [Toy1DResidualBlock(self.width, beta=self.beta, mean_center=self.mean_center) for _ in range(self.depth)]
        )
        self.output = nn.Linear(self.width, 1)
        initialize_he_linear(self.output, random_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_activation(self.input(x))
        for block in self.blocks:
            out = block(out)
        return self.output(out)

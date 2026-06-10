# Terms For Presentation

## Shortcut Connection

A connection that bypasses one or more layers. In ResNet, the common form is `y = F(x) + x`.

## Identity Mapping

A mapping that returns the input unchanged. Identity shortcuts give signals and gradients a direct route.

## Residual Branch

The nonlinear transformation `F(x)` inside a residual block, usually implemented with convolution, batch normalization, and activation layers.

## Pre-Activation

A residual block design that applies normalization and activation before convolution and avoids an extra activation after addition.

## Gradient Norm

A scalar summary of gradient magnitude. This project records layer-wise gradient norms to compare propagation stability.

## Gradient Stability Ratio

The ratio between average shallow-layer gradient norm and average deep-layer gradient norm. Extreme values suggest uneven gradient distribution.

## Lesion

A controlled intervention that disables selected residual branches to measure model sensitivity.

# Identity Mapping

## Core Idea

An identity mapping returns its input unchanged:

```text
h(x) = x
```

In a residual block:

```text
x_{l+1} = x_l + F_l(x_l)
```

the shortcut contributes an identity term. During backpropagation:

```text
dL/dx_l = dL/dx_{l+1} * (I + dF_l/dx_l)
```

The `I` term is the direct gradient path from the identity shortcut.

## Pre-Activation ResNet

Pre-activation blocks move batch normalization and ReLU before the convolutions:

```text
x -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> + x
```

This keeps the post-addition path closer to identity than the standard post-activation block:

```text
x -> Conv -> BN -> ReLU -> Conv -> BN -> + x -> ReLU
```

## Scaled Shortcuts

Scaled shortcut experiments replace the identity path with:

```text
x_{l+1} = lambda * x_l + F_l(x_l)
```

When `lambda = 1`, the shortcut is identity. Values below or above 1 weaken or amplify the direct path and can affect gradient stability.

# ResNet Background

## Why Deeper Plain Networks Can Be Hard To Train

In principle, a deeper network should be at least as expressive as a shallower one. Extra layers could learn identity mappings and preserve the behavior of a shallower model.

In practice, plain deep convolutional networks can show optimization degradation:

- training loss decreases slowly or stalls;
- training accuracy can be worse than a shallower baseline;
- shallow-layer gradients can become very small or unstable;
- adding depth can make optimization harder even before overfitting is considered.

## Degradation Problem

The degradation problem means that increasing depth can increase both training error and test error. It differs from overfitting, where training error is low but test error is high.

## Residual Learning

Instead of directly learning a target mapping `H(x)`, a residual block learns:

```text
F(x) = H(x) - x
H(x) = F(x) + x
```

The basic residual block can be summarized as:

```text
y = x + F(x)
```

The shortcut path `x` gives the block a direct route for forward signals and backward gradients.

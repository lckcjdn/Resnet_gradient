# Short-Path Ensemble Behavior

Residual networks can be viewed as collections of many computational paths. A simplified residual stack can be written as:

```text
y = (I + F_L)(I + F_{L-1})...(I + F_1)x
```

Expanding this expression creates terms corresponding to paths that use different subsets of residual branches.

## Effective Paths

An effective path is a path that contributes meaningfully to prediction or gradient flow. The nominal depth of a ResNet can be large, but many influential paths may be shorter than the full network depth.

## Lesion Study

A residual branch lesion test disables selected branches at evaluation time:

```text
y = shortcut(x) + mask * F(x)
```

If accuracy declines smoothly as more residual branches are disabled, that behavior is consistent with path redundancy and ensemble-like behavior. It does not prove that all paths are equally important.

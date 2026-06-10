# Experiment Matrix

| Experiment | Model | Depth | Key Variable | Config | Status |
|---|---:|---:|---|---|---|
| Plain baseline | PlainNet | 56 | no shortcut | `configs/plain56.yaml` | planned |
| Standard residual | ResNetV1 | 56 | identity shortcut | `configs/resnet56.yaml` | planned |
| Pre-activation residual | PreActResNet | 56 | pre-activation identity path | `configs/preact_resnet56.yaml` | planned |
| Scaled shortcut | ScaledShortcutResNet | 56 | lambda = 0.5 | `configs/scaled_lambda_05.yaml` | planned |
| Scaled shortcut | ScaledShortcutResNet | 56 | lambda = 0.9 | `configs/scaled_lambda_09.yaml` | planned |
| Scaled shortcut | ScaledShortcutResNet | 56 | lambda = 1.0 | `configs/scaled_lambda_10.yaml` | planned |
| Scaled shortcut | ScaledShortcutResNet | 56 | lambda = 1.1 | `configs/scaled_lambda_11.yaml` | planned |
| Lesion study | PreActResNet | 56 | branch masks | `configs/lesion.yaml` | planned |

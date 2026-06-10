# ResNet Gradient Stability & Short-Path Ensemble Study — Codex Project Initialization Spec

> 用途：本文件用于交给 Codex 初始化一个完整课程项目。项目目标是通过可复现实验验证 ResNet 的两个核心机制：
> 1. **Identity Mapping / Identity Shortcut** 让前向信号和反向梯度具有更直接的传播路径；
> 2. **Short-Path Ensemble Behavior** 使 ResNet 不像一条单一超深网络，而更像许多不同长度路径的组合。
>
> 最终产物包括：代码、实验记录、过程性结论、图表、CSV 表格、PPT 可用素材、Git 仓库结构与文档。

---

## 0. Codex 总任务说明

请 Codex 按照本文件完成项目初始化、代码实现、实验运行、图表生成、文档记录和 Git 管理。

### 0.1 项目总目标

本项目不是追求最高分类精度，而是通过 controlled experiments 证明以下观点：

1. **Plain CNN 随深度增加更容易出现优化困难和梯度传播不稳定。**
2. **ResNet 的 identity shortcut 提供直接梯度传播通路，使 layer-wise gradient norm 更平滑、更稳定。**
3. **Pre-activation ResNet 更接近 Identity Mappings 论文提出的理想结构，因此应表现出更稳定的训练和梯度传播。**
4. **Scaled shortcut 实验可以验证 shortcut 偏离 identity 后，训练稳定性和梯度传播可能变差。**
5. **Residual branch lesion 实验可以验证 ResNet 具有类似多个浅层路径集成的行为，而不是完全依赖一条完整深路径。**

### 0.2 Codex 初始化要求

Codex 需要先完成以下框架搭建：

- 初始化标准 Python 项目结构；
- 初始化自身项目级 harness 框架；
- 初始化 `.codex/` 目录；
- 创建 project-level skill，用于记录实验、保存结果、更新文档；
- 创建 `docs/` 文件夹，保存：
  - 理论说明；
  - 实验计划；
  - 实验过程日志；
  - 过程性结论；
  - 最终报告素材；
- 创建 `results/` 文件夹，保存：
  - 日志；
  - checkpoint；
  - 梯度统计；
  - CSV 表格；
  - 图像；
- 初始化 Git，到远程仓库https://github.com/lckcjdn/Resnet_gradient；
- 创建 `.gitignore`；
- 完成第一次 commit；
- 如提供远程仓库地址，则自动 push；如未提供，则在文档中记录需要用户填入远程地址。

---

# Part I. 原理部分：汇报前需要先讲清楚的概念

本部分应被 Codex 拆分并保存到：

```text
docs/theory/01_resnet_background.md
docs/theory/02_identity_mapping.md
docs/theory/03_short_path_ensemble.md
docs/theory/04_terms_for_presentation.md
```

---

## 1. ResNet 背景：深层网络为什么难训练？

### 1.1 深层网络的直觉优势

理论上，更深的神经网络具有更强的表达能力。
如果一个浅层网络可以拟合某个函数，那么在更深的网络中，多出来的层至少可以学习 identity mapping，从而不应比浅层网络更差。

但是实际训练中，简单堆叠更多层的 Plain CNN 往往出现：

- 训练 loss 降不下去；
- 训练准确率反而下降；
- 浅层梯度变小或不稳定；
- 深层网络比浅层网络更难优化。

这类问题通常不完全是过拟合，因为它甚至可能发生在训练集上。

### 1.2 Degradation Problem

**Degradation problem** 指的是：
当网络深度增加后，模型的训练误差和测试误差都可能变差。

它不同于过拟合：

| 问题 | 典型现象 | 主要原因 |
|---|---|---|
| Overfitting | 训练误差低，测试误差高 | 泛化能力不足 |
| Degradation | 训练误差也变高 | 深层网络优化困难 |

原始 ResNet 论文提出 residual learning framework，就是为了缓解这种深层网络优化退化问题。

### 1.3 Residual Learning

假设理想映射为：

```text
H(x)
```

普通网络直接学习：

```text
H(x)
```

ResNet 改为学习残差：

```text
F(x) = H(x) - x
```

于是：

```text
H(x) = F(x) + x
```

残差块可以写为：

```text
y = x + F(x)
```

其中：

- `x` 是 shortcut identity path；
- `F(x)` 是 residual branch；
- `+` 是 element-wise addition。

---

## 2. Identity Mappings in Deep Residual Networks

对应论文：

> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
> **Identity Mappings in Deep Residual Networks**, 2016.
> arXiv: https://arxiv.org/abs/1603.05027

### 2.1 需要讲清楚的专业名词

#### 2.1.1 Shortcut Connection

Shortcut connection 是跳过若干层的连接。

在 ResNet 中最常见的是：

```text
y = F(x) + x
```

这里 `x` 不经过卷积变换，直接加到 residual branch 的输出上。

#### 2.1.2 Identity Mapping

Identity mapping 指的是：

```text
h(x) = x
```

输入是什么，输出就是什么。

在 ResNet 中，如果 shortcut 是 identity mapping，则信息可以不经过 residual branch，直接从一个 block 传播到后续 block。

#### 2.1.3 Residual Branch

Residual branch 是残差块中的非线性变换部分：

```text
F(x) = Conv-BN-ReLU-Conv-BN(...)
```

它学习的是对输入 `x` 的修正，而不是完整映射。

#### 2.1.4 Pre-activation

Pre-activation ResNet 将 BN 和 ReLU 放在卷积之前：

```text
x → BN → ReLU → Conv → BN → ReLU → Conv → + x
```

而不是 Standard ResNet v1 的：

```text
x → Conv → BN → ReLU → Conv → BN → + x → ReLU
```

Pre-activation 的目的之一是让 addition 之后的路径更接近 identity，使信号和梯度传播更直接。

#### 2.1.5 After-addition Activation

Standard ResNet v1 在 addition 后通常有 ReLU：

```text
y = ReLU(F(x) + x)
```

Identity Mappings 论文中强调：如果 addition 后的激活函数也尽量接近 identity，前向和反向传播会更直接。

---

## 3. Identity Mapping 的梯度传播解释

对于标准残差块：

```text
x_{l+1} = x_l + F_l(x_l)
```

反向传播为：

```text
∂L / ∂x_l = ∂L / ∂x_{l+1} · (I + ∂F_l / ∂x_l)
```

其中 `I` 来自 identity shortcut。

这意味着梯度可以分成两部分：

```text
直接路径梯度：∂L / ∂x_{l+1} · I
残差分支梯度：∂L / ∂x_{l+1} · ∂F_l / ∂x_l
```

如果没有 shortcut，梯度必须连续经过多层非线性变换，容易出现连乘导致的梯度消失或爆炸。

如果有 identity shortcut，梯度至少有一条更直接的路径可以传播。

### 3.1 Scaled Shortcut 的理论动机

为了验证 identity shortcut 是否关键，可以人为改变 shortcut：

```text
x_{l+1} = λx_l + F_l(x_l)
```

此时反向传播为：

```text
∂L / ∂x_l = ∂L / ∂x_{l+1} · (λI + ∂F_l / ∂x_l)
```

当：

```text
λ = 1
```

shortcut 是标准 identity mapping。

当：

```text
λ < 1
```

直接梯度路径被削弱。

当：

```text
λ > 1
```

直接梯度路径被放大。

因此，Scaled Shortcut Ablation 可以用来验证：

> ResNet 的收益不是简单来自“加法连接”，而是来自接近 identity mapping 的直接路径。

---

## 4. Residual Networks Behave Like Ensembles of Relatively Shallow Networks

对应论文：

> Andreas Veit, Michael Wilber, Serge Belongie.
> **Residual Networks Behave Like Ensembles of Relatively Shallow Networks**, 2016.
> arXiv: https://arxiv.org/abs/1605.06431

### 4.1 需要讲清楚的专业名词

#### 4.1.1 Path

在 ResNet 中，每个 residual block 都有两条可能路径：

```text
1. 经过 residual branch: F(x)
2. 经过 shortcut: x
```

多个 residual blocks 叠加后，网络中会形成很多不同长度的路径。

#### 4.1.2 Effective Path

Effective path 指对最终预测或梯度贡献较大的路径。

ResNet 的名义深度可能很深，但实际训练中起主要作用的路径可能远短于完整深度。

#### 4.1.3 Ensemble Behavior

Ensemble behavior 指模型的表现类似多个子模型的组合。

Veit 等人的观点是：

> ResNet 可以被看成许多不同长度路径的集合，而不是一条单一的超深网络。

#### 4.1.4 Lesion Study

Lesion study 指人为移除或破坏模型中的某些模块，观察性能如何变化。

在 ResNet 中可以测试：

```text
y = x + F(x)
```

将 residual branch 关闭后：

```text
y = x
```

如果删除部分 residual branches 后性能是平滑下降，而不是立即崩溃，说明网络可能具有路径冗余和 ensemble-like behavior。

---

## 5. Short-Path Ensemble 的路径展开解释

若忽略非线性细节，一个 L-block ResNet 可以写为：

```text
y = (I + F_L)(I + F_{L-1})...(I + F_1)x
```

展开后会出现许多项：

```text
x
F_1(x)
F_2(x)
F_2(F_1(x))
...
```

每一项都可以理解为一条路径。

由于每个 block 都可以选择：

```text
经过 F_l
或跳过 F_l
```

所以理论上路径数量会随 block 数量快速增长。

这并不意味着所有路径都同等重要。
实验上需要验证的是：

- ResNet 对部分 residual branch 删除是否鲁棒；
- 只保留一部分 active blocks 时性能是否逐渐下降；
- 说明模型不是完全依赖完整深层路径。

---

# Part II. 实验部分：用图和表证明原理

本部分应被 Codex 保存到：

```text
docs/experiment_plan.md
docs/experiment_matrix.md
docs/expected_figures_and_tables.md
```

---

## 6. 总体实验设置

### 6.1 数据集

优先使用：

```text
CIFAR-10
```

原因：

- 数据小，容易下载；
- 训练成本低；
- ResNet 在 CIFAR 上有经典设置；
- 适合课程项目快速复现。

可选：

```text
CIFAR-100
Fashion-MNIST
```

但主实验建议只使用 CIFAR-10。

### 6.2 公平比较原则

所有模型尽量保持：

```text
same dataset
same data augmentation
same optimizer
same learning rate
same batch size
same epochs
same random seed
same logging method
same evaluation metric
```

建议使用：

```text
Optimizer: SGD
Momentum: 0.9
Weight decay: 5e-4
Batch size: 128
Epochs: 30 / 50 / 100
Learning rate: 0.1
Scheduler: MultiStepLR or CosineAnnealingLR
Seeds: 0, 1, 2
```

如果算力有限：

```text
Epochs: 10 / 20
Depth: 20 / 32
```

如果算力足够：

```text
Epochs: 100
Depth: 20 / 56
```

### 6.3 模型列表

必做模型：

```text
PlainNet-56
Standard ResNet-56
PreAct ResNet-56
ScaledShortcut ResNet-56 λ=0.5
ScaledShortcut ResNet-56 λ=0.9
ScaledShortcut ResNet-56 λ=1.0
ScaledShortcut ResNet-56 λ=1.1
```

可选扩展：

```text
PlainNet-20
Standard ResNet-20
PreAct ResNet-20
ScaledShortcut ResNet-56 λ=1.5
```

---

## 7. 实验 1：PlainNet vs Standard ResNet vs PreAct ResNet

### 7.1 实验目的

证明：

```text
ResNet 相比 PlainNet 更容易优化；
PreAct ResNet 由于更接近 identity mapping，梯度传播更稳定。
```

### 7.2 对应原理

对应：

```text
Identity Mapping
Direct Gradient Propagation
Degradation Problem
```

### 7.3 实验设置

模型：

```text
PlainNet-56
Standard ResNet-56
PreAct ResNet-56
```

指标：

```text
train loss
train accuracy
test accuracy
layer-wise gradient norm
gradient norm heatmap
gradient stability ratio
```

### 7.4 必须产出的图

#### Figure 1: Training Loss Curves

路径：

```text
results/figures/fig01_training_loss_plain_resnet_preact.png
```

内容：

```text
横轴：epoch
纵轴：training loss
曲线：PlainNet-56 / Standard ResNet-56 / PreAct ResNet-56
```

证明目标：

```text
ResNet / PreAct ResNet 收敛更稳定；
PlainNet 深层训练更困难。
```

#### Figure 2: Test Accuracy Curves

路径：

```text
results/figures/fig02_test_accuracy_plain_resnet_preact.png
```

内容：

```text
横轴：epoch
纵轴：test accuracy
曲线：PlainNet-56 / Standard ResNet-56 / PreAct ResNet-56
```

证明目标：

```text
ResNet 系列更容易从深度中获益。
```

#### Figure 3: Layer-wise Gradient Norm

路径：

```text
results/figures/fig03_layerwise_gradient_norm_epoch_last.png
```

内容：

```text
横轴：layer index
纵轴：log10(gradient norm + 1e-12)
曲线：PlainNet-56 / Standard ResNet-56 / PreAct ResNet-56
```

证明目标：

```text
PlainNet 浅层梯度更容易衰减；
ResNet 梯度在层间更平滑；
PreAct ResNet 最接近稳定直接传播。
```

#### Figure 4: Gradient Heatmap

路径：

```text
results/figures/fig04_gradient_heatmap_plain_resnet_preact.png
```

内容：

```text
横轴：layer index
纵轴：epoch
颜色：log10(gradient norm + 1e-12)
```

证明目标：

```text
展示梯度稳定性随训练过程的变化。
```

### 7.5 必须产出的表

#### Table 1: Final Performance

路径：

```text
results/tables/table01_final_performance.csv
docs/tables/table01_final_performance.md
```

字段：

```text
model
depth
params
epochs
final_train_loss
final_train_acc
final_test_acc
best_test_acc
final_grad_stability_ratio
seed
```

#### Table 2: Gradient Stability Summary

路径：

```text
results/tables/table02_gradient_stability.csv
docs/tables/table02_gradient_stability.md
```

字段：

```text
model
mean_grad_norm
std_grad_norm
min_grad_norm
max_grad_norm
shallow_grad_norm
deep_grad_norm
shallow_to_deep_ratio
log_grad_norm_variance
seed
```

### 7.6 预期过程性结论

Codex 应将实验后结论写入：

```text
docs/process_conclusions.md
```

模板：

```markdown
## Experiment 1 Process Conclusion

- PlainNet-56 的训练 loss 是否明显高于 ResNet？
- PlainNet-56 是否出现浅层梯度明显衰减？
- Standard ResNet 与 PreAct ResNet 谁的梯度分布更平滑？
- 当前结果是否支持 identity shortcut 改善梯度传播？
- 是否存在反例或不稳定现象？
```

---

## 8. 实验 2：Scaled Shortcut Ablation

### 8.1 实验目的

证明：

```text
Shortcut 是否接近 identity mapping 会影响梯度稳定性。
```

### 8.2 对应原理

对应：

```text
x_{l+1} = λx_l + F_l(x_l)
```

当 `λ=1` 时是 identity shortcut。
当 `λ` 偏离 1 时，直接梯度路径被削弱或放大。

### 8.3 实验设置

模型：

```text
ScaledShortcut ResNet-56 λ=0.5
ScaledShortcut ResNet-56 λ=0.9
ScaledShortcut ResNet-56 λ=1.0
ScaledShortcut ResNet-56 λ=1.1
```

可选：

```text
λ=1.5
```

### 8.4 必须产出的图

#### Figure 5: Lambda Ablation Loss Curve

路径：

```text
results/figures/fig05_lambda_ablation_loss.png
```

内容：

```text
横轴：epoch
纵轴：training loss
曲线：λ=0.5 / 0.9 / 1.0 / 1.1
```

证明目标：

```text
λ=1.0 附近训练更稳定；
λ 偏离 1 后优化可能变差。
```

#### Figure 6: Lambda Ablation Gradient Stability

路径：

```text
results/figures/fig06_lambda_ablation_gradient_stability.png
```

内容：

```text
横轴：λ
纵轴：gradient stability ratio 或 log_grad_norm_variance
```

证明目标：

```text
identity shortcut 对梯度传播稳定性更有利。
```

#### Figure 7: Lambda Ablation Accuracy

路径：

```text
results/figures/fig07_lambda_ablation_accuracy.png
```

内容：

```text
横轴：λ
纵轴：best test accuracy / final test accuracy
```

证明目标：

```text
shortcut 偏离 identity 可能影响最终性能。
```

### 8.5 必须产出的表

#### Table 3: Shortcut Lambda Ablation

路径：

```text
results/tables/table03_lambda_ablation.csv
docs/tables/table03_lambda_ablation.md
```

字段：

```text
lambda
model
final_train_loss
best_test_acc
final_test_acc
mean_grad_norm
log_grad_norm_variance
shallow_to_deep_ratio
training_stability_note
seed
```

### 8.6 预期过程性结论

写入：

```text
docs/process_conclusions.md
```

模板：

```markdown
## Experiment 2 Process Conclusion

- λ=1.0 是否取得最稳定梯度？
- λ<1 是否出现浅层梯度衰减更明显？
- λ>1 是否出现梯度放大或训练不稳定？
- 结果是否支持 identity shortcut 的关键性？
```

---

## 9. 实验 3：Residual Branch Lesion Study

### 9.1 实验目的

证明：

```text
ResNet 不完全依赖一条完整深层路径，而具有类似浅层路径集成的行为。
```

### 9.2 对应原理

对应：

```text
Residual Networks Behave Like Ensembles of Relatively Shallow Networks
```

残差块：

```text
y = x + F(x)
```

关闭 residual branch：

```text
y = x
```

如果随机关闭部分 residual branch 后准确率不是立即崩溃，而是平滑下降，则说明模型具有路径冗余和 ensemble-like behavior。

### 9.3 实验设置

使用训练好的：

```text
PreAct ResNet-56
Standard ResNet-56
```

在测试阶段进行 lesion，不重新训练。

Drop ratio：

```text
0%
10%
30%
50%
70%
90%
```

Drop strategy：

```text
random_drop
early_drop
late_drop
uniform_interval_drop
```

### 9.4 必须产出的图

#### Figure 8: Lesion Accuracy Curve

路径：

```text
results/figures/fig08_lesion_accuracy_curve.png
```

内容：

```text
横轴：drop ratio
纵轴：test accuracy
曲线：random_drop / early_drop / late_drop / uniform_interval_drop
```

证明目标：

```text
准确率随 residual branch 删除比例平滑下降；
支持 ensemble-like behavior。
```

#### Figure 9: Lesion Loss Curve

路径：

```text
results/figures/fig09_lesion_loss_curve.png
```

内容：

```text
横轴：drop ratio
纵轴：test loss
曲线：不同 drop strategy
```

#### Figure 10: Lesion Sensitivity by Block Index

路径：

```text
results/figures/fig10_lesion_sensitivity_by_block.png
```

内容：

```text
横轴：被删除 block index
纵轴：accuracy drop
```

证明目标：

```text
分析不同 block 的重要性是否均匀；
观察 early / late blocks 是否更敏感。
```

### 9.5 必须产出的表

#### Table 4: Lesion Study Summary

路径：

```text
results/tables/table04_lesion_summary.csv
docs/tables/table04_lesion_summary.md
```

字段：

```text
model
drop_strategy
drop_ratio
test_loss
test_accuracy
accuracy_drop
num_total_blocks
num_dropped_blocks
seed
```

### 9.6 预期过程性结论

写入：

```text
docs/process_conclusions.md
```

模板：

```markdown
## Experiment 3 Process Conclusion

- 随机删除 residual branches 后准确率是否平滑下降？
- 删除少量 branch 是否没有导致模型崩溃？
- early_drop 与 late_drop 哪个更敏感？
- 结果是否支持 short-path ensemble behavior？
```

---

## 10. 实验 4：Active Path Number Analysis 可选

### 10.1 实验目的

进一步验证：

```text
ResNet 中存在较短有效路径；
只保留一部分 active residual blocks 时，模型仍可能保留部分性能。
```

### 10.2 实验设置

在测试阶段给每个 residual branch 加 mask：

```text
y = x + m_l F(x)
```

其中：

```text
m_l = 1 表示保留
m_l = 0 表示关闭
```

控制 active blocks 数量：

```text
all
40
30
20
10
5
0
```

### 10.3 必须产出的图

#### Figure 11: Active Blocks vs Accuracy

路径：

```text
results/figures/fig11_active_blocks_accuracy.png
```

内容：

```text
横轴：active block number
纵轴：test accuracy
```

### 10.4 必须产出的表

#### Table 5: Active Blocks Summary

路径：

```text
results/tables/table05_active_blocks_summary.csv
docs/tables/table05_active_blocks_summary.md
```

字段：

```text
model
active_blocks
selection_strategy
test_accuracy
accuracy_drop
seed
```

---

# Part III. 项目框架要求

Codex 必须初始化如下项目结构。

```text
resnet-gradient-path-study/
│
├── README.md
├── PROJECT_INIT.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── Makefile
│
├── .codex/
│   ├── README.md
│   ├── project_instructions.md
│   ├── skills/
│   │   ├── experiment_recorder/
│   │   │   └── SKILL.md
│   │   ├── result_auditor/
│   │   │   └── SKILL.md
│   │   └── git_reporter/
│   │       └── SKILL.md
│   └── harness/
│       ├── README.md
│       ├── task_plan.md
│       ├── run_policy.md
│       └── checklist.md
│
├── configs/
│   ├── default.yaml
│   ├── plain56.yaml
│   ├── resnet56.yaml
│   ├── preact_resnet56.yaml
│   ├── scaled_lambda_05.yaml
│   ├── scaled_lambda_09.yaml
│   ├── scaled_lambda_10.yaml
│   ├── scaled_lambda_11.yaml
│   └── lesion.yaml
│
├── docs/
│   ├── README.md
│   ├── experiment_plan.md
│   ├── experiment_matrix.md
│   ├── expected_figures_and_tables.md
│   ├── experiment_log.md
│   ├── process_conclusions.md
│   ├── final_report_outline.md
│   ├── theory/
│   │   ├── 01_resnet_background.md
│   │   ├── 02_identity_mapping.md
│   │   ├── 03_short_path_ensemble.md
│   │   └── 04_terms_for_presentation.md
│   └── tables/
│       └── README.md
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── cifar.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── plain_cnn.py
│   │   ├── resnet_v1.py
│   │   ├── preact_resnet.py
│   │   └── scaled_shortcut_resnet.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── checkpoint.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── gradient_monitor.py
│   │   ├── lesion.py
│   │   ├── metrics.py
│   │   └── table_writer.py
│   └── visualization/
│       ├── __init__.py
│       ├── plot_curves.py
│       ├── plot_gradients.py
│       └── plot_lesion.py
│
├── harness/
│   ├── __init__.py
│   ├── run_experiment.py
│   ├── run_suite.py
│   ├── artifact_manager.py
│   ├── experiment_registry.py
│   ├── report_writer.py
│   └── sanity_check.py
│
├── scripts/
│   ├── init_project.py
│   ├── train_model.py
│   ├── run_all_experiments.py
│   ├── run_lesion_study.py
│   ├── collect_gradients.py
│   ├── generate_figures.py
│   ├── generate_tables.py
│   ├── update_docs.py
│   └── git_snapshot.sh
│
├── notebooks/
│   └── quick_visual_check.ipynb
│
├── data/
│   └── README.md
│
└── results/
    ├── README.md
    ├── logs/
    ├── checkpoints/
    ├── gradients/
    ├── figures/
    ├── tables/
    └── runs/
```

---

# Part IV. Codex Harness 设计

项目中的 `harness/` 是代码层面的实验调度框架；`.codex/harness/` 是 Codex 自身执行规范和任务记录框架。

---

## 11. 代码层 harness

### 11.1 `harness/run_experiment.py`

功能：

- 读取单个 YAML 配置；
- 创建 run id；
- 初始化日志目录；
- 调用训练；
- 保存 checkpoint；
- 保存 metrics；
- 保存 gradient statistics；
- 自动更新 `docs/experiment_log.md`。

输入示例：

```bash
python harness/run_experiment.py --config configs/preact_resnet56.yaml
```

### 11.2 `harness/run_suite.py`

功能：

- 一次运行多个实验；
- 支持 quick / full 两种模式；
- quick 用于验证代码；
- full 用于最终汇报结果。

输入示例：

```bash
python harness/run_suite.py --suite quick
python harness/run_suite.py --suite full
```

### 11.3 `harness/artifact_manager.py`

功能：

- 统一管理输出路径；
- 创建如下目录：

```text
results/runs/{run_id}/
results/logs/
results/checkpoints/
results/gradients/
results/figures/
results/tables/
```

run_id 格式：

```text
YYYYMMDD-HHMMSS_model_depth_seed
```

例如：

```text
20260610-153000_preact_resnet56_seed0
```

### 11.4 `harness/experiment_registry.py`

功能：

- 记录实验名称；
- 记录对应 config；
- 记录预计输出图和表；
- 记录是否已完成。

### 11.5 `harness/report_writer.py`

功能：

- 将 CSV 表格转换成 markdown table；
- 自动更新 `docs/tables/`；
- 自动把关键结果摘要写入 `docs/process_conclusions.md`。

### 11.6 `harness/sanity_check.py`

功能：

- 检查环境；
- 检查 CUDA 是否可用；
- 检查 CIFAR-10 是否能加载；
- 检查模型 forward 是否正常；
- 检查梯度统计是否能记录；
- 检查输出目录是否存在。

---

## 12. Codex 自身 harness

Codex 需要创建：

```text
.codex/harness/task_plan.md
.codex/harness/run_policy.md
.codex/harness/checklist.md
```

### 12.1 `.codex/harness/task_plan.md`

内容包括：

```markdown
# Codex Task Plan

## Phase 1: Initialize Project
- Create folder structure.
- Create Python package.
- Create docs.
- Create configs.
- Create skills.
- Initialize git.

## Phase 2: Implement Core Models
- PlainNet
- Standard ResNet
- PreAct ResNet
- ScaledShortcut ResNet

## Phase 3: Implement Training and Logging
- Dataloader
- Trainer
- Evaluator
- Gradient monitor
- Checkpointing

## Phase 4: Implement Experiments
- Plain vs ResNet vs PreAct
- Lambda ablation
- Lesion study
- Active blocks optional

## Phase 5: Generate Figures and Tables
- Curves
- Heatmaps
- CSV tables
- Markdown tables

## Phase 6: Documentation and Git
- Update experiment logs.
- Update process conclusions.
- Create final report outline.
- Commit and push.
```

### 12.2 `.codex/harness/run_policy.md`

内容包括：

```markdown
# Run Policy

1. Never overwrite previous experiment results.
2. Every run must have a unique run_id.
3. Every run must save config, metrics, logs, and gradient statistics.
4. Every completed run must update docs/experiment_log.md.
5. Every generated figure must be listed in docs/expected_figures_and_tables.md.
6. Every table must be saved as CSV and Markdown.
7. Any failed run must be recorded with error message and next action.
8. Before pushing to Git, run sanity check.
```

### 12.3 `.codex/harness/checklist.md`

内容包括：

```markdown
# Codex Execution Checklist

## Before coding
- [ ] Read PROJECT_INIT.md.
- [ ] Create project tree.
- [ ] Create .gitignore.
- [ ] Create docs directory.
- [ ] Create .codex skills.

## Before running experiments
- [ ] Run sanity_check.py.
- [ ] Run one forward pass for each model.
- [ ] Run one mini training epoch.
- [ ] Confirm gradient monitor works.

## After each experiment
- [ ] Save metrics.csv.
- [ ] Save gradient_stats.csv.
- [ ] Save config copy.
- [ ] Save checkpoint.
- [ ] Update experiment_log.md.
- [ ] Update process_conclusions.md.

## Before final commit
- [ ] Generate all required figures.
- [ ] Generate all required tables.
- [ ] Update docs/tables.
- [ ] Update final_report_outline.md.
- [ ] Commit changes.
```

---

# Part V. Codex Skills

Codex 需要创建以下 skills。
这些 skills 是项目级约束文档，不依赖外部系统也能作为执行规范使用。

---

## 13. Skill 1: Experiment Recorder

保存到：

```text
.codex/skills/experiment_recorder/SKILL.md
```

内容：

```markdown
# Experiment Recorder Skill

## Purpose

This skill ensures every experiment is reproducible and documented.

## When to use

Use this skill after:
- creating a new experiment config;
- starting a training run;
- completing a training run;
- generating figures;
- generating tables;
- encountering an error.

## Required records

Every experiment must record:

```text
run_id
date_time
git_commit
config_path
model_name
dataset
optimizer
learning_rate
batch_size
epochs
seed
device
status
main_outputs
key_metrics
notes
```

## Documentation targets

Update:

```text
docs/experiment_log.md
docs/process_conclusions.md
results/runs/{run_id}/run_summary.md
```

## Rules

1. Do not overwrite old logs.
2. Append new records chronologically.
3. If a run fails, record the failure and error message.
4. If a result contradicts the expected hypothesis, record it honestly.
5. Save both raw CSV and human-readable Markdown summary.
```

---

## 14. Skill 2: Result Auditor

保存到：

```text
.codex/skills/result_auditor/SKILL.md
```

内容：

```markdown
# Result Auditor Skill

## Purpose

This skill checks whether figures and tables support the stated experimental hypothesis.

## When to use

Use this skill after generating:
- training curves;
- accuracy curves;
- gradient norm plots;
- gradient heatmaps;
- lambda ablation figures;
- lesion study figures;
- CSV summary tables.

## Audit questions

For each result, answer:

1. What hypothesis does this result test?
2. Which figure/table supports it?
3. Is the result consistent with the hypothesis?
4. Are there possible confounding factors?
5. Does the result need rerun with another seed?
6. Is the conclusion too strong?

## Required output

Append audit notes to:

```text
docs/process_conclusions.md
```

## Warning

Do not claim that ResNet completely eliminates vanishing gradients.
Use careful wording:

```text
supports
suggests
is consistent with
provides empirical evidence
```

Avoid overclaiming:

```text
proves completely
guarantees
always solves
```
```

---

## 15. Skill 3: Git Reporter

保存到：

```text
.codex/skills/git_reporter/SKILL.md
```

内容：

```markdown
# Git Reporter Skill

## Purpose

This skill ensures code, docs, and experiment results are versioned.

## When to use

Use this skill:
- after project initialization;
- after implementing a major module;
- after completing an experiment group;
- before final delivery.

## Required commands

```bash
git status
git add .
git commit -m "<clear message>"
```

If remote exists:

```bash
git push
```

If remote does not exist, record instructions in:

```text
docs/git_upload_instructions.md
```

## Commit message examples

```text
init: create project structure and docs
feat: implement plain and residual models
feat: add gradient monitor and logging harness
exp: add lambda shortcut ablation results
exp: add lesion study figures and tables
docs: update process conclusions
```

## Rules

1. Do not commit large datasets.
2. Do not commit unnecessary checkpoints unless explicitly required.
3. Commit final figures and CSV tables.
4. Use .gitignore to exclude raw data and heavy model files when needed.
```

---

# Part VI. 代码实现要求

---

## 16. 模型实现

### 16.1 PlainNet

文件：

```text
src/models/plain_cnn.py
```

要求：

- 与 ResNet-56 尽量使用相近层数；
- 使用 Conv-BN-ReLU；
- 无 shortcut；
- 输出 CIFAR-10 分类结果。

### 16.2 Standard ResNet

文件：

```text
src/models/resnet_v1.py
```

残差块：

```text
Conv-BN-ReLU-Conv-BN-Add-ReLU
```

要求支持：

```text
ResNet-20
ResNet-32
ResNet-56
```

### 16.3 PreAct ResNet

文件：

```text
src/models/preact_resnet.py
```

残差块：

```text
BN-ReLU-Conv-BN-ReLU-Conv-Add
```

要求：

- addition 后尽量不接额外 ReLU；
- shortcut 在通道和分辨率不变时使用 identity；
- 通道变化时使用 projection shortcut。

### 16.4 ScaledShortcut ResNet

文件：

```text
src/models/scaled_shortcut_resnet.py
```

残差块：

```text
y = F(x) + λ shortcut(x)
```

要求：

- λ 从 config 读取；
- 默认 λ=1.0；
- 支持 λ=0.5, 0.9, 1.0, 1.1；
- 记录 λ 到 metrics 和 run summary。

---

## 17. 梯度统计实现

文件：

```text
src/analysis/gradient_monitor.py
```

### 17.1 需要记录的对象

优先记录每个 block 最后一个 Conv 的权重梯度：

```text
block_i.conv2.weight.grad
```

也可以记录每个 block 的所有参数梯度均值。

### 17.2 每个 epoch 记录一次

建议在每个 epoch 的最后一个 batch 后记录：

```text
layer_index
layer_name
grad_norm_l2
grad_norm_l1
grad_abs_mean
grad_abs_max
log10_grad_norm
epoch
model
seed
run_id
```

保存到：

```text
results/gradients/{run_id}_gradient_stats.csv
```

### 17.3 Gradient Stability Ratio

定义：

```text
shallow_to_deep_ratio = mean_grad_norm(first_k_layers) / mean_grad_norm(last_k_layers)
```

建议：

```text
k = 5
```

同时记录：

```text
log_grad_norm_variance = variance(log10_grad_norm across layers)
```

解释：

- ratio 极小：浅层梯度弱，可能梯度衰减；
- ratio 极大：梯度层间分布不均；
- log variance 越大，说明层间梯度越不稳定。

---

## 18. Lesion Study 实现

文件：

```text
src/analysis/lesion.py
```

### 18.1 Residual Branch Mask

每个 residual block 增加 mask：

```python
y = shortcut(x) + mask * residual_branch(x)
```

mask 取值：

```text
1: 保留 residual branch
0: 删除 residual branch
```

### 18.2 Drop Strategies

支持：

```text
random_drop
early_drop
late_drop
uniform_interval_drop
```

### 18.3 Drop Ratio

支持：

```text
0.0
0.1
0.3
0.5
0.7
0.9
```

### 18.4 输出

保存：

```text
results/tables/table04_lesion_summary.csv
results/figures/fig08_lesion_accuracy_curve.png
results/figures/fig09_lesion_loss_curve.png
results/figures/fig10_lesion_sensitivity_by_block.png
```

---

# Part VII. 图表生成规范

---

## 19. Figure List

Codex 必须生成或预留以下图像文件。

| Figure ID | 文件名 | 证明目标 |
|---|---|---|
| Fig. 1 | `fig01_training_loss_plain_resnet_preact.png` | ResNet 更容易优化 |
| Fig. 2 | `fig02_test_accuracy_plain_resnet_preact.png` | ResNet 精度更稳定 |
| Fig. 3 | `fig03_layerwise_gradient_norm_epoch_last.png` | ResNet 梯度层间分布更稳定 |
| Fig. 4 | `fig04_gradient_heatmap_plain_resnet_preact.png` | 展示梯度随 epoch 和层数变化 |
| Fig. 5 | `fig05_lambda_ablation_loss.png` | λ 偏离 identity 对训练影响 |
| Fig. 6 | `fig06_lambda_ablation_gradient_stability.png` | λ 对梯度稳定性影响 |
| Fig. 7 | `fig07_lambda_ablation_accuracy.png` | λ 对最终性能影响 |
| Fig. 8 | `fig08_lesion_accuracy_curve.png` | lesion 平滑下降支持 ensemble 行为 |
| Fig. 9 | `fig09_lesion_loss_curve.png` | lesion 对 loss 的影响 |
| Fig. 10 | `fig10_lesion_sensitivity_by_block.png` | 分析 block 重要性 |
| Fig. 11 | `fig11_active_blocks_accuracy.png` | 可选，active path 数量分析 |

---

## 20. Table List

| Table ID | 文件名 | 作用 |
|---|---|---|
| Table 1 | `table01_final_performance.csv` | 汇总最终性能 |
| Table 2 | `table02_gradient_stability.csv` | 汇总梯度稳定性指标 |
| Table 3 | `table03_lambda_ablation.csv` | 汇总 shortcut λ 消融 |
| Table 4 | `table04_lesion_summary.csv` | 汇总 lesion 实验 |
| Table 5 | `table05_active_blocks_summary.csv` | 可选，active blocks 分析 |

每个 CSV 都需要转换为 Markdown 表格，保存到：

```text
docs/tables/
```

---

# Part VIII. 文档记录规范

---

## 21. `docs/experiment_log.md` 模板

Codex 需要初始化如下内容：

```markdown
# Experiment Log

## Run Template

### Run ID

### Date

### Git Commit

### Config

### Model

### Dataset

### Training Settings

### Status

### Main Results

### Output Files

### Notes

---
```

每次运行后追加：

```markdown
## Run: 20260610-153000_preact_resnet56_seed0

- Date:
- Git commit:
- Config:
- Model:
- Dataset:
- Device:
- Epochs:
- Status:
- Best test accuracy:
- Final train loss:
- Gradient stability ratio:
- Output figures:
- Output tables:
- Notes:
```

---

## 22. `docs/process_conclusions.md` 模板

Codex 需要初始化如下内容：

```markdown
# Process Conclusions

This document records intermediate conclusions during the project.
All conclusions should be written carefully and grounded in generated figures/tables.

## Conclusion Style Rules

Use:

- "The result suggests..."
- "The result is consistent with..."
- "The figure provides evidence that..."
- "The current run does not fully support..."

Avoid:

- "This completely proves..."
- "ResNet always..."
- "Gradient vanishing is fully solved..."

---

## Experiment 1: PlainNet vs ResNet vs PreAct ResNet

### Current Evidence

### Interpretation

### Limitations

### Next Actions

---

## Experiment 2: Scaled Shortcut Ablation

### Current Evidence

### Interpretation

### Limitations

### Next Actions

---

## Experiment 3: Residual Branch Lesion

### Current Evidence

### Interpretation

### Limitations

### Next Actions

---
```

---

## 23. `docs/final_report_outline.md` 模板

用于后续制作期末 PPT。

```markdown
# Final Report / PPT Outline

## Slide 1: Title
ResNet Gradient Stability and Short-Path Ensemble Behavior

## Slide 2: Motivation
Deep networks are powerful but hard to optimize.

## Slide 3: Degradation Problem
Plain deep networks may have higher training error.

## Slide 4: Residual Learning
H(x) = F(x) + x

## Slide 5: Identity Mapping
Identity shortcut provides direct gradient propagation.

## Slide 6: Short-Path Ensemble
ResNet behaves like many paths of different lengths.

## Slide 7: Experimental Design
PlainNet / ResNet / PreAct ResNet / Scaled Shortcut / Lesion Study

## Slide 8: Results — Loss and Accuracy
Use Fig. 1 and Fig. 2.

## Slide 9: Results — Gradient Stability
Use Fig. 3 and Fig. 4.

## Slide 10: Results — Shortcut Ablation
Use Fig. 5, Fig. 6, Fig. 7 and Table 3.

## Slide 11: Results — Lesion Study
Use Fig. 8, Fig. 9, Fig. 10 and Table 4.

## Slide 12: Conclusion
Identity shortcut improves gradient propagation.
ResNet has short effective paths.
Experiments support both mechanisms.
```

---

# Part IX. Git 初始化与上传

---

## 24. `.gitignore` 要求

必须忽略：

```gitignore
__pycache__/
*.pyc
.env
.venv/
venv/
data/cifar-10-batches-py/
data/raw/
results/checkpoints/*.pt
results/checkpoints/*.pth
results/runs/*/checkpoints/
.DS_Store
.ipynb_checkpoints/
```

保留：

```text
results/figures/
results/tables/
docs/
configs/
src/
harness/
scripts/
.codex/
```

是否保留 checkpoint 由用户决定。默认不上传大型 checkpoint。

---

## 25. Git 命令

初始化：

```bash
git init
git add .
git commit -m "init: create ResNet gradient path study project"
```

如果用户提供 GitHub 远程地址：

```bash
git branch -M main
git remote add origin <REMOTE_URL>
git push -u origin main
```

如果使用 GitHub CLI：

```bash
gh repo create resnet-gradient-path-study --private --source=. --remote=origin --push
```

如果没有远程地址，Codex 需要创建：

```text
docs/git_upload_instructions.md
```

内容包括：

```markdown
# Git Upload Instructions

The local repository has been initialized and committed.

To upload it to GitHub:

```bash
git branch -M main
git remote add origin <YOUR_REMOTE_URL>
git push -u origin main
```
```

---

# Part X. Codex 执行优先级

---

## 26. 最小可行版本 MVP

如果时间有限，Codex 应先完成 MVP：

```text
1. 项目结构
2. docs/theory 文档
3. PlainNet-56
4. PreAct ResNet-56
5. 训练脚本
6. 梯度记录
7. Fig. 1, Fig. 2, Fig. 3, Fig. 4
8. Table 1, Table 2
9. experiment_log.md
10. process_conclusions.md
```

这已经可以支撑期末汇报。

---

## 27. 完整版本 Full Version

完整版本继续完成：

```text
1. Standard ResNet-56
2. Scaled Shortcut Ablation
3. Lesion Study
4. Active Blocks Analysis
5. 所有 Figures
6. 所有 Tables
7. final_report_outline.md
8. Git push
```

---

# Part XI. Codex 第一条执行指令

下面是可以直接交给 Codex 的第一条任务。

```markdown
You are working inside a new project named `resnet-gradient-path-study`.

Read `PROJECT_INIT.md` first and follow it strictly.

Your first task is to initialize the full project framework, including:

1. Python project structure.
2. `.codex/` folder with project instructions, harness documents, and skills.
3. `docs/` folder with theory documents, experiment plan, experiment log, process conclusions, expected figures and tables, and final report outline.
4. `src/`, `harness/`, `scripts/`, `configs/`, `results/`, `data/`, and `notebooks/`.
5. `.gitignore`, `requirements.txt`, `pyproject.toml`, `Makefile`, and `README.md`.
6. Initialize Git and make the first commit.

Do not run full experiments yet.
Only create the framework and minimal sanity-check code.

After initialization, report:
- created files;
- project tree;
- next recommended implementation step;
- git status.
```

---

# Part XII. 参考文献

1. He, K., Zhang, X., Ren, S., Sun, J. **Deep Residual Learning for Image Recognition**. arXiv:1512.03385.
   https://arxiv.org/abs/1512.03385

2. He, K., Zhang, X., Ren, S., Sun, J. **Identity Mappings in Deep Residual Networks**. arXiv:1603.05027.
   https://arxiv.org/abs/1603.05027

3. Veit, A., Wilber, M., Belongie, S. **Residual Networks Behave Like Ensembles of Relatively Shallow Networks**. arXiv:1605.06431.
   https://arxiv.org/abs/1605.06431

---

# Part XIII. 最终项目结论应避免的过强表述

不要写：

```text
ResNet 彻底解决了梯度消失问题。
ResNet 一定不会梯度爆炸。
实验完全证明了理论。
```

推荐写：

```text
实验结果支持 identity shortcut 有助于稳定梯度传播。
实验结果与 ResNet 具有 short-path ensemble behavior 的解释一致。
在本项目设置下，PreAct ResNet 相比 PlainNet 表现出更稳定的 layer-wise gradient norm。
Scaled shortcut ablation 表明，shortcut 偏离 identity mapping 可能削弱训练稳定性。
Lesion study 表明，ResNet 对部分 residual branch 的移除具有一定鲁棒性。
```

---

# Part XIV. 期末汇报建议标题

中文标题：

```text
基于梯度范数与路径消融的 ResNet 残差连接有效性验证
```

英文标题：

```text
An Experimental Study of Gradient Stability and Short-Path Ensemble Behavior in ResNet
```

---

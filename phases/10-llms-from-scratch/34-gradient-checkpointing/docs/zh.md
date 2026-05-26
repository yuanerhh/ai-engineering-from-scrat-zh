# 梯度检查点与激活重计算

> 反向传播需要保存所有中间激活。在 700 亿参数、128K 上下文时，每个 rank 的激活量高达 3 TB。检查点以 FLOPs 换内存：重新计算而非保存。问题在于丢弃哪些片段，而答案不是"全部"。

**类型：** 构建
**语言：** Python（含 numpy，可选 torch）
**前置条件：** Phase 10 第04课（预训练微型 GPT），Phase 10 第05课（扩展与分布式）
**时间：** 约 70 分钟

## 问题所在

训练一个 Transformer 需要为每一层存储所有在反向传播中需要求导的操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出和残差流。对于隐藏大小 `d`、序列长度 `L`、批大小 `B` 的一层，激活量约为 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，每层 BF16 激活约 800 MB。64 层模型就是 51 GB 的激活——这还没有乘以微批大小、没有加上注意力 softmax 中间值（每个头 `L^2`），也没有考虑张量并行的部分副本。

两面账：BF16 权重加优化器状态可能放入 80 GB，但激活内存会超限。梯度检查点（又称激活重计算）是标准解法——丢弃大部分激活，在反向传播时重新运行前向传播来恢复它们。代价：额外的 FLOPs。收益：内存下降的比例等于检查点片段数与总层数之比。

朴素实现下，检查点每步大约多消耗 33% 的前向传播 FLOPs。做得好——按照 Korthikanti 等人的"智能选择"进行选择性检查点——可以将内存节省 5 倍，而 FLOP 开销不超过 5%。在 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE 的情况下这至关重要：你既负担不起额外内存，也负担不起浪费的计算。

## 概念讲解

### 反向传播实际需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`，计算它们需要：

- `input`（计算线性层的 `grad_params = input.T @ grad_output`）
- 某些激活导数中间值（ReLU/GELU/softmax 的导数依赖于激活值）

前向传播自动在自动微分图中存储这些内容，每个 `tensor.retain_grad()` 和每个需要输入的操作都保留了一个引用。

### 朴素的完整检查点

将网络分成 `N` 个片段。前向传播期间，只存储每个片段的*输入*。当反向传播需要中间值时，重新运行该片段的前向传播以实体化它们，然后求导。

示例：32 层 Transformer 分成每个 1 层的 32 个片段。

- 内存：32 个层输入（小）vs 32 × （每层激活量）（大）
- 额外计算：每个片段额外一次前向，即约多 33% 的前向 FLOPs（反向是前向的 2 倍，完整步骤变为 1 + 1 + 2 = 4 单位而非 1 + 2 = 3 单位）

这是 Chen 等人 2016 年的原始方案：每 `sqrt(L)` 层设一个检查点以平衡内存和计算。对于 L=64，就是 8 个检查点。

### 选择性检查点（Korthikanti，2022）

并非所有激活的存储代价相同。注意力 softmax 输出的大小为 `B*L*L*heads`，随序列长度**二次方**增长；FFN 隐藏激活为 `B*L*4d`，线性增长。对于长序列，softmax 占主导。

选择性检查点保留廉价存储的激活（线性投影、残差），只重新计算昂贵的（注意力）。你付出最少的 FLOPs 来重新计算，但节省了 O(L^2) 的内存。

Megatron-Core 将其实现为"选择性"激活重计算，用于 2024 年以后大多数前沿训练运行。

### 卸载（Offload）

重新计算的替代方案：在前向传播和反向传播之间将激活转移到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重新计算代价时有优势。混合策略很常见：对一些层做检查点，对其他层做卸载。

FSDP2 将卸载作为一等选项。当 GPU 受内存瓶颈但 CPU-GPU 传输有余量时，卸载表现出色。

### 重计算代价模型

每 `L` 层中每 `k` 层设一个朴素检查点时每步的 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 每片段一次额外前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
开销 = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点，只重新计算注意力内核而非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层的激活量为 `A`，`L` 层的总激活内存为 `L * A`。

完整检查点（片段大小为 1）：只存储 `L * input_volume`（对于标准 Transformer 约 `L * 1/10 A`），节省约 `9 * L * A * 1/10`。

每 `k` 层设一个检查点：存储 `L/k * A` 加上当前活跃片段内 `k-1` 层的激活。

在 `k = sqrt(L)` 时，内存和重计算代价都随 `sqrt(L)` 扩展——均匀代价层的最优权衡。

### 何时不做检查点

- 流水线阶段中已在进行中的最内层——它们反正必须完成
- 如果第一层和最后一层主导了阶段的计算（在 Transformer 中很少见）
- 已经使用 FlashAttention 的注意力内核——Flash 已经快速重新计算 softmax，额外的层级检查点几乎没有额外收益

### 实现模式

1. **函数包装器：** 将片段包装在 `torch.utils.checkpoint.checkpoint(fn, input)` 中。PyTorch 只存储 `input`，在反向传播时重新计算其他所有内容。

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些片段被包装。

3. **手动显式重计算：** 自己编写反向传播，调用自定义的 `recompute_forward` 以存储的输入重复前向传播。

三种方法功能上等价，包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入在重计算时必须聚集或重新散布；需要处理通信代价
- **流水线并行：** 典型模式是对每个流水线阶段的前向传播做检查点，使反向顺序的微批次可以复用激活内存
- **FP8 重计算：** 重计算期间更新的 amax 历史必须与原始前向传播匹配，否则 FP8 缩放因子会漂移，大多数框架会快照缩放因子

## 动手实践

### 第 1 步：带片段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：需要所有激活的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：代价模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优片段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 实际运用

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` — PyTorch 中的标准包装器，包装一个函数，只存储输入，在反向传播时重新计算
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式，2024 年以后前沿训练中的标准
- **FSDP2 卸载**：FSDP2 中的 `offload_policy` 将激活分片卸载到 CPU，而非重新计算
- **DeepSpeed ZeRO-Offload**：优化器状态和激活的 CPU 卸载，与检查点互补

## 产出物

本课产出 `outputs/prompt-activation-recompute-policy.md`——一个接受模型配置（层数、hidden、seq、batch）和可用 GPU 内存，并输出逐层重计算策略（无/选择性/完整/卸载）的提示词。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（完整激活）vs `model_forward_checkpointed` + `model_backward_checkpointed`（片段）。参数梯度必须在机器精度上完全一致。

2. 将片段大小 `k` 从 1 扫到 `L`，绘制 FLOP 开销和内存，找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块的输入但不存储其中间值，在 seq=8192 的 32 层模型上测量 FLOP 开销相比完整层检查点的差异。

4. 添加卸载：将片段输入保存到模拟的"CPU 缓冲区"（一个单独的列表），以字节/时间测量"PCIe 带宽"，找出卸载和重计算之间的盈亏平衡点。

5. 对真实的 PyTorch Transformer 进行基准测试，有无 `torch.utils.checkpoint`。通过 `torch.cuda.max_memory_allocated` 测量内存和步骤时间。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 梯度检查点 | "通过重做前向节省内存" | 只存储片段输入；在反向传播时重新计算中间值以获取梯度支持张量 |
| 激活重计算 | "与检查点相同" | 同一技术的 HPC 风格名称 |
| 片段大小 (k) | "每个检查点多少层" | 其中间值被丢弃并一起重新实体化的层数 |
| 选择性检查点 | "Korthikanti 的技巧" | 只重新计算昂贵存储的激活（注意力 softmax）；保留廉价的 |
| 完整检查点 | "朴素版本" | 重新计算每个片段中每层的中间值 |
| 块检查点 | "粗粒度" | 对整个 Transformer 块做检查点；最大粒度 |
| FLOP 开销 | "计算税" | 每步额外 FLOPs = （重计算 FLOPs）/（前向 + 反向 FLOPs）；朴素 33%，选择性 5% |
| 激活卸载 | "发送到 CPU" | 在前向和反向之间将激活移到 CPU RAM；重新计算的替代方案 |
| sqrt-L 规则 | "经典最优" | 对于均匀代价层，最优检查点间隔为 sqrt(L) 层 |
| 注意力 softmax 量 | "O(L^2) 问题" | `L^2 * heads * batch` 个浮点数；在长上下文时主导激活内存 |

## 延伸阅读

- [Chen 等人，2016 — 用次线性内存代价训练深度网络](https://arxiv.org/abs/1604.06174) —— 形式化梯度检查点的原始论文
- [Korthikanti 等人，2022 — 减少大型 Transformer 模型中的激活重计算](https://arxiv.org/abs/2205.05198) —— 选择性激活重计算和正式代价分析
- [Pudipeddi 等人，2020 — 用新执行算法以恒定内存训练大型神经网络](https://arxiv.org/abs/2002.05645) —— 通过反向模式重新实体化的替代恒定内存方法
- [Ren 等人，2021 — ZeRO-Offload：使十亿规模模型训练民主化](https://arxiv.org/abs/2101.06840) —— 规模化的激活卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) —— 标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— 选择性、完整和块模式

# 注意力机制——突破性进展

> 解码器不再眯眼看压缩摘要，开始直视整个源序列。此后的一切都是注意力加工程。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 09（序列到序列模型）
**时长：** 约 45 分钟

## 问题背景

第 09 课以一个可测量的失败收尾：在玩具复制任务上训练的 GRU 编码器-解码器，从长度 5 时的 89% 准确率降到长度 80 时接近随机水平。原因是结构性的，而非训练 bug：编码器获取的每一点信息都必须压缩到一个固定大小的隐藏状态中，解码器永远看不到其他任何东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行修复方案：不是只给解码器最终编码器状态，而是保留每个编码器状态。在每个解码器步骤，计算编码器状态的加权平均，权重表示"解码器现在需要查看编码器位置 `i` 多少？"那个加权平均就是上下文，它在每个解码器步骤都会变化。

这就是全部思想。Transformer 扩展了它：自注意力将其应用于单个序列，多头注意力并行运行它。但 2014 年的版本已经打破了瓶颈，一旦你理解了它，到 Transformer 的跨越只是工程，不是概念。

## 核心概念

在每个解码器步骤 `t`：

1. 使用前一个解码器隐藏状态 `s_{t-1}` 作为**查询（query）**。
2. 对每个编码器隐藏状态 `h_1, ..., h_T` 打分。每个编码器位置一个标量。
3. 对得分做 softmax，得到和为 1 的注意力权重 `α_{t,1}, ..., α_{t,T}`。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`，即编码器状态的加权平均。
5. 解码器接受 `c_t` 加上前一个输出 token，产生下一个 token。

加权平均是关键所在。当解码器需要将"Je"翻译为"I"时，它对"Je"上的编码器状态赋予高权重。当需要"not"时，对"pas"赋予高权重。上下文向量每步都在重塑。

## 形状（所有人都会犯错的地方）

这是每次注意力实现第一次都会出错的地方。请仔细阅读。

| 对象 | 形状 | 备注 |
|------|------|------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力得分 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加法）得分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状为 `(d_s,)`，`h_i` 形状为 `(d_h,)`。
- `W_a` 形状为 `(d_attn, d_s)`，`U_a` 形状为 `(d_attn, d_h)`。
- tanh 内部的和形状为 `(d_attn,)`。
- `v_α` 形状为 `(d_attn,)`。与 `v_α` 的内积折叠为标量。**这就是 `v_α` 的作用。** 不是魔法，而是将注意力维向量转为标量得分的投影。

**Luong（乘法）得分。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束，如果编码器是双向的则跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W` 形状为 `(d_s, d_h)`。消除了维度相等的约束。
- `concat`：本质上是 Bahdanau 形式，很少用，因为前两种更便宜。

**一个值得命名的 Bahdanau/Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的解码器状态），Luong 使用 `s_t`（*之后*的状态）。混淆它们会产生极其难以调试的细微错误梯度。选择一篇论文并坚持其约定。

## 动手实现

### 步骤一：加法（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上表检查你的形状。`encoder_states` 形状为 `(T_enc, d_h)`，`projected_enc` 形状为 `(T_enc, d_attn)`，`projected_dec` 形状为 `(d_attn,)` 并广播，`combined` 形状为 `(T_enc, d_attn)`，`scores` 形状为 `(T_enc,)`，`weights` 形状为 `(T_enc,)`，`context` 形状为 `(d_h,)`。搞定。

### 步骤二：Luong 点积和通用注意力

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个三行。这就是 Luong 论文的影响力所在。在大多数任务上准确率相同，代码少得多。

### 步骤三：数值计算示例

给定三个编码器状态（大约代表"cat"、"sat"、"mat"）和一个与第一个最对齐的解码器状态，注意力分布集中在位置 0。如果解码器状态偏移以与最后一个对齐，注意力移到位置 2。上下文向量随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行获胜。然后将解码器状态移向第三个编码器状态，观察权重转移。就是这样。注意力就是显式对齐。

### 步骤四：为什么这是通向 Transformer 的桥梁

将上面的语言翻译为 Q/K/V：

- **查询（Query）** = 解码器状态 `s_{t-1}`
- **键（Key）** = 编码器状态（我们打分的对象）
- **值（Value）** = 编码器状态（我们加权求和的对象）

在经典注意力中，键和值是同一个东西。自注意力将它们分开：你可以用不同的学习投影分别用于 K 和 V，将一个序列对自身进行查询。多头注意力用不同的学习投影并行运行。Transformer 多次叠加整个阶段并放弃了 RNN。

数学是一样的，形状是一样的。从 Bahdanau 注意力到缩放点积注意力的教学跨越主要是符号表示的变化。

## 生产使用

PyTorch 和 TensorFlow 直接提供注意力。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是 Transformer 注意力层。5 个位置的查询批次、10 个位置的键/值批次、各 128 维、8 个头。`output` 是新的上下文增强查询，`weights` 是可以可视化的 5×10 对齐矩阵。

### 经典注意力仍然重要的场景

- 教学。单头、单层、基于 RNN 的版本使每个概念都清晰可见。
- Transformer 无法容纳的设备端序列任务。
- 2014-2017 年的任何论文。不了解 Bahdanau 的约定就会误读它们。
- 机器翻译中的细粒度对齐分析。即使在 Transformer 模型上，原始注意力权重也是可解释性工具，理解它们需要知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来可以解释。它们是在各位置上和为 1 的权重；你可以绘制它们；高意味着"注意了这里"。评审者喜欢它们。

但它们的可解释性不如看起来的那么强。Jain 和 Wallace（2019 年）表明，在某些任务上注意力分布可以被打乱和替换为任意替代方案而不改变模型预测。在没有消融或反事实检验的情况下，永远不要将注意力权重作为推理证据来报告。

## 上手实践

将以下内容保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 `softmax` 掩码，使编码器中的填充 token 获得零注意力权重。在有可变长度序列的批次上测试。
2. **中等。** 向 Luong `general` 形式添加多头注意力。将 `d_h` 分成 `n_heads` 组，每头运行注意力，拼接结果。验证单头情况与早期实现一致。
3. **困难。** 在第 09 课的玩具复制任务上训练带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率与序列长度的关系图。与无注意力基线比较。你应该看到随长度增加差距扩大，确认注意力消除了瓶颈。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 注意力（Attention） | 注视事物 | 值序列的加权平均，权重由查询-键相似度计算。 |
| 查询、键、值（Q、K、V） | QKV | 三种投影：Q 提问，K 是匹配对象，V 是返回的内容。 |
| 加法注意力 | Bahdanau | 前馈得分：`v^T tanh(W q + U k)`。 |
| 乘法注意力 | Luong 点积/通用 | 得分为 `q^T k` 或 `q^T W k`。更便宜，在大多数任务上准确率相同。 |
| 对齐矩阵（Alignment matrix） | 漂亮的图 | 注意力权重作为 `(T_dec, T_enc)` 网格。读它可以看到模型注意了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种得分变体及其比较
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性警示
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可运行的 PyTorch 教程

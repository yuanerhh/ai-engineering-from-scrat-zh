# 用于文本的 CNN 和 RNN

> 卷积学习 n-gram，循环网络记忆上下文。两者都被注意力机制超越，但在受限硬件上仍然重要。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 3 · 11（PyTorch 入门）、Phase 5 · 03（词嵌入）、Phase 4 · 02（从零实现卷积）
**时长：** 约 75 分钟

## 问题背景

TF-IDF 和 Word2Vec 产生了忽略词序的平面向量。建立在它们之上的分类器无法区分"dog bites man"（狗咬人）和"man bites dog"（人咬狗）。词序有时携带关键信号。

在 Transformer 出现之前，两类架构填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用一维卷积。宽度为 3 的过滤器是可学习的三元组检测器：它跨越三个词并输出一个得分。叠加不同宽度（2、3、4、5）的过滤器来检测多尺度模式。最大池化到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 一次处理一个 token，维护一个向前传递信息的隐藏状态。顺序处理、有记忆、输入长度灵活。从 2014 年到 2017 年主导序列建模，然后注意力机制出现了。

本课构建两者，然后指出促使注意力机制诞生的失败模式。

## 核心概念

**TextCNN**（Kim，2014）。Token 被嵌入后，宽度为 `k` 的一维卷积在连续 `k` 个嵌入的 n-gram 上滑动过滤器，产生特征图。对该特征图的全局最大池化选出最强的激活。拼接来自多个过滤器宽度的最大池化输出，送入分类器头。

为什么有效：过滤器是可学习的 n-gram；最大池化与位置无关，所以"not good"无论出现在评论开头还是中间都触发同一特征。三种过滤器宽度各 100 个，给你 300 个可学习的 n-gram 检测器。训练是并行的，没有序列依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 跨时间共享。时间 `T` 的隐藏状态是整个前缀的摘要。对于分类，对 `h_1 ... h_T` 进行池化（最大、均值或最后一个）。

普通 RNN 存在梯度消失问题。**LSTM** 添加了决定遗忘什么、存储什么和输出什么的门，使梯度在长序列中保持稳定。**GRU** 将 LSTM 简化为两个门；用更少的参数实现类似的性能。

**双向 RNN** 运行一个正向 RNN 和一个反向 RNN，并拼接隐藏状态。每个 token 的表示都能看到左右两侧的上下文。对于标注任务至关重要。

## 动手实现

### 步骤一：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。池化后的输出大小固定，与输入长度无关。

### 步骤二：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列做最大池化，而非取最后状态。对于分类，最大池化通常优于取最后一个隐藏状态，因为长序列末尾的信息往往主导最后状态。

### 步骤三：梯度消失演示（直觉）

没有门控的普通 RNN 无法学习长程依赖。考虑一个玩具任务：预测 token `A` 是否出现在序列的任何位置。如果 `A` 在第 1 个位置而序列长 100 个 token，来自损失的梯度必须通过 99 次循环权重的乘法反向传播。如果权重小于 1，梯度消失；如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 权重=0.9，经过 100 步：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第 100 步到第 1 步的梯度实际上为零。
```

LSTM 通过**细胞状态（cell state）**来解决这个问题，细胞状态仅通过加法交互贯穿网络（遗忘门以乘法缩放它，但梯度仍然沿"高速公路"流动）。GRU 用更少的参数实现类似功能。两者都能在 100+ 步序列上稳定训练。

### 步骤四：为什么这仍然不够

即使有 LSTM，三个问题依然存在。

1. **序列瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 个串行的前向/反向步骤，无法在时间维度上并行化。
2. **编码器-解码器设置中的固定大小上下文向量。** 解码器只看到编码器的最终隐藏状态，它压缩了整个输入。长输入丢失细节。第 09 课直接涵盖这一点。
3. **远程依赖准确率上限。** LSTM 优于普通 RNN，但仍难以在 200+ 步中传播特定信息。

注意力机制解决了这三个问题。Transformer 完全放弃了循环。第 10 课是转折点。

## 生产使用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 是生产就绪的。训练代码是标准的。

Hugging Face 提供你可以插入作为输入层的预训练嵌入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

根据约束条件选择的清单：

- **边缘/设备端推理。** 使用 GloVe 嵌入的 TextCNN 比 Transformer 小 10-100 倍。如果你的部署目标是手机，这就是对应的技术栈。
- **流式/在线分类。** RNN 一次处理一个 token；Transformer 需要完整序列。对于实时传入的文本，LSTM 仍然胜出。
- **用于基线的小模型。** 在新任务上快速迭代。在 CPU 上 5 分钟内训练一个 TextCNN。
- **有限数据下的序列标注。** BiLSTM-CRF（第 06 课）对于 1000-10000 个标注句子仍然是生产级 NER 架构。

其他所有情况都去用 Transformer。

## 上手实践

将以下内容保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个三类玩具数据集上训练 TextCNN（你自己构造数据）。验证过滤器宽度 (2, 3, 4) 在平均 F1 上优于单一宽度 (3)。
2. **中等。** 为 LSTM 分类器实现最大池化、均值池化和最后状态池化。在小数据集上比较；记录哪种池化胜出并给出假设。
3. **困难。** 构建 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的仅 CRF 基线和 BERT 微调进行比较。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| TextCNN | 文本 CNN | 在词嵌入上堆叠的一维卷积加全局最大池化。Kim（2014）。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 添加输入/遗忘/输出门和细胞状态。在长序列中稳定训练。 |
| GRU | 简化 LSTM | 两个门代替三个。准确率相近，参数更少。 |
| 双向（Bidirectional） | 双方向 | 正向 + 反向 RNN 拼接。每个 token 都能看到两侧的上下文。 |
| 梯度消失（Vanishing gradient） | 训练信号消亡 | 普通 RNN 中反复乘以小于 1 的权重使早期步骤的梯度实际归零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文，八页，易读
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文，意外地清晰
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 对所有人都易于理解的那些图解

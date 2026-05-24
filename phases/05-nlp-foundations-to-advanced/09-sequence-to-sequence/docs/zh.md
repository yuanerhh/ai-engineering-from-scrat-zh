# 序列到序列模型

> 两个 RNN 假装在翻译。它们遭遇的瓶颈就是注意力机制存在的原因。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 08（用于文本的 CNN + RNN）、Phase 3 · 11（PyTorch 入门）
**时长：** 约 75 分钟

## 问题背景

分类将可变长度序列映射为单个标签。翻译将可变长度序列映射为另一个可变长度序列。输入和输出使用不同的词汇表，可能是不同的语言，且长度不保证相等。

seq2seq 架构（Sutskever、Vinyals、Le，2014）用一个刻意简单的方案破解了这个问题：两个 RNN。一个读取源句子并产生一个固定大小的上下文向量；另一个读取该向量并逐 token 生成目标句子。与第 08 课编写的代码相同，只是以不同方式组合在一起。

这值得研究有两个原因。首先，上下文向量瓶颈是 NLP 中最具教学意义的失败。它为注意力机制和 Transformer 的所有优点提供了动机。其次，训练方法（teacher forcing、scheduled sampling、推理时的 beam search）仍然适用于包括 LLM 在内的每个现代生成系统。

## 核心概念

**编码器（Encoder）。** 读取源句子的 RNN。其最终隐藏状态是**上下文向量（context vector）**——对整个输入的固定大小摘要。理论上不丢失源句子的任何信息。

**解码器（Decoder）。** 从上下文向量初始化的另一个 RNN。每一步接受上一个生成的 token 作为输入，并产生对目标词汇表的分布。采样或取 argmax 以选择下一个 token，然后反馈进去。重复直到产生 `<EOS>` token 或达到最大长度。

**训练：** 每个解码步骤的交叉熵损失，在序列上求和。通过两个网络的标准反向传播。

**Teacher forcing。** 训练期间，解码器在步骤 `t` 的输入是位置 `t-1` 的*真实*（ground-truth）token，而非解码器自己的前一个预测。这稳定了训练；没有它，早期错误会级联，模型永远无法学习。在推理时，必须使用模型自己的预测，所以始终存在训练/推理分布差距，这个差距被称为**曝光偏差（exposure bias）**。

**瓶颈。** 编码器对源句子学到的一切都必须压缩到那一个上下文向量中。长句子丢失细节，罕见词被模糊，词序调整（chat noir vs. black cat）必须被记忆而不是计算。

注意力机制（第 10 课）通过让解码器查看*所有*编码器隐藏状态（而不仅仅是最后一个）来解决这个问题。这就是其全部要义。

## 动手实现

### 步骤一：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden` 的形状为 `[1, batch, hidden_dim]`——最后一步。第 08 课说"对 outputs 进行池化用于分类"。这里我们保留最后一个隐藏状态作为上下文向量，忽略每步的 outputs。

### 步骤二：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次调用一步。输入：一批单个 token 和当前隐藏状态。输出：下一个 token 的词汇表 logits 和更新后的隐藏状态。

### 步骤三：带 teacher forcing 的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得命名的参数：`ignore_index=0` 跳过填充 token 的损失；`teacher_forcing_ratio` 是每步使用真实 token 与模型预测的概率。从 1.0（完全 teacher forcing）开始，在训练过程中逐渐降低到约 0.5，以缩小曝光偏差差距。

### 步骤四：推理循环（贪婪解码）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪婪解码在每步选择概率最高的 token。它可能走偏：一旦你确定了一个 token，就无法撤回。**束搜索（Beam search）**保持前 `k` 个部分序列，最终选择得分最高的完整序列。束宽 3-5 是标准设置。

### 步骤五：瓶颈演示

在一个玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度，观察准确率。

```
seq_len=5   复制准确率：98%
seq_len=10  复制准确率：91%
seq_len=20  复制准确率：62%
seq_len=40  复制准确率：23%
```

单个 GRU 隐藏状态无法无损地记忆 40 个 token 的输入。每个编码器步骤都包含这些信息，但解码器只看到最后一个状态。注意力机制直接解决了这个问题。

## 生产使用

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供在数十亿 token 上训练的完整编码器-解码器模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器放弃了 RNN，转而使用 Transformer。高层形态（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全相同。每个块内部的机制不同。

### 何时仍然选择基于 RNN 的 seq2seq

对于新项目几乎永远不要。具体例外：

- 流式翻译，一次消费一个 token 且内存有界。
- 设备端文本生成，Transformer 的内存成本过高。
- 教学目的。理解编码器-解码器瓶颈是理解 Transformer 为何胜出的最快路径。

### 曝光偏差及其缓解方法

- **Scheduled sampling。** 训练过程中逐渐降低 teacher forcing 比例，使模型学会从自己的错误中恢复。
- **最小风险训练（Minimum risk training）。** 在句子级 BLEU 分数上而非 token 级交叉熵上训练。更接近你真正想要的。
- **强化学习微调（RLHF）。** 用指标奖励序列生成器。用于现代 LLM 的 RLHF。

这三种方法仍然适用于基于 Transformer 的生成。

## 上手实践

将以下内容保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现玩具复制任务。在输入输出对相同的目标序列上训练 GRU seq2seq。测量长度为 5、10、20 时的准确率。复现瓶颈现象。
2. **中等。** 添加束宽为 3 的束搜索解码。在小型平行语料库上测量 BLEU 与贪婪解码相比。记录束搜索在哪里胜出（通常在最后几个 token）以及在哪里没有区别。
3. **困难。** 在 1 万对释义数据集上微调 `facebook/bart-base`。将微调模型的束搜索-4 输出与基础模型在保留输入上的输出进行比较。报告 BLEU 并选取 10 个定性示例。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 编码器（Encoder） | 输入 RNN | 读取源序列，产生每步隐藏状态和最终上下文向量。 |
| 解码器（Decoder） | 输出 RNN | 从上下文向量初始化，逐 token 生成目标序列。 |
| 上下文向量（Context vector） | 摘要 | 编码器的最终隐藏状态。固定大小。注意力机制要解决的瓶颈。 |
| Teacher forcing | 使用真实 token | 训练时输入真实的前一个 token。稳定训练。 |
| 曝光偏差（Exposure bias） | 训练/测试差距 | 模型在真实 token 上训练，从未练习从自己的错误中恢复。 |
| 束搜索（Beam search） | 更好的解码 | 每步保持前 k 个部分序列，而非贪婪地确定一个。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq 论文，四页
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和编码器-解码器框架
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力机制论文，读完本课后立即阅读
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可运行的 seq2seq + 注意力代码

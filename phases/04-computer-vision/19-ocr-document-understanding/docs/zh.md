# OCR 与文档理解

> OCR 是一个三阶段流程——检测文本框、识别字符、然后进行布局排列。每个现代 OCR 系统都在重新排列这些阶段或将它们合并。

**类型：** 学习 + 使用
**语言：** Python
**前置条件：** Phase 4 第 06 课（目标检测），Phase 7 第 02 课（自注意力）
**时长：** 约 45 分钟

## 学习目标

- 梳理经典 OCR 流程（检测→识别→布局）和现代端到端替代方案（Donut、Qwen-VL-OCR）
- 实现 CTC（联接时序分类，Connectionist Temporal Classification）损失用于序列到序列的 OCR 训练
- 使用 PaddleOCR 或 EasyOCR 进行无需训练的生产文档解析
- 区分 OCR、布局解析和文档理解，并为每种任务选择正确的工具

## 问题背景

充满文字的图像无处不在：收据、发票、证件、扫描书籍、表格、白板、路牌、截图。从中提取结构化数据——不只是字符，而是"这是总金额"——是应用视觉中最具价值的问题之一。

该领域分为三个技能层次：

1. **OCR 本身**：将像素转换为文本。
2. **布局解析（Layout parsing）**：将 OCR 输出分组为区域（标题、正文、表格、页眉）。
3. **文档理解（Document understanding）**：从布局中提取结构化字段（"invoice_total = $42.50"）。

每个层次都有经典和现代两种方法，"我想要图像中的文本"和"我需要这张收据上的总金额"之间的差距比大多数团队意识到的要大。

## 核心概念

### 经典流程

```mermaid
flowchart LR
    IMG["图像"] --> DET["文本检测<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["词/行<br/>边界框"]
    BOX --> CROP["裁剪每个区域"]
    CROP --> REC["识别<br/>(CRNN + CTC)"]
    REC --> TXT["文本字符串"]
    TXT --> LAY["布局<br/>排序"]
    LAY --> OUT["阅读顺序文本"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**产生逐行或逐词的四边形。
- **识别**将每个区域裁剪为固定高度，运行 CNN + BiLSTM + CTC 产生字符序列。
- **布局**重建阅读顺序（拉丁文从上到下、从左到右；阿拉伯文、日文等则不同）。

### CTC 一段话

OCR 识别从固定长度的特征图中产生可变长度的序列。CTC（Graves 等，2006）让你无需字符级对齐就能训练。模型在每个时间步输出（词汇 + 空白符）的分布；CTC 损失对所有在合并重复项和删除空白符后归约为目标文本的对齐方式进行边缘化（marginalize）。

```
原始输出：" h h h _ _ e e l l _ l l o _ _"
合并重复项并删除空白符后："hello"
```

CTC 是 CRNN 在 2015 年得以工作的原因，也是 2026 年大多数生产 OCR 模型仍在训练的原因。

### 现代端到端模型

- **Donut**（Kim 等，2022）— ViT 编码器 + 文本解码器；直接从图像输出 JSON。没有文本检测器，没有布局模块。
- **TrOCR** — ViT + Transformer 解码器，用于行级 OCR。
- **Qwen-VL-OCR / InternVL** — 针对 OCR 任务微调的完整视觉语言模型；2026 年复杂文档上的最高精度。
- **PaddleOCR** — 成熟生产包中的经典 DB + CRNN 流程；仍是开源主力。

端到端模型需要更多数据和计算，但跳过了多阶段流程中的错误累积。

### 布局解析

对于结构化文档，运行布局检测器（LayoutLMv3、DocLayNet），为每个区域打标签：标题、段落、图形、表格、脚注。阅读顺序就变成"按布局顺序遍历区域，然后拼接"。

对于表格，使用**键值提取（Key-Value extraction）**模型（富视觉文档用 Donut，纯扫描件用 LayoutLMv3）。它们接受图像 + 检测到的文本 + 位置，并预测结构化键值对。

### 评估指标

- **字符错误率（CER，Character Error Rate）** — Levenshtein 距离/参考文本长度。越低越好。生产目标：干净扫描件低于 2%。
- **词错误率（WER，Word Error Rate）** — 同上，但在词级别。
- **结构化字段的 F1** — 用于键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 编辑距离** — 用于端到端文档解析；Donut 论文引入了归一化树编辑距离。

## 动手实现

### 步骤一：CTC 损失 + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) 对词汇（含索引 0 的空白符）的 log-softmax
    targets:        (N, S) int 目标（无空白符）
    input_lengths:  (N,) 每个样本使用的时间步数
    target_lengths: (N,) 每个样本的目标长度
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: 索引序列列表（已删除空白符，已合并重复项）
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时使用高效的 CuDNN 实现。贪心解码器比束搜索（beam search）更简单，CER 通常在 1% 以内。

### 步骤二：微型 CRNN 识别器

用于行级 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN 最大池化将高度降为 1）。宽度是 CTC 的时间维度。

### 步骤三：合成 OCR 数据

生成黑底白字的数字字符串用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR 数据集添加字体、噪声、旋转、模糊和颜色。上述流程完全相同。

### 步骤四：训练概要

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单的合成数据上，损失应在 200 步内从约 3 下降到约 0.2。

## 生产实践

三条生产路径：

- **PaddleOCR** — 成熟、快速、多语言。一行代码：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** — Python 原生，多语言，PyTorch 骨干。
- **Tesseract** — 经典；在模型难以处理的旧扫描文档上仍然有用。

对于端到端文档解析，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于结构可重复的收据、发票和表格，微调 Donut。对于任意文档或需要推理的 OCR，Qwen-VL-OCR 等 VLM 是当前默认选择。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OCR | "从像素提取文本" | 将图像区域转换为字符序列 |
| CTC | "无对齐损失" | 无需逐时间步标签训练序列模型的损失；对对齐方式进行边缘化 |
| CRNN | "经典 OCR 模型" | 卷积特征提取器 + BiLSTM + CTC；2015 年的基线，仍用于生产 |
| Donut | "端到端 OCR" | ViT 编码器 + 文本解码器；直接从图像输出 JSON |
| 布局解析（Layout parsing） | "找区域" | 在文档中检测并标注标题/表格/图形/段落区域 |
| 阅读顺序（Reading order） | "文本序列" | 将识别区域排序为句子；拉丁文简单，混合布局则不然 |
| CER / WER | "错误率" | 以字符或词为粒度的 Levenshtein 距离/参考文本长度 |
| VLM-OCR | "能阅读的 LLM" | 针对 OCR 任务训练或提示的视觉语言模型；当前复杂文档的 SOTA |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 原始 CNN+RNN+CTC 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 原始 CTC 论文；密集包含算法思想
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) — 无需 OCR 的文档理解 Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源生产 OCR 技术栈

# 视觉 Transformer（ViT）

> 将图像切成 patch，把每个 patch 当成一个词，运行标准 Transformer。不必回头。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 7 第 02 课（自注意力），Phase 4 第 04 课（图像分类）
**时长：** 约 45 分钟

## 学习目标

- 从头实现 patch 嵌入、可学习位置嵌入、类别 token（class token）和 Transformer 编码器块，构建一个最小 ViT
- 解释为何 ViT 曾被认为需要海量预训练数据，直到 DeiT 和 MAE 证明并非如此
- 比较 ViT、Swin 和 ConvNeXt 的架构先验（无先验、局部窗口注意力、卷积骨干）
- 使用 `timm` 和标准线性探测/微调方法，在小型数据集上微调预训练 ViT

## 问题背景

十年来，卷积等同于计算机视觉。CNN 具有强大的归纳偏置——局部性、平移等变性——没有人认为这些能被替代。然后 Dosovitskiy 等（2020）证明，将普通 Transformer 应用于展平的图像 patch（完全没有卷积机制），在规模上可以与最好的 CNN 媲美甚至超越。

问题在于"在规模上"。仅用 ImageNet-1k 训练 ViT 会输给 ResNet。在 ImageNet-21k 或 JFT-300M 上预训练，再在 ImageNet-1k 上微调的 ViT 则会超越它。结论是：Transformer 缺乏有用的归纳偏置，但可以从足够的数据中学习它们。后续工作（DeiT、MAE、DINO）表明，有了正确的训练方案——强数据增强、自监督预训练、蒸馏——ViT 在小数据上也能训练得很好。

到 2026 年，纯 CNN 在边缘设备上仍具竞争力（ConvNeXt 最强），但 Transformer 主导其他一切领域：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 块结构是最值得掌握的。

## 核心概念

### 流程架构

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["Patch 嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>(196, 768) tokens"]
    FLAT --> CAT["添加<br/>[CLS] token"]
    CAT --> POS["加上可学习<br/>位置嵌入"]
    POS --> ENC["N 个 Transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤：Patch → Token → 注意力 → 分类器。每种变体（DeiT、Swin、ConvNeXt、MAE 预训练）改变其中一两个步骤，其余保持不变。

### Patch 嵌入

第一个卷积是关键。卷积核大小 16，步长 16，使 224×224 的图像变成 14×14 的 16×16 patch 网格，每个 patch 投影到 768 维嵌入。这单个卷积同时完成了分块（patchify）和线性投影。

```
输入：(3, 224, 224)
Conv (3 -> 768, k=16, s=16, 无填充)：
输出：(768, 14, 14)
展平空间维度：(196, 768)
```

196 个 patch = 196 个 token。每个 token 的特征维度为 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 类别 Token（Class Token）

一个单独的可学习向量，添加到序列开头：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 Transformer 块后，`[CLS]` 的输出是全局图像表示。分类头只读取这一个向量。

### 位置嵌入

Transformer 没有内置的空间位置概念。在每个 token 上加一个可学习向量：

```
tokens = tokens + learned_pos_embedding   (形状同为 (197, 768))
```

该嵌入是模型的参数，基于梯度的训练使其适应 2D 图像结构。正弦 2D 替代方案也存在，但实践中很少使用。

### Transformer 编码器块

标准结构。多头自注意力、MLP、残差连接、前置 LayerNorm（pre-LN）。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP 为两层，激活函数为 GELU：Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每块 12 个注意力头，共 8600 万参数。

### 为什么使用前置 LN

早期 Transformer 使用后置 LN（`x = LN(x + sublayer(x))`），超过 6-8 层后不使用学习率预热（warmup）就难以训练。前置 LN（`x = x + sublayer(LN(x))`）可以稳定训练更深的网络，无需预热。每个 ViT 和所有现代 LLM 都使用前置 LN。

### Patch 大小的权衡

- 16×16 patch → 196 个 token，标准配置。
- 32×32 patch → 49 个 token，更快但分辨率更低。
- 8×8 patch → 784 个 token，更精细但 O(n²) 的注意力成本急剧增加。

Patch 越大 = token 越少 = 越快但空间细节越少。SwinV2 在层次窗口中使用 4×4 patch。

### DeiT 在 ImageNet-1k 上训练 ViT 的方案

原始 ViT 需要 JFT-300M 才能超越 CNN。DeiT（Touvron 等，2020）仅用 ImageNet-1k，通过四项改变将 ViT-B 训练到 top-1 81.8%：

1. 强数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（Stochastic depth，训练时随机丢弃整个块）。
3. 重复增强（同一张图像每批次采样 3 次）。
4. 从 CNN 教师蒸馏（可选，可进一步提升精度）。

所有现代 ViT 训练方案都源于 DeiT。

### Swin vs ConvNeXt

- **Swin**（Liu 等，2021）— 基于窗口的注意力。每个块在局部窗口内做注意力；交替块移动窗口以混合跨窗口信息。在保留注意力算子的同时引入了类似 CNN 的局部性先验。
- **ConvNeXt**（Liu 等，2022）— 重新设计的 CNN，匹配 Swin 的架构选择（深度卷积、LayerNorm、GELU、倒置瓶颈）。表明差距不在于"注意力 vs 卷积"，而在于"现代训练方案 + 架构"。

2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级的选择；正确选择取决于你的推理栈（ConvNeXt 在边缘设备上编译更好）和预训练数据集。

### MAE 预训练

掩码自编码器（He 等，2022）：随机遮罩 75% 的 patch，训练编码器只处理可见的 25%，训练小型解码器从编码器输出重建被遮罩的 patch。预训练后，丢弃解码器，微调编码器。

MAE 使 ViT 仅在 ImageNet-1k 上即可训练，达到 SOTA，是当前默认的自监督方案。

## 动手实现

### 步骤一：Patch 嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一次卷积，一次展平，一次转置。这就是图像到 token 的全部步骤。

### 步骤二：Transformer 块

前置 LN、多头自注意力、带 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 处理分头、缩放点积和输出投影。`batch_first=True` 使形状为 `(N, seq, dim)`。

### 步骤三：完整 ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 280 万参数——一个可以在 CPU 上运行的微型 ViT。真实的 ViT-B 有 8600 万参数；使用相同的类定义，设置 `dim=768, depth=12, num_heads=12` 即可。

### 步骤四：健全性检查——单张图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应无错误运行。概率之和为 1。

## 生产实践

`timm` 提供了每种 ViT 变体及其 ImageNet 预训练权重，只需一行代码：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` 是 2026 年视觉 Transformer 的生产默认选择。在同一 API 下支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等数十种模型。

对于多模态工作（图像 + 文本），`transformers` 提供 CLIP、SigLIP、BLIP-2、LLaVA。这些模型中的图像编码器都是 ViT 的变体。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Patch 嵌入（Patch embedding） | "第一个卷积" | 卷积核大小 = 步长 = patch 大小的卷积；将图像转换为 token 嵌入网格 |
| 类别 Token（Class token） | "[CLS]" | 添加到 token 序列开头的可学习向量；其最终输出是全局图像表示 |
| 位置嵌入（Positional embedding） | "可学习位置" | 添加到每个 token 的可学习向量，使 Transformer 知道每个 patch 的来源位置 |
| 前置 LN（Pre-LN） | "子层前的 LayerNorm" | 稳定的 Transformer 变体：`x + sublayer(LN(x))`，而非 `LN(x + sublayer(x))` |
| 多头注意力（Multi-head attention） | "并行注意力" | 标准 Transformer 注意力，分成 num_heads 个独立子空间，然后拼接 |
| ViT-B/16 | "基础版，patch 16" | 标准大小：dim=768，depth=12，heads=12，patch_size=16，image=224；约 8600 万参数 |
| DeiT | "数据高效 ViT" | 仅在 ImageNet-1k 上用强增强训练 ViT；证明不严格需要大型预训练数据集 |
| MAE | "掩码自编码器" | 自监督预训练：遮罩 75% 的 patch，重建；主流 ViT 预训练方案 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — 如何仅在 ImageNet-1k 上训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 预训练
- [timm documentation](https://huggingface.co/docs/timm) — 生产中所有视觉 Transformer 的参考文档

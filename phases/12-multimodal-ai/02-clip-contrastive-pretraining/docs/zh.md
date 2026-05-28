# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）用一个足以支撑此后五年发展的核心思想证明了一切：仅用嘈杂的网络图文配对数据和对比损失，将图像编码器与文本编码器对齐到同一向量空间，无需任何监督标签，4 亿对数据。由此产生的嵌入空间可用于零样本分类、图文检索，并作为视觉塔插入 2026 年的每个 VLM。SigLIP 2（2025）用 sigmoid 替换 softmax，以更低成本超越了 CLIP。本课从 InfoNCE 到 sigmoid 配对损失，逐步推导数学原理，并用纯 Python 标准库实现训练步骤。

**类型：** 构建
**编程语言：** Python（标准库，InfoNCE + sigmoid 损失实现）
**前置知识：** Phase 12 · 01（ViT 图像块）、Phase 7（Transformer）
**预计时间：** 约 180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为何 sigmoid 配对损失（SigLIP）可以扩展到超大批量（32,768+），而无需 softmax 所需的 all-gather 通信开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，实现零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练提供的四个调节杠杆：批量大小、温度、提示模板、数据质量。

## 问题背景

CLIP 之前的视觉方法依赖监督学习：收集带标注的数据集（ImageNet：120 万张图像，1000 个类别），训练 CNN，部署上线。标注成本高昂，标注结果偏向于标注者能达成共识的内容，且无法在不微调的情况下迁移到新任务。

而网络上的图文对超过十亿，且免费可得。一张金毛猎犬的图片，配上"我的狗 Max 在公园里"这样的 alt 文字，已经包含了监督信号——文字描述了图片。问题在于：能否将这些信号转化为有效的训练？

CLIP 的答案：将图文配对视为一个匹配任务。给定一批 N 张图像和 N 段描述，学习将每张图像与其对应描述匹配，同时排除 N-1 个干扰项。监督信号是"这两个东西属于一起；这 N-1 个不属于"，无需类别标签，无需人工标注，只需一个对比损失。

由此产生的嵌入空间的能力超出了 CLIP 的训练目标。ImageNet 零样本之所以有效，是因为"a photo of a cat"会嵌入到从未被明确标注为猫的猫图片附近。这个赌注催生了 2026 年的每一个 VLM。

## 核心概念

### 双塔编码器

CLIP 有两个编码器：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出 D 维向量。
- 文本编码器 `g`：小型 Transformer，每段描述输出 D 维向量。

两个编码器都将输出归一化为单位长度。相似度为 `cos(f(x), g(y)) = f(x)ᵀ g(y)`（因为两者都是单位范数）。

对一批 N 个（图像、描述）配对，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习的温度参数（CLIP 初始化为 0.07，在对数空间中学习）。

### InfoNCE 损失

CLIP 对行和列使用对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是其对应描述
loss_t2i = CE(Sᵀ, labels=identity)   # 每段描述的正样本是其对应图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像与其描述的匹配程度超过批内所有其他描述。"负样本"就是批中所有其他元素。批量越大 = 负样本越多 = 信号越强。CLIP 在批量 32k 下训练；规模至关重要。

### 温度参数

`tau` 控制 softmax 的锐化程度。低 tau → 分布尖锐，类似硬负样本挖掘效果；高 tau → 平滑，所有样本都有贡献。CLIP 学习 `log(1/tau)`，并进行截断以防止坍塌。SigLIP 2 固定初始 tau，改用可学习的偏置项。

### 为何 sigmoid 扩展性更好（SigLIP）

Softmax 需要对整个相似度矩阵进行同步计算。在分布式训练中，必须在所有副本之间 all-gather 每个嵌入，再做 softmax，通信量关于副本数是平方增长的。

SigLIP 将 softmax 替换为逐元素 sigmoid：对每对 `(i, j)`，损失是一个二元分类——"这两个是配对的吗？"正类标签是对角线，其余全是负样本。损失为：

```
L = -1/N Σ_{i,j} [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j`，否则为 0。每对损失相互独立，无需 all-gather，每个 GPU 计算其本地块并求和。SigLIP 2 在批量 32k–512k 下通信开销极低，而 CLIP 需要等比例增加通信量。

### 零样本分类

给定 N 个类别名称，为每个类别构建文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入待分类图像。余弦相似度的 argmax 即为预测类别，无需在目标类别上进行任何训练。

提示模板至关重要。CLIP 原始论文对每个类别使用 80 个模板（普通描述、艺术风格、照片、绘画等）并对嵌入取均值，可提升 ImageNet 零样本准确率 3 个百分点。现代使用通常只选一两个模板。

### 线性探针与微调

零样本是基线。线性探针（在冻结的 CLIP 特征上训练一个线性层用于目标类别）在领域内任务上优于零样本。全量微调在领域内任务上优于线性探针，但可能损害零样本迁移能力。三种场景各有不同的权衡。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增：
- NaFlex：单一模型支持可变长宽比和分辨率。
- 更好的密集特征，用于分割和深度估计，目标是作为 VLM 中的冻结骨干。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅限英语。
- 10 亿参数规模，而 CLIP 最多 4 亿。

在 2026 年的开放 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。对于纯图文检索场景（特定 LAION-2B 训练分布与查询模式匹配时），CLIP 仍是默认选择。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 同一思路，18 亿对，90% 噪声，证明了嘈杂数据的规模化有效性。OpenCLIP（LAION）：CLIP 在 LAION-400M / 2B 上的开放复现，多种规模，是首选开放检查点。EVA-CLIP：从掩码图像建模初始化，为 VLM 提供强骨干。BASIC：Google 的 CLIP+ALIGN 混合方案。同一家族，不同数据与调参方式。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本准确率上约达 76%（CLIP-G、OpenCLIP-G）。突破需要更多数据（SigLIP 2 达 80%+）或架构改进（监督分类头、更多参数）。基准测试趋于饱和；真正的价值在于下游 VLM 消费的嵌入空间。

## 动手实践

`code/main.py` 实现了：

1. 玩具双塔编码器（基于哈希的图像特征和文本字符特征），无需 numpy 即可观察 InfoNCE 的形状。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定性）。
3. 用于对比的 sigmoid 配对损失。
4. 零样本分类流程：计算与一组文本提示的余弦相似度，取 argmax 得到预测结果。

运行并观察损失曲线。绝对数值是玩具级别的，但曲线形状与真实 CLIP 训练器的输出一致。

## 产出技能

本课产出 `outputs/skill-clip-zero-shot.md`：给定一组图像（通过路径）和目标类别列表，用 CLIP 模板构建文本提示，用指定检查点（如 `openai/clip-vit-large-patch14`）嵌入两端，返回 top-1 / top-5 预测结果及相似度得分。该技能拒绝对不在提示列表中的类别做出判断。

## 练习

1. 手工计算 4 对样本的 InfoNCE。构造 4×4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用 Python 实现验证手工计算结果。

2. SigLIP 除温度外还有偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批量中正负样本比例严重不均（每行正样本远少于负样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫狗零样本分类器。尝试两个提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率，模板集成是否优于单模板？

4. 计算 512 GPU 在批量 32k 下 softmax InfoNCE vs sigmoid 配对损失的通信开销。哪个是 O(N)，哪个是 O(N²)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放规律论文（arXiv:2212.07143，Cherti 等）。从图表中复现其关于数据规模的结论：在固定模型规模下，ImageNet 零样本准确率与训练数据规模之间是什么对数线性关系？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| InfoNCE | "对比损失" | 批量相似度矩阵上的交叉熵；每个元素的正样本是其配对元素，其余为负样本 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对二元交叉熵；无 softmax，无 all-gather，分布式训练中扩展成本低 |
| 温度（Temperature） | "tau" | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布的锐化程度 |
| 零样本（Zero-shot） | "无需微调的分类" | 用文本提示构建类别嵌入，通过余弦相似度分类；无需在目标类别上训练 |
| 提示模板（Prompt template） | "a photo of a ..." | 类别名称周围的文本脚手架；影响零样本准确率 1–5 个百分点 |
| 双塔编码器（Dual encoder） | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享 D 维空间中 |
| 硬负样本（Hard negative） | "难区分的干扰项" | 与正样本足够相似，使模型必须努力区分的负样本 |
| 线性探针（Linear probe） | "冻结 + 一层" | 仅在冻结特征上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 在任意长宽比和分辨率下无需重缩放即可处理图像的能力 |
| 温度缩放（Temperature scaling） | "对数参数化的 tau" | CLIP 参数化 `log(1/tau)` 使梯度行为更好；截断防止 tau 接近零时的坍塌 |

## 延伸阅读

- [Radford 等 — Learning Transferable Visual Models From Natural Language Supervision（arXiv:2103.00020）](https://arxiv.org/abs/2103.00020) — CLIP 原始论文。
- [Zhai 等 — Sigmoid Loss for Language Image Pre-Training（arXiv:2303.15343）](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等 — ALIGN（arXiv:2102.05918）](https://arxiv.org/abs/2102.05918) — 嘈杂网络数据的规模化。
- [Cherti 等 — Reproducible scaling laws for contrastive language-image learning（arXiv:2212.07143）](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放规律。

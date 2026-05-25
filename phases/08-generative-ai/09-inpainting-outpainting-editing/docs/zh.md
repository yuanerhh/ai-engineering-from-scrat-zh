# 图像修复、扩展与编辑

> 文本到图像创造新事物。图像修复修复旧事物。在生产中，70% 的付费图像工作是编辑——替换背景、移除 logo、扩展画布、重新生成手部。图像修复是扩散模型赚取其存在价值的地方。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 8 · 07（潜在扩散）、Phase 8 · 08（ControlNet 与 LoRA）
**时长：** 约 75 分钟

## 问题背景

客户发来一张完美的产品照片，背景中有一个分散注意力的标志。你想要擦掉这个标志，同时保持其他所有内容像素级不变。你不能从头开始运行文本到图像——结果会有不同的颜色、不同的光照、不同的产品角度。你想要*仅*重新生成被遮罩的区域，并且希望重新生成内容与周围上下文相符。

这就是图像修复（Inpainting）。变体：

- **图像修复（Inpainting）。** 在遮罩内重新生成，保持遮罩外的像素。
- **图像扩展（Outpainting）。** 在遮罩外（或画布之外）重新生成，保持内部。
- **图像编辑（Image editing）。** 重新生成整张图像，但保持对原始图像的语义或结构忠实度（SDEdit、InstructPix2Pix）。

2026 年每个扩散流水线都提供图像修复模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于相同的原理工作。

## 核心概念

![图像修复：遮罩感知去噪与上下文保留重注入](../assets/inpainting.svg)

### 朴素方案（及其为何错误）

使用遮罩运行标准文本到图像。在每个采样步骤，将带噪潜在表示的未遮罩区域替换为前向扩散的干净图像。这能工作……但很糟糕。边界伪影会渗透进来，因为模型没有关于遮罩区域内容的信息。

### 正确的图像修复模型

训练一个修改过的 U-Net，接受 9 个输入通道而非 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外通道是 VAE 编码的源图像副本加上单通道遮罩。训练时，随机遮罩图像区域，并训练模型仅对遮罩区域去噪，同时将未遮罩区域作为干净的条件信号提供。推理时，模型可以"看到"遮罩区域周围的内容，并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或类似）输入。Diffusers 的 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等，2022）——自由编辑

将源图像加噪到某个中间 `t`，然后从 `t` 开始用新提示词运行反向链到 0。无需重新训练。起始 `t` 的选择在忠实度和创意自由之间权衡：

- `t/T = 0.3` → 与源几乎相同，小的风格变化
- `t/T = 0.6` → 适度编辑，保留粗粒度结构
- `t/T = 0.9` → 从接近纯噪声生成，最小源保留

### InstructPix2Pix（Brooks 等，2023）

在 `(输入图像, 指令, 输出图像)` 三元组上微调扩散模型。推理时，同时以输入图像和文本指令为条件（"让它变成日落"、"添加一条龙"）。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint（Lugmayr 等，2022）

保持标准无条件扩散模型。在每个反向步骤，重新采样——偶尔跳回更嘈杂的状态并重新生成。避免边界伪影。在没有训练好的图像修复模型时使用。

## 动手实现

`code/main.py` 在 5 维数据上实现了一个玩具一维图像修复方案。我们在 5 维混合数据上训练 DDPM，其中每个样本是来自两个聚类之一的 5 个浮点数。推理时，我们"遮罩" 5 个维度中的 2 个，在每步注入未遮罩三个维度的前向带噪版本，仅重新生成遮罩维度。

### 步骤一：5 维 DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤二：对所有 5 维训练去噪器

标准 DDPM。网络对 5 维带噪输入输出 5 维噪声预测。

### 步骤三：推理时遮罩感知反向

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 将未遮罩维度替换为干净源的新鲜带噪版本
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后在 x_t 上运行正常的反向步骤
```

这是朴素方案，在玩具一维数据上有效。真实图像修复使用 9 通道输入，因为纹理连贯性更重要。

### 步骤四：图像扩展

图像扩展是遮罩反转的图像修复：遮罩新（以前不存在）的画布，用原始内容填充其余部分。相同的训练目标。

## 常见陷阱

- **接缝。** 朴素方案会留下可见边界，因为梯度信息不会流过遮罩。修复：将遮罩扩大 8-16 像素，或使用正确的图像修复模型。
- **遮罩泄漏。** 如果条件图像的未遮罩区域质量低或有噪声，会污染遮罩内的生成。稍微去噪或模糊。
- **CFG 与遮罩大小交互。** 在小遮罩上使用高 CFG = 饱和的补丁。小编辑时降低 CFG。
- **SDEdit 忠实度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能失去主题的身份。扫描并保存检查点。
- **提示词不匹配。** 提示词应描述*整张*图像，而非只是新内容。"一只猫坐在椅子上"，而非"一只猫"。

## 生产使用

| 任务 | 流水线 |
|------|-------|
| 移除对象，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + "日落时的蓝天" |
| 扩展画布 | SDXL 扩展模式（8px 羽化）或带扩展遮罩的 Flux-Fill |
| 重新生成手部/面部 | SD-Inpaint 加上重新描述主题的提示词 + ControlNet-Openpose |
| 改变一个区域的风格 | 在遮罩区域用 `t/T=0.5` 进行 SDEdit |
| "让它变成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）用于最难情况 |

SAM（Meta 的 Segment Anything，2023）+ 扩散图像修复是 2026 年的背景移除流水线。SAM 2（2024）适用于视频。

## 上手实践

保存 `outputs/skill-editing-pipeline.md`。该技能接受原始图像 + 编辑描述 + 可选遮罩（或 SAM 提示词），输出：遮罩生成方案、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或图像修复模式，以及质量保证清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将遮罩的维度比例从 0.2 变到 0.8。在什么比例时，修复质量（遮罩维度的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每隔 10 个反向步骤，跳回 5 步（添加噪声）并重新去噪。测量它是否减少了遮罩边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill 在 20 个人脸重新生成任务上的表现。分别评分姿态遵循度和身份保留度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 图像修复（Inpainting） | "填充孔洞" | 在遮罩内重新生成；保持遮罩外的像素。 |
| 图像扩展（Outpainting） | "扩展画布" | 在画布之外重新生成；保持内部。 |
| 9 通道 U-Net | "正确的图像修复模型" | 以 `带噪 | 编码源 | 遮罩` 为输入的 U-Net。 |
| SDEdit | "带噪声级别的图生图" | 加噪到时间 `t`，用新提示词去噪。 |
| InstructPix2Pix | "纯文本编辑" | 在（图像, 指令, 输出）三元组上微调的扩散模型。 |
| RePaint | "无需重新训练" | 反向过程中定期重新加噪以减少接缝。 |
| SAM | "分割任何东西" | 通过点击或框选生成遮罩；与修复配合使用。 |
| Flux-Kontext | "带上下文的编辑" | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## 生产注意：编辑流水线对延迟敏感

编辑图像的用户期望 5 秒以内的往返时间。在 L4 上 1024² 的 30 步 SDXL-Inpaint 需要 3-4 秒，加上 SAM 遮罩生成（约 200 毫秒）和 VAE 编码/解码（合计约 500 毫秒）。在生产框架中，这是 TTFT 受限而非吞吐量受限——批大小 1，低并发，最小化每个阶段：

- **SAM-H 是最慢的。** SAM-H 在 1024² 约需 200 毫秒；SAM-ViT-B 约需 40 毫秒，质量损失微小。SAM 2（视频）增加了时间开销；单图像编辑不要使用它。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 编码到潜在表示。如果你有上次生成的潜在表示（在迭代编辑 UI 中很典型），通过 `latents=...` 直接传入以跳过一次 VAE 编码。
- **遮罩扩张对吞吐量也重要。** 小遮罩意味着大部分 U-Net 前向传播被浪费（未遮罩像素无论如何都会被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 始终运行完整 U-Net；只有 9 通道正确修复变体才能利用遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 对 `(源图像, 指令)` 进行单次前向传播——没有单独的遮罩，没有 SDEdit 噪声扫描。在 H100 上约 1.5 秒完成一次编辑。架构启示：折叠各阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的图像修复
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩来源
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力级别编辑
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年工具

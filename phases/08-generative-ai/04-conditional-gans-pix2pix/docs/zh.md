# 条件 GAN 与 Pix2Pix

> 2014-2017 年的第一个重大突破是控制 GAN 的生成内容。附加一个标签、一张图像或一句话。Pix2Pix 实现了图像版本，至今在窄域图像到图像任务上仍胜过所有通用文本到图像模型。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 8 · 03（GAN）、Phase 4 · 06（U-Net）、Phase 3 · 07（CNN）
**时长：** 约 75 分钟

## 问题背景

无条件 GAN 随机采样任意面孔。用于演示有用，用于生产无用。你想要的是：*将草图映射到照片*、*将地图映射到航拍图*、*将白天场景映射到夜晚*、*为灰度图像着色*。在所有这些任务中，你被给定一张输入图像 `x`，必须输出具有某种语义对应关系的 `y`。每个 `x` 对应多个合理的 `y`。均方误差将它们压平成一团。对抗损失不会，因为"看起来真实"是尖锐的。

条件 GAN（Mirza & Osindero，2014）将条件 `c` 作为输入添加到 `G` 和 `D` 两者中。Pix2Pix（Isola 等，2017）对此进行了专门化：条件是完整的输入图像，生成器是 U-Net，判别器是*基于图块*的分类器（PatchGAN），损失是对抗损失 + L1。即使在 2026 年，该方案在窄域图像到图像领域的性能也超过从头训练的文本到图像模型，因为它在*配对数据*上训练——你拥有完全所需的信号。

## 核心概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 Dropout（无输入噪声——Isola 发现显式噪声会被忽略）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 是否看起来真实。

**U-Net 生成器。** 编码器-解码器，在瓶颈处有跳跃连接。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节就会消失。

**PatchGAN 判别器。** D 不是输出单个真/假分数，而是输出一个 `N×N` 网格，其中每个单元格判断约 70×70 像素的感受野。取平均。这是马尔可夫随机场假设：真实感是局部的。训练速度更快，参数更少，输出更锐利。

**损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并将 G 推向已知目标。L1 比 L2 提供更锐利的边缘（中位数，而非均值）。`λ = 100` 是 Pix2Pix 的默认值。

## CycleGAN——当你没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu 等，2017）以额外损失为代价放弃了这一要求：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你可以在没有配对样本的情况下将马转换为斑马、夏天转换为冬天。

2026 年，无配对图像到图像主要通过扩散（ControlNet、IP-Adapter）而非 CycleGAN 实现，但循环一致性的思想几乎在每篇无配对域适应论文中都得以延续。

## 动手实现

`code/main.py` 在一维数据上实现了一个微型条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务：为给定类别从条件分布中生成样本。

### 步骤一：将条件附加到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码是最简单的方式。更大的模型使用学习到的嵌入、FiLM 调制或交叉注意力。

### 步骤二：条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件下*的真实分布，而不是边际分布。

### 步骤三：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** G 学会了边际化，D 从不惩罚，因为条件信号很弱。修复：更激进地在 D 中引入条件（早期层，而非仅在后期层），使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂向任意看起来真实的输出，而非忠实的输出。Pix2Pix 风格任务从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊输出，因为 L1 仍然是 L_p 范数。训练稳定后逐渐降低。
- **D 中的真实标签泄漏。** 将 `(x, y)` 拼接作为 D 的输入，而不仅仅是 `y`。没有这一步，D 无法检查一致性。
- **每类别模式塌缩。** 每个类别可以独立塌缩。运行类条件多样性检查。

## 生产使用

2026 年图像到图像任务的最佳方案：

| 任务 | 最佳方案 |
|------|---------|
| 草图→照片，同域，配对数据 | Pix2Pix / Pix2PixHD（仍然快速、仍然锐利） |
| 草图→照片，无配对 | 带 Scribble 条件模型的 ControlNet |
| 语义分割→照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散；GAN 方法已过时 |
| 深度→照片 | Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 着色 | ColTran、基于扩散的着色器或 Pix2Pix-color |
| 白天→夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方法 |

当 (a) 有数千个配对样本、(b) 任务窄且可重复、(c) 需要快速推理时，Pix2Pix 仍是正确选择。在通用开放域任务上，扩散胜出。

## 上手实践

保存 `outputs/skill-img2img-chooser.md`。该技能接受任务描述、数据可用性（配对与无配对、N 个样本）以及延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本和评估协议（LPIPS、FID、特定任务指标）。

## 练习

1. **简单。** 修改 `code/main.py` 以添加第三个类别。确认 G 仍然将每个类别的噪声映射到正确的模式。
2. **中等。** 在一维设置中用感知风格损失替换 L1（例如用一个小型冻结 D 作为特征提取器）。这改变了条件分布的锐利度吗？
3. **困难。** 在一维设置中勾勒出 CycleGAN：两个分布、两个生成器、循环损失。证明它能在没有配对数据的情况下学会在两者之间进行映射。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 条件 GAN（Conditional GAN） | "带标签的 GAN" | G(z, c)，D(x, c)。两个网络都看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 带 U-Net G 和 PatchGAN D + L1 损失的配对 cGAN。 |
| U-Net | "带跳跃连接的编码器-解码器" | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | "局部真实感分类器" | D 输出每图块分数而非全局分数。 |
| CycleGAN | "无配对图像翻译" | 两个 G + 循环一致性损失；无需配对数据。 |
| SPADE | "GauGAN" | 用语义图归一化中间激活；语义分割到图像。 |
| FiLM | "特征级线性调制" | 来自条件的每特征仿射变换；廉价的条件化。 |

## 生产注意：Pix2Pix 作为延迟受限基线

当你有配对数据和窄域任务（草图→渲染、语义图→照片、白天→夜晚）时，Pix2Pix 的一次性推理在延迟上比扩散快一个数量级。生产比较通常是：

| 方案 | 步数 | 在单块 L4 上 512² 的典型延迟 |
|------|------|---------------------------|
| Pix2Pix（U-Net 前向传播） | 1 | ~30 毫秒 |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 秒 |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 秒 |
| ControlNet + SDXL base | 20-30 | ~3-5 秒 |

Pix2Pix 在静态批处理下的吞吐量上胜出（每个请求的 FLOP 相同）。扩散在质量和泛化上胜出。现代做法通常是为窄域任务部署 Pix2Pix 风格的蒸馏模型，并为长尾输入提供扩散回退方案。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器

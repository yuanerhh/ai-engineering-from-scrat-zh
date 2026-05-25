# GAN——生成器与判别器

> Goodfellow 在 2014 年的技巧是完全跳过密度。两个网络。一个制造假货。一个抓住它们。它们战斗，直到假货与真实无法区分。它不应该奏效。它经常不奏效。当它奏效时，样本在窄域中仍然是文献中最锐利的。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 3 · 02（反向传播）、Phase 3 · 08（优化器）、Phase 8 · 02（VAE）
**时长：** 约 75 分钟

## 问题背景

VAE 产生模糊样本，因为它们的 MSE 解码器损失是*平均*图像的贝叶斯最优——而许多合理数字的平均值是模糊数字。你想要一个奖励*合理性*而非对任何单一目标的逐像素接近度的损失。合理性没有封闭形式。你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和假图像。训练一个生成器 `G(z)` 来愚弄 `D`。`G` 的损失信号是 `D` 当前认为是什么让东西看起来真实。这个信号随 `G` 的改进而更新，追逐一个移动目标。如果两个网络都收敛，`G` 在从未写出 `log p(x)` 的情况下学会了数据分布。

这是对抗训练。数学上是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026 年，GAN 不再是最先进的生成器（扩散和流匹配夺走了那顶王冠）。但 StyleGAN 2/3 仍然是有史以来最锐利的面孔模型，GAN 判别器被用作扩散训练中的*感知损失*，对抗训练驱动着快速的一步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能部署实时扩散。

## 核心概念

![GAN 训练：生成器和判别器在极小极大博弈中](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。解码器形状的网络（密集或转置卷积）。

**判别器 `D(x)`。** 将样本映射到标量概率（或分数）。真实 → 1，假 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。真实=1、假=0 上的二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 置信时饱和并消除梯度）。

**训练循环。** `D` 一步，`G` 一步。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，则 `D` 无法比随机猜测做得更好，到处输出 0.5；`G` 不再得到梯度。平衡。

**为什么失效。** 模式塌缩（`G` 找到一个 `D` 无法分类的模式并永远铸造它），梯度消失（`D` 学得太快，`log D` 饱和），训练不稳定（学习率、批大小，任何东西）。

## 使 GAN 工作的变体

| 年份 | 创新 | 修复 |
|------|------|------|
| 2015 | DCGAN | 卷积/转置卷积，批归一化，LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN、WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | 谱归一化 | Lipschitz 约束判别器。2026 年的判别器中仍在使用。 |
| 2018 | Progressive GAN | 先在低分辨率训练，再添加层。第一个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定域真实感的最先进方案。 |
| 2021 | StyleGAN3 | 无混叠，平移等变——2026 年仍是面孔黄金标准。 |
| 2022 | StyleGAN-XL | 条件、类别感知、更大规模。 |
| 2024 | R3GAN | 加强正则化重新包装；无技巧即可在 1024² 上工作。 |

## 动手实现

`code/main.py` 在一维数据上训练一个微型 GAN：两个高斯的混合。生成器和判别器是单隐藏层 MLP。我们手写前向、反向和极小极大循环。目标是在两种关键失效模式（模式塌缩 + 梯度消失）发生时看到它们。

### 步骤一：非饱和损失

原始 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 高置信地将 G 的假货分类为假货时趋向 0。此时 G 的梯度基本为零——G 无法改进。非饱和形式 `-log D(G(z))` 具有相反的渐近线：当 D 置信时它爆炸，给 G 一个强烈信号。

```python
def g_loss(d_fake):
    # 最大化 log D(G(z))  <=>  最小化 -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 步骤二：每步生成器对应一步判别器

```python
for step in range(steps):
    # 训练 D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训练 G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新的假货
    update_G(fake_batch)
```

G 用新的假货，否则梯度是陈旧的。

### 步骤三：监视模式塌缩

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 模式塌缩：一个模式被饥饿")
```

典型症状：两个真实模式之一停止被生成。判别器停止纠正它，因为它从未被看作假货。

## 常见陷阱

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 达到 >95% 准确率，G 就死了。
- **生成器记住一个模式。** 向 D 的输入添加噪声，使用小批量判别器层，或切换到 WGAN-GP。
- **批归一化泄露统计数据。** 真实批 + 假批流过同一 BN 层会混合它们的统计数据。改用实例归一化或谱归一化。
- **初始分数博弈。** FID 和 IS 在低样本量时噪声大。评估时使用 ≥1 万个样本。
- **条件任务的一次性采样是谎言。** 你仍然需要 CFG 尺度、截断技巧和重采样来获得可用输出。

## 生产使用

2026 年的 GAN 栈：

| 情况 | 选择 |
|------|------|
| 固定姿态的真实感人脸 | StyleGAN3（最锐利，最小） |
| 动漫 / 风格化面孔 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速一步文本到图像 | 扩散的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练器内的感知损失 | 图像裁剪上的小型 GAN 判别器 |
| 任何多模态、开放式 | 不要——使用扩散或流匹配 |

GAN 锐利但狭窄。一旦你的域扩展——照片、任意文本提示、视频——切换到扩散。对抗技巧以组件的形式存在（感知损失、蒸馏），而非独立生成器。

## 上手实践

保存 `outputs/skill-gan-debugger.md`。该技能接受失败的 GAN 运行（损失曲线、样本网格、数据集大小），并输出可能原因的排名列表、单行修复和重新运行协议。

## 练习

1. **简单。** 使用默认设置运行 `code/main.py`。然后将 `D_LR = 5 * G_LR` 并重新运行。G 的损失坍缩为常数有多快？
2. **中等。** 将 Goodfellow BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪至 `[-0.01, 0.01]`。训练更稳定吗？比较实际收敛时间。
3. **困难。** 将一维示例扩展到二维数据（环上 8 个高斯的混合）。追踪生成器在步骤 1k、5k、10k 时捕获了多少个 8 个模式。实现小批量判别并重新测量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 生成器（Generator） | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器（Discriminator） | "D" | 分类器 `D: x → [0, 1]`，真实 vs 假。 |
| 极小极大（Minimax） | "这个博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失（Non-saturating loss） | "修复" | 对 G 使用 `-log D(G(z))` 而非 `log(1 - D(G(z)))`。 |
| 模式塌缩（Mode collapse） | "G 记住了一件事" | 生成器尽管数据多样但产生很少不同的输出。 |
| WGAN | "Wasserstein" | 用地球移动距离 + 梯度惩罚替换 BCE；更平滑的梯度。 |
| 谱归一化（Spectral norm） | "Lipschitz 技巧" | 约束 D 的权重范数以约束其斜率；稳定训练。 |
| StyleGAN | "那个有效的" | 映射网络 + AdaIN；面孔最佳，2026 年仍然如此。 |

## 生产注意：一次性推理是 GAN 持久的优势

GAN 不再在开放域生成的样本质量上获胜，但它们仍然在推理成本上获胜。在生产推理文献词汇中，GAN 具有：

- **没有预填充，没有解码阶段。** 单次 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **没有 KV 缓存压力。** 唯一的状态是权重。批大小受激活内存而非缓存约束。
- **轻松的连续批处理。** 由于每个请求需要相同的固定 FLOP，在服务器目标占用率下的静态批通常是最优的。不需要飞行中的调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文本到图像的主导技术：它将 20-50 步的扩散流水线折叠成 1-4 个 GAN 风格的前向传播，同时保持扩散基础的分布。对抗损失作为将慢速生成器变为快速生成器的训练时旋钮而存续。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — 谱归一化
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo

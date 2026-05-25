# 流匹配与修正流

> 扩散模型需要 20-50 个采样步骤，因为它们从噪声到数据走的是弯曲路径。流匹配（Lipman 等，2023）和修正流（Liu 等，2022）训练了直线路径。更直的路径意味着更少步骤，意味着更快的推理。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了流匹配。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 8 · 06（DDPM）、Phase 1（微积分）
**时长：** 约 45 分钟

## 问题背景

DDPM 的反向过程是从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其压缩到 20-50 个确定性步骤。你想要更少步骤——理想情况下是一步。瓶颈在于求解反向过程的常微分方程是刚性的；路径是弯曲的。

如果你能训练模型使从噪声到数据的路径是*直线*，单步 Euler 从 `t=1` 到 `t=0` 就能工作。流匹配直接构建这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ 数据` 的直线插值，训练向量场 `v_θ(x, t)` 匹配其时间导数，推理时进行积分。

修正流（Liu 2022）更进一步：用 reflow 过程迭代地拉直路径，产生逐渐接近线性的常微分方程。两次 reflow 迭代后，2 步采样器可以匹配 50 步 DDPM 的质量。

## 核心概念

![流匹配：噪声和数据之间的直线插值](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ 数据`，`x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义神经向量场 `v_θ(x_t, t)` 并训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这是**条件流匹配**损失（Lipman 2023）。训练是无模拟的：你永远不需要展开常微分方程。只需采样 `(x_0, x_1, t)` 并进行回归。

### 采样

推理时，*反向*积分学习到的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，Euler 步进到 `t=0`。

### 修正流（Liu 2022）

直线流有效，但学习到的路径*并非真正直线*——它们是弯曲的，因为许多 `x_0` 可以映射到同一个 `x_1`。修正流的 reflow 步骤：

1. 用随机配对训练流模型 v_1。
2. 通过从 `x_1` 积分 v_1 到其终点 `x_0`，采样 N 个配对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。因为配对现在是"常微分方程匹配"的，它们之间的直线插值确实更平坦。
4. 重复。

实践中 2 次 reflow 迭代接近线性，使 2-4 步推理成为可能。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配模型蒸馏而来的。

### 为什么 2024 年图像领域流匹配获胜

三个原因：

1. **无模拟训练**——训练期间不展开常微分方程，实现简单。
2. **更好的损失几何**——直线路径有一致的信噪比，而 DDPM ε 损失在调度边缘的信噪比很差。
3. **更快的推理**——在 SDXL-Turbo 质量下 4-8 步；一致性蒸馏下 1 步。

## 流匹配 vs DDPM——精确联系

使用高斯条件路径的流匹配是具有*特定噪声调度*的扩散。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，流匹配恢复了用 Stratonovich 重新表述的扩散，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，两者在代数上是等价的。

流匹配的贡献：目标的*清晰度*（简单的速度）、更干净的损失，以及实验非高斯插值的许可。

## 动手实现

`code/main.py` 在双模高斯混合上实现了一维流匹配。向量场 `v_θ(x, t)` 是用直线目标训练的小型 MLP。推理时，积分 1、2、4 和 20 个 Euler 步骤并比较样本质量。

### 步骤一：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 反向传播 + 更新
```

### 步骤二：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 步骤三：比较步数

期望 4 步采样器已经匹配 20 步的质量——这对延迟来说意义重大。

## 常见陷阱

- **时间参数化。** 流匹配使用 `t ∈ [0, 1]`，`t=0` 在数据处，`t=1` 在噪声处。DDPM 使用 `t ∈ [0, T]`，`t=0` 在数据处，`t=T` 在噪声处。方向相同，尺度不同。论文经常搞错这一点。
- **调度选择。** 修正流的直线是"那个"流匹配调度，但你可以使用余弦或 logit-normal t 采样（SD3 就是这样做的）以获得更好的尺度覆盖。
- **Reflow 成本。** 为 reflow 生成配对数据集是每个样本的一次完整推理传播。只在真正需要 1-2 步推理时才做 reflow。
- **无分类器引导仍然适用。** 只需在线性组合中用 v 替换 ε：`v_cfg = (1+w) v_cond - w v_uncond`。

## 生产使用

| 使用场景 | 2026 年技术栈 |
|---------|-------------|
| 文本到图像，最佳质量 | 流匹配：SD3、Flux.1-dev |
| 文本到图像，1-4 步 | 蒸馏流匹配：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从流匹配基础进行一致性蒸馏（LCM、PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | 流匹配与扩散混合（Sora、Veo、Stable Video） |
| 科学/物理（粒子轨迹、分子） | 流匹配 + 等变向量场 |

每当 2025-2026 年的论文说"比扩散更快"，几乎总是流匹配 + 蒸馏。

## 上手实践

保存 `outputs/skill-fm-tuner.md`。该技能接受扩散风格的模型规格，将其转换为流匹配训练配置：调度选择、时间采样分布（均匀/logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较 1 步 vs 20 步的 MSE 与真实数据分布。
2. **中等。** 从均匀 `t` 采样切换到 logit-normal（将采样集中在中间 t 值）。模型质量改善了吗？
3. **困难。** 实现一次 reflow 迭代：通过积分第一个模型生成配对的 (x_0, x_1)，在这些配对上训练第二个模型，并比较 1 步样本质量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 流匹配（Flow matching） | "直线扩散" | 训练 `v_θ(x, t)` 沿插值匹配 `x_1 - x_0`。 |
| 修正流（Rectified flow） | "Reflow" | 迭代拉直学习流的过程。 |
| 速度场（Velocity field） | "v_θ" | 模型的输出——移动 `x_t` 的方向。 |
| 直线插值（Straight-line interpolant） | "那条路径" | `x_t = (1-t)·x_0 + t·x_1`；导数是平凡的。 |
| Euler 采样器（Euler sampler） | "一阶常微分方程求解器" | 最简单的积分器；路径直时效果好。 |
| Logit-normal t | "SD3 采样" | 将 `t` 采样集中在梯度最强的中间值。 |
| 一致性蒸馏（Consistency distillation） | "一步采样器" | 训练学生直接将任意 `x_t` 映射到 `x_0`。 |
| 带速度的 CFG（CFG with velocity） | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；相同技巧，新变量。 |

## 生产注意：Flux.1-schnell 是流匹配最快的体现

流匹配的生产胜利是 Flux.1-schnell——一个蒸馏到 1-4 推理步骤同时保持 Flux-dev 级质量的流匹配 DiT。参考部署方案：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 4 步 vs dev 50 步），VAE 解码。成本核算：

| 变体 | 步数 | L4 上 1024² 的延迟 | 总 FLOP（相对） |
|------|------|-------------------|---------------|
| Flux.1-dev（原始） | 50 | ~15 秒 | 1.0× |
| Flux.1-schnell | 4 | ~1.2 秒 | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 秒 | 0.25× |
| SDXL-Lightning 2 步 | 2 | ~0.3 秒 | 0.03× |

生产规则：**流匹配基础 + 蒸馏 = 2026 年快速文本到图像的默认方案。** 每个主要厂商都推出了这种组合：SD3-Turbo（SD3 + 流 + 蒸馏）、Flux-schnell（Flux-dev + 修正流拉直）、CogView-4-Flash。纯扩散基础仅存在于旧版检查点。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 修正流
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模修正流
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 涵盖流匹配 + 扩散的通用框架
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散/流的一步蒸馏
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — Turbo 变体
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的流匹配

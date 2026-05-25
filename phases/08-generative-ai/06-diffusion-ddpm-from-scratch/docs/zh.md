# 扩散模型——从零实现 DDPM

> Ho、Jain、Abbeel（2020）给了这个领域一个无法放弃的方案。用噪声在一千个小步骤中破坏数据。训练一个神经网络来预测噪声。推理时逆转这个过程。今天每个主流图像、视频、3D 和音乐模型都运行在这个循环上，可能还叠加了流匹配或一致性技巧。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 3 · 02（反向传播）、Phase 8 · 02（VAE）
**时长：** 约 75 分钟

## 问题背景

你想要一个 `p_data(x)` 的采样器。GAN 玩极小极大博弈，经常发散。VAE 从高斯解码器产生模糊样本。你真正想要的是一个满足以下条件的训练目标：(a) 单一稳定损失（无鞍点，无极小极大），(b) `log p(x)` 的下界（因此有似然），(c) 匹配 SOTA 质量的样本。

Sohl-Dickstein 等（2015）有一个理论答案：定义一个逐步添加高斯噪声的马尔可夫链 `q(x_t | x_{t-1})`，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020）表明损失可以简化为一行——预测噪声——并清理了数学推导。2020 年这是一个有趣的发现。2021 年它产生了最先进的样本。2022 年它成为 Stable Diffusion。2026 年它是基础架构。

## 核心概念

![DDPM：前向加噪，反向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步骤中添加高斯噪声。封闭形式——使数学可处理的原因——是累积步骤也是高斯的：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 是噪声调度。在 T=1000 步上将 `β_t` 从 1e-4 线性增加到 0.02，`x_T` 近似为 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 是 `sqrt(β_t)` 或学习到的方差。这个表达式看起来复杂，但它只是代数——在给定后验 `q(x_{t-1} | x_t, x_0)` 并用噪声预测估计值替换 `x_0` 后求解 `x_{t-1}`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，选一个随机 `t`，采样 `ε ~ N(0, I)`，通过封闭形式一次性计算带噪声的 `x_t`，然后对噪声进行回归。一个损失，无极小极大，无 KL，无重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 迭代到 `1` 执行反向步骤。完成。

## 为什么有效

三个直觉：

1. **去噪容易；生成难。** 在 `t=T` 时，数据是纯噪声——网络只需解决一个简单问题。在 `t=0` 时，网络只需清理几个像素。在中间 `t` 时，问题很难，但网络从每个噪声级别得到大量梯度流过相同的权重。

2. **变相的分数匹配。** Vincent（2011）证明预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*分数*。反向 SDE 使用这个分数沿密度梯度行走——朝高概率区域的有引导随机游走。

3. **ELBO 化简为简单 MSE。** 完整的变分下界每个时间步有一个 KL 项。用 DDPM 的参数化，这些 KL 项简化为带特定系数的噪声预测 MSE；Ho 去掉了系数（称之为"简单"损失），质量*反而提升了*。

## 动手实现

`code/main.py` 实现了一维 DDPM。数据是双模混合。"网络"是一个小型 MLP，接受 `(x_t, t)` 并输出预测的噪声。训练是那一行损失。采样迭代反向链。

### 步骤一：前向调度（封闭形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤二：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤三：一个训练步骤

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤四：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于 40 个时间步和 24 单元 MLP 的一维问题，这在约 200 个轮次内学会了双模混合。

## 时间步条件化

网络需要知道它在对哪个时间步去噪。两种标准选项：

- **正弦嵌入。** 类似 Transformer 位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP 传递，广播到网络中。
- **FiLM / 组归一化条件化。** 在每个块将嵌入投影为每通道缩放/偏置（FiLM）。

我们的玩具代码使用正弦 → 拼接。生产 U-Net 使用 FiLM。

## 常见陷阱

- **调度影响很大。** 线性 `β` 是 DDPM 默认值，但余弦调度（Nichol & Dhariwal，2021）在相同计算量下给出更好的 FID。如果质量停滞，切换调度。
- **时间步嵌入很脆弱。** 将原始 `t` 作为浮点数传入对玩具一维有效，但对图像无效；始终使用适当的嵌入。
- **V 预测 vs ε 预测。** 对于窄域范围（很小或很大的 t），`ε` 的信噪比很差。V 预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都使用它。
- **无分类器引导（CFG）。** 推理时，同时计算条件和无条件 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，`w ≈ 3-7`。见第 08 课。
- **1000 步太多了。** 生产使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## 生产使用

| 角色 | 2026 年典型技术栈 |
|------|----------------|
| 图像像素空间扩散（小型、玩具） | DDPM + U-Net |
| 图像潜在扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜在扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜在扩散 | Encodec + 扩散 Transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用生成骨干。流匹配（第 13 课）是 2024-2026 年的竞争者，在相同质量下通常在推理速度上胜出。

## 上手实践

保存 `outputs/skill-diffusion-trainer.md`。该技能接受数据集 + 计算预算，输出：调度（线性/余弦/sigmoid）、预测目标（ε/v/x）、步数、引导尺度、采样器家族和评估协议。

## 练习

1. **简单。** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的视觉直方图）如何退化？在什么 T 值下双模结构崩溃？
2. **中等。** 从 ε 预测切换到 v 预测。重新推导反向步骤。比较最终样本质量。
3. **困难。** 添加无分类器引导。对类别标签 `c ∈ {0, 1}` 进行条件化，训练时 10% 的时间丢弃它，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 时测量条件模式命中率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 前向过程（Forward process） | "添加噪声" | 固定马尔可夫链 `q(x_t | x_{t-1})` 破坏数据。 |
| 反向过程（Reverse process） | "去噪" | 学习到的链 `p_θ(x_{t-1} | x_t)` 重建数据。 |
| β 调度（β schedule） | "噪声阶梯" | 每步方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出 `x_0` 到 `x_t` 的封闭形式。 |
| 简单损失（Simple loss） | "MSE 噪声" | `||ε - ε_θ(x_t, t)||²`；所有变分推导都归结到这里。 |
| ε 预测（ε-prediction） | "预测噪声" | 输出是添加的噪声；标准 DDPM。 |
| V 预测（V-prediction） | "预测速度" | 输出是 `α·ε - σ·x`；在 t 上条件更好。 |
| DDPM | "那篇论文" | Ho 等 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同训练目标。 |
| 无分类器引导（CFG） | "CFG" | 混合条件和无条件噪声预测以放大条件化。 |

## 生产注意：扩散推理是一个步数问题

DDPM 论文运行 T=1000 个反向步骤。没有人在生产中这样部署。每个真实推理栈选择以下三种策略之一——每种都清晰地对应于"延迟来自哪里"的生产框架：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16 步）。反向循环的即插即用替换；训练好的 `ε_θ` 权重不变。将延迟削减 20-50 倍。
2. **蒸馏。** 训练一个学生在更少步骤中匹配教师：渐进式蒸馏（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。将延迟再削减 5-10 倍，需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。将每步延迟削减约 2 倍。与 (1) 和 (2) 叠加。

对于生产扩散服务器，预算对话与生产文献对 LLM 的描述相同：延迟是 `步数 × 步骤成本 + VAE 解码`，吞吐量是 `批大小 × (步数 × 步骤成本)^-1`。TTFT 很小（一步）；TPOT 等效于完整响应时间，因为从用户角度来看图像生成是"一次性完成"的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散论文，超前于其时代
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少步骤
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习方差
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一符号，最清晰的方案

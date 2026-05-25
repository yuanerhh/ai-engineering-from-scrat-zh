# 自编码器与变分自编码器（VAE）

> 普通自编码器压缩然后重建。它在记忆。它不会生成。加一个技巧——强迫编码看起来像高斯分布——你就得到了一个采样器。这个单一技巧，即 `z = μ + σ·ε` 的重参数化，是你 2026 年使用的每个潜在扩散和流匹配图像模型都在输入端有一个 VAE 的原因。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 3 · 02（反向传播）、Phase 3 · 07（CNN）、Phase 8 · 01（分类）
**时长：** 约 75 分钟

## 问题背景

将 784 像素的 MNIST 数字压缩为 16 个数字的编码，然后重建。普通自编码器在重建 MSE 上表现优异，但编码空间是一团凹凸不平的混乱。在编码空间中随机选一个点，解码它，你会得到噪声。它没有采样器。它是一个伪装成生成模型的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑的分布，你可以从中采样——比如各向同性高斯 `N(0, I)`，(b) 解码任何样本都能产生一个合理的数字，(c) 编码器和解码器仍然压缩得很好。三个目标，一个架构，一个损失。

Kingma 的 2013 年 VAE 通过训练编码器输出分布 `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z` 来解决这个问题。推理时，丢掉编码器，采样 `z ~ N(0, I)`，解码。KL 惩罚是强制编码空间结构化的关键。

2026 年 VAE 很少独立发布——在原始图像质量上已被扩散模型超越——但它是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学习 VAE 就是在学习你使用的每个图像流水线的不可见的第一层。

## 核心概念

![自编码器 vs VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。这些定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样是不可微的。将样本重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一个非参数噪声——梯度通过 `μ` 和 `σ` 流动。

**损失。** 证据下界（ELBO），两个项：

```
损失 = 重建 + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项将 `x̂` 推向 `x`。KL 项将 `q(z|x)` 推向先验。两者相互权衡。小 β（<1）= 更锐利的样本，编码空间不那么高斯。大 β（>1）= 更干净的编码空间，更模糊的样本。β-VAE（Higgins 2017）使这个旋钮闻名，并启动了解缠研究。

**采样。** 推理时：抽取 `z ~ N(0, I)`，通过解码器前向传播。一次前向传播——不像扩散那样迭代采样。

## 动手实现

`code/main.py` 不使用 numpy 或 torch 实现了一个微型 VAE。输入是从 8 维双成分高斯混合中抽取的 8 维合成数据。编码器和解码器是单隐藏层 MLP。我们实现 tanh 激活、前向传播、损失和手写反向传播。不是生产代码——是教学。

### 步骤一：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而非 `σ`，这样网络输出是无约束的（σ 的 softplus 是一个陷阱——梯度在 σ ≈ 0 时消失）。

### 步骤二：重参数化和解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤三：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

精确的封闭形式 KL，因为两个分布都是高斯的。不要用数值积分。2026 年仍有人写蒙特卡洛 KL 估计的代码——这无缘无故地慢 3 倍。

### 步骤四：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 常见陷阱

- **后验塌缩（Posterior collapse）。** KL 项如此激进地将 `q(z|x) → N(0, I)`，以至于 `z` 不携带任何关于 `x` 的信息。修复：β 退火（从 β=0 开始，逐渐提升到 1）、自由位或跳过非活跃维度上的 KL。
- **模糊样本。** 高斯解码器似然意味着 MSE 重建，这是 L2（均值）的贝叶斯最优——一组合理数字的均值是模糊数字。修复：离散解码器（VQ-VAE、NVAE），或只将 VAE 用作编码器，在潜在表示上叠加扩散（这正是 Stable Diffusion 所做的）。
- **β 太大，太早。** 见后验塌缩。从 β≈0.01 开始并逐渐提升。
- **潜在维度太小。** 16 维对 MNIST 有效，256 维对 ImageNet 256²，2048 维对 ImageNet 1024²。Stable Diffusion 的 VAE 将 512×512×3 压缩为 64×64×4（空间面积压缩 32 倍，通道压缩 32 倍）。

## 生产使用

2026 年的 VAE 栈：

| 情况 | 选择 |
|------|------|
| 扩散的图像潜在编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜在编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜在表示 | Sora 的时空图块，Latte VAE，WAN VAE |
| 解缠表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜在（用于 Transformer 建模） | VQ-VAE、RVQ（残差量化） |
| 连续潜在（用于生成） | 普通 VAE，然后在该潜在空间上条件化流/扩散模型 |

潜在扩散模型是一个编码器和解码器之间有扩散模型的 VAE。VAE 做粗略压缩，扩散模型做重活。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen Transformer）也是同样的模式。

## 上手实践

保存 `outputs/skill-vae-trainer.md`。

该技能接受：数据集概况 + 潜在维度目标 + 下游用途（重建、采样或潜在扩散输入），并输出：架构选择（普通/β/VQ/RVQ）、β 调度、潜在维度、解码器似然（高斯 vs 分类）和评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的弗雷歇距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。哪个 β 对你的合成数据是帕累托最优的？
2. **中等。** 将高斯解码器似然替换为 Bernoulli 似然（交叉熵损失）。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为迷你 VQ-VAE：用 K=32 个条目的码本中的最近邻查找替换连续 `z`。比较重建 MSE 并报告有多少码本条目被使用（码本塌缩是真实存在的）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 自编码器（Autoencoder） | "编码-解码网络" | `x → z → x̂`，学习 MSE。不是生成模型。 |
| VAE | "带采样器的自编码器" | 编码器输出分布，KL 惩罚塑造编码空间。 |
| ELBO | "证据下界" | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`；当 `q = p(z|x)` 时紧。 |
| 重参数化（Reparameterization） | "`z = μ + σ·ε`" | 将随机节点重写为确定性 + 纯噪声。使通过采样的反向传播成为可能。 |
| 先验（Prior） | "`p(z)`" | 潜在表示的目标分布，通常为 `N(0, I)`。 |
| 后验塌缩（Posterior collapse） | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器必须凭空想象。 |
| β-VAE | "可调 KL 权重" | `损失 = recon + β·KL`。β 越大 = 解缠越多但越模糊。 |
| VQ-VAE | "离散潜在" | 用最近的码本向量替换连续 `z`；使 Transformer 建模成为可能。 |

## 生产注意：VAE 是扩散服务器中最热的路径

在 Stable Diffusion / Flux / SD3 流水线中，VAE 每个请求被调用两次——一次编码（如果做 img2img / 修复），一次解码。在 1024² 分辨率下，解码器传播通常是整个流水线中单个最大的激活内存峰值，因为它将 `128×128×16` 的潜在表示上采样回 `1024×1024×3`。两个实际后果：

- **对解码进行切片或平铺。** `diffusers` 提供了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。平铺用 `O(tile²)` 内存换取 `O(H·W)` 内存，代价是一个小的拼接伪影。在消费级 GPU 上处理 1024²+ 时是必须的。
- **解码器用 bf16，最终调整大小用 fp32 数值。** SD 1.x VAE 以 fp32 发布，在 1024²+ 下转为 fp16 时*默默产生 NaN*。SDXL 附带了 `madebyollin/sdxl-vae-fp16-fix`——总是优先使用 fp16-fix 变体或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解缠 β-VAE
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像 VAE
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准

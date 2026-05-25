# 潜在扩散与 Stable Diffusion

> 在 512×512 图像的像素空间运行扩散是计算资源的严重浪费。Rombach 等（2022）意识到你不需要所有 786k 维度来生成图像——你只需要足够捕捉语义结构，其余的交给单独的解码器。在 VAE 的潜在空间内运行扩散。这一个想法就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 8 · 02（VAE）、Phase 8 · 06（DDPM）、Phase 7 · 09（ViT）
**时长：** 约 75 分钟

## 问题背景

在 512² 进行像素空间扩散意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。每个采样步骤对 500M 参数 U-Net 约需 100 GFLOPS。五十步就是每张图像 5 TFLOPS。在十亿图像上训练，计算成本是荒唐的。

这些 FLOP 大多用于将感知上不重要的细节推过网络——有损 VAE 可以压缩掉的高频纹理。Rombach 的想法：一次性训练一个 VAE（*第一阶段*），冻结它，然后完全在 4 通道 64×64 的潜在空间中运行扩散（*第二阶段*）。相同的 U-Net。1/16 的像素。在可比质量下约减少 64 倍的 FLOP。

这就是 Stable Diffusion 的方案。SD 1.x / 2.x 在 `64×64×4` 潜在表示上使用了 860M U-Net，SDXL 在 `128×128×4` 上使用了 2.6B U-Net，SD3 将 U-Net 换成了使用流匹配的扩散 Transformer（DiT）。Flux.1-dev（Black Forest Labs，2024）部署了一个 12B 参数的 DiT-MMDiT。所有模型都运行在相同的两阶段基础上。

## 核心概念

![潜在扩散：VAE 压缩 + 潜在空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **第一阶段——VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴 8× 下采样 + 调整通道，使总潜在大小约为像素数的 1/16。损失 = 重建（L1 + LPIPS 感知）+ KL（权重小，因为我们不需要从 `z` 精确采样）。通常用对抗损失训练，使解码图像锐利。

2. **第二阶段——在 `z` 上进行扩散。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）对 `z_t` 进行去噪。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件化。** 两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个交叉注意力注入：每个 U-Net 块接受 `[Q = 图像特征, K = V = 文本 token]` 并将其混合。Token 是文本影响图像的唯一途径。

**损失函数与第 06 课相同。** 相同的 DDPM / 流匹配 MSE 噪声损失。只是切换了数据域。

## 架构变体

| 模型 | 年份 | 骨干 | 潜在形状 | 文本编码器 | 参数量 |
|------|------|------|---------|-----------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 token） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + 精细化器 | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏 | 128×128×4 | 同上 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | 蒸馏 MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

趋势：用 DiT（潜在图块上的 Transformer）替换 U-Net，扩展文本编码器（T5 在提示词遵循上胜过 CLIP），增加潜在通道数（4 → 16 提供更多细节空间）。

## 动手实现

`code/main.py` 将玩具一维"VAE"（恒等编码器 + 解码器，用于演示；真实的 VAE 是卷积网络）叠加在第 06 课的 DDPM 上，并添加了带无分类器引导的类别条件化。它表明无论你在原始一维值还是编码值上运行，相同的扩散损失都有效——这是关键洞见。

### 步骤一：编码器/解码器

```python
def encode(x):    return x * 0.5          # 玩具"压缩"到更小的尺度
def decode(z):    return z * 2.0
```

真实的 VAE 有训练好的权重。出于教学目的，这个线性映射足以说明扩散在 `z` 上操作而不关心原始数据空间。

### 步骤二：在 `z` 空间中进行扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` 解码。

### 步骤三：无分类器引导

训练时，10% 的时间丢弃类别标签（替换为 null token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样性），`w = 3` = 默认，`w = 7+` = 饱和/过度锐利。

### 步骤四：文本条件化（概念，非代码）

将类别标签替换为冻结文本编码器的输出。通过交叉注意力将文本嵌入输入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类条件扩散模型与 Stable Diffusion 之间唯一实质性的区别。

## 常见陷阱

- **VAE 尺度不匹配。** SD 1.x VAE 在编码后应用了一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这一点会让 U-Net 在方差极不正确的潜在表示上训练。每个检查点都带有一个这样的常数。
- **文本编码器悄悄出错。** SD3 需要 T5-XXL 支持 >=128 token，回退到仅 CLIP 会导致损失。始终检查 `use_t5=True` 否则提示词忠实度会大幅下降。
- **混用潜在空间。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL 潜在表示上训练的 LoRA 在 SD3 上不起作用。Hugging Face diffusers 0.30+ 拒绝加载不匹配的检查点。
- **CFG 太高。** `w > 10` 产生饱和、油腻的图像，并以牺牲多样性为代价过拟合提示词。最佳区间是 `w = 3-7`。
- **负向提示泄漏。** 空负向提示成为 null token；填写的负向提示成为 `ε_uncond`。这两者不同；某些流水线默默地默认为 null。

## 生产使用

2026 年的生产技术栈：

| 目标 | 推荐骨干 |
|------|---------|
| 窄域，配对数据，从头训练模型 | SDXL 微调（LoRA / 全量）——最快上线 |
| 开放域文本到图像，开放权重 | Flux.1-dev（12B，Apache / 非商业）或 SD3.5-Large |
| 最快推理，开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示词遵循，托管 | GPT-Image / DALL-E 3（仍在），Midjourney v7，Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）——原生接受图像 + 文本 |
| 研究，基线 | SD 1.5——古老但研究充分 |

## 上手实践

保存 `outputs/skill-sd-prompter.md`。该技能接受文本提示词 + 目标风格，输出：模型 + 检查点、CFG 尺度、采样器、负向提示词、分辨率、可选的 ControlNet/IP-Adapter 组合，以及逐步质量保证清单。

## 练习

1. **简单。** 在引导 `w ∈ {0, 1, 3, 7, 15}` 下运行 `code/main.py`。记录每类别的平均样本。在什么 `w` 值下，类别均值超过真实数据均值？
2. **中等。** 将玩具线性编码器换成带重建损失的 tanh-MLP 编码器/解码器对。在新的潜在表示上重新训练扩散。样本质量改变了吗？
3. **困难。** 用 diffusers 设置真实的 Stable Diffusion 推理：加载 `sdxl-base`，用 CFG=7 运行 30 步 Euler，计时。然后切换到 `sdxl-turbo`，4 步，CFG=0。相同主题，不同质量——描述发生了什么变化以及为什么。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 第一阶段（First stage） | "那个 VAE" | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段（Second stage） | "那个 U-Net" | 潜在空间上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调节条件强度。 |
| Null token | "空提示词嵌入" | 用于 `ε_uncond` 的无条件嵌入。 |
| 交叉注意力（Cross-attention） | "文本如何进入" | 每个 U-Net 块将文本 token 作为 K 和 V 进行注意力。 |
| DiT | "扩散 Transformer" | 用潜在图块上的 Transformer 替换 U-Net；扩展性更好。 |
| MMDiT | "多模态 DiT" | SD3 的架构：文本和图像流进行联合注意力。 |
| VAE 缩放因子（VAE scaling factor） | "魔法数字" | 将潜在表示除以约 5.4，使扩散在单位方差空间中操作。 |

## 生产注意：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是典型的"我有消费级 GPU，能否上线？"方案。关键是将生产推理文献列出的三旋钮方案应用于扩散 DiT：

1. **交错加载。** Flux 有三个永远不需要同时在显存中的网络：T5-XXL 文本编码器（fp32 约 10 GB）、CLIP-L（小型）、12B MMDiT 和 VAE。先编码提示词，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 每次只能容纳一个阶段。
2. **通过 bitsandbytes 进行 4 位量化。** 对 T5 编码器和 DiT 都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存减少 8 倍，文本到图像的质量下降不可感知。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 随着每次前向传播推进自动在 CPU 和 GPU 之间交换模块。增加 10-20% 延迟，但使流水线能够运行。

内存计算：`10 GB T5 / 8 = 1.25 GB` 量化，`12B 参数 × 0.5 字节 = 约 6 GB` 量化 DiT，加上激活。这是 TP=1 推理的极端情况——没有模型并行，最大量化。生产中会在 H100 上运行 TP=2 或 TP=4；对于单个开发笔记本，这是正确方案。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，MMDiT
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 系列
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 以上所有检查点的参考实现

# ControlNet、LoRA 与条件控制

> 单靠文本是笨拙的控制信号。ControlNet 让你克隆预训练的扩散模型，并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你只需训练 1000 万个参数就能微调 20 亿参数的模型。两者结合将 Stable Diffusion 从玩具变成了 2026 年每家机构都在使用的图像流水线。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 8 · 07（潜在扩散）、Phase 10（从零构建 LLM——LoRA 基础）
**时长：** 约 75 分钟

## 问题背景

"一个穿红裙的女人牵着狗在繁忙街道上行走"这样的提示词并没有给模型提供*狗在哪里*、*女人的姿态*或*街道透视*的信息。文本固定了你需要指定图像的约 10% 的信息。其余的是视觉的，无法用文字高效描述。

为每个信号（姿态、深度、Canny 边缘、分割）从头训练新的条件模型代价太高。你希望保持 2.6B 参数的 SDXL 骨干冻结，附加一个读取条件的小型侧网络，让它轻推骨干的中间特征。这就是 ControlNet。

你还想在不重新训练完整模型的情况下教模型新概念（你的脸、你的产品、你的风格）。你想要一个小 100 倍的增量。这就是 LoRA——插入现有注意力权重的低秩适配器。

ControlNet + LoRA + 文本 = 2026 年的从业者工具包。大多数生产图像流水线在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 核心概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器部分。冻结原始网络。训练克隆版本接受额外的条件输入（边缘、深度、姿态）。通过*零卷积*跳跃连接将克隆连回原始网络的解码器部分（初始化为零的 1×1 卷积——从无操作开始，学习一个增量）。

```
SD U-Net 解码器:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 从恒等变换开始——即使训练前也不会造成损害。在 100 万个（提示词、条件、图像）三元组上用标准扩散损失训练。

每个模态的 ControlNet 作为小型侧模型提供（SDXL 约 360M，SD 1.5 约 70M）。推理时可以组合：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等，2021）

对模型中任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层秩 4-16 是标准，重度微调秩 64-128。新参数数量：`2 · d · r` 而非 `d²`。对 `d=640` 的 SDXL 注意力，`r=16`：每个适配器 20k 参数而非 410k——减少 20 倍。整个模型：LoRA 通常是 20-200MB，而基础模型是 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常范围。多个 LoRA 可以叠加相加（注意它们以非线性方式交互）。

### IP-Adapter（Ye 等，2023）

一个接受*图像*作为条件（与文本一起）的小型适配器。使用 CLIP 图像编码器产生图像 token，将其与文本 token 一起注入交叉注意力。每个基础模型约 20MB。让你可以"用这个参考图像的风格生成图像"，而无需 LoRA。

## 可组合性矩阵

| 工具 | 控制内容 | 大小 | 使用时机 |
|------|---------|------|---------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主题、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 来自参考图像的风格或主题 | 20MB | 文本无法描述的外观 |
| Textual Inversion | 将单个概念作为新 token | 10KB | 旧方法，大多被 LoRA 取代 |
| DreamBooth | 对主题进行完整微调 | 2-5GB | 强身份，高计算量 |
| T2I-Adapter | 更轻量的 ControlNet 替代 | 70MB | 边缘设备、推理预算 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者结合使用。

## 动手实现

`code/main.py` 在一维上模拟了两种机制：

1. **LoRA。** 一个预训练线性层 `W`。冻结它。训练低秩 `B @ A` 使 `W + BA` 匹配目标线性层。证明 `r = 1` 足以完美学习秩为 1 的修正。

2. **ControlNet-lite。** 一个"冻结基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出通过一个初始化为零的可学习标量进行门控（我们版本的零卷积）。训练并观察门控值如何上升。

### 步骤一：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 是冻结的；A, B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤二：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在步骤 0 时，输出与基础相同。早期训练缓慢更新 `gate`——没有灾难性漂移。

## 常见陷阱

- **LoRA 过度缩放。** `α = 2` 或 `α = 3` 是常见的"让它更强"的技巧，会产生过度风格化/破损的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 同时使用权重 1.0 的姿态 ControlNet 和权重 1.0 的深度 ControlNet 通常会过冲。权重之和 ≈ 1.0 是安全的默认值。
- **LoRA 用在错误的基础上。** SDXL LoRA 在 SD 1.5 上悄悄变成无操作，因为注意力维度不匹配。Diffusers 0.30+ 会发出警告。
- **Textual Inversion 漂移。** 在一个检查点上训练的 token 在另一个检查点上严重漂移。LoRA 的可移植性更好。
- **LoRA 权重合并与存储。** 你可以将 LoRA 烘焙到基础模型权重中以加快推理（无运行时加法），但会失去在运行时缩放 `α` 的能力。保留两个版本。

## 生产使用

| 目标 | 2026 年流水线 |
|------|-------------|
| 复现品牌艺术风格 | 在约 30 张精选图像上以秩 32 训练的 LoRA |
| 在生成图像中放入我的脸 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图像 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 图像修复（第 09 课） |
| 快速一步风格 | SDXL-Turbo 上的 LCM-LoRA |

## 上手实践

保存 `outputs/skill-sd-toolkit-composer.md`。该技能接受任务（输入资源：提示词、可选参考图像、可选姿态、可选深度、可选涂鸦）并输出工具栈、权重和可复现的种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 变到 4。在什么秩时，LoRA 完全匹配秩为 2 的目标增量？
2. **中等。** 对两个目标变换分别训练两个 LoRA。一起加载并展示它们的叠加交互。什么时候交互会打破线性？
3. **困难。** 用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。随着栈权重变化，测量 FID 与提示词遵循度之间的权衡。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| ControlNet | "空间控制" | 克隆编码器 + 零卷积跳跃；读取条件图像。 |
| 零卷积（Zero convolution） | "从恒等变换开始" | 初始化为零的 1×1 卷积；ControlNet 从无操作开始。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；参数比完整微调少 100 倍。 |
| 秩 r（rank r） | "那个旋钮" | LoRA 压缩；4-16 是典型值，重度个性化用 64+。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 的小型图像条件化适配器。 |
| DreamBooth | "完整主题微调" | 在约 30 张主题图像上训练完整模型。 |
| Textual Inversion | "新 token" | 只学习一个新词嵌入；旧方法，大多被取代。 |

## 生产注意：LoRA 热交换、ControlNet 通道、多租户服务

真实的文本到图像 SaaS 在同一个基础检查点上服务数百个 LoRA 和十几个 ControlNet。服务问题很像 LLM 多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下涵盖了 LLM 的情形）：

- **热交换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础中可以使每步推理快约 3-5%，但会冻结 `α` 和基础。将 LoRA 以秩 r 增量形式保存在显存中；diffusers 提供了 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于每请求激活。交换成本是 `2 · d · r · num_layers` 个权重——MB 级别，亚秒级。
- **ControlNet 作为第二注意力通道。** 克隆的编码器与基础并行运行。两个权重各为 1.0 的 ControlNet = 每步两次额外前向传播，而非一次合并传播。批大小空间呈二次方下降。为每个活跃的 ControlNet 预算约 1.5× 步骤成本。
- **量化的 LoRA 也可以。** 如果你量化了基础（见第 07 课，8GB 上的 Flux），LoRA 增量也可以干净地量化到 8 位或 4 位。QLoRA 风格的加载让你可以在 4 位 Flux 基础上叠加 5-10 个 LoRA 而不会超出内存。

Flux 专项：将基础量化为 4 位；在量化基础上以 `weight_name="pytorch_lora_weights.safetensors"` 叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）仍然有效。这是 2026 年大多数 SaaS 机构使用的方案。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；移植到扩散）
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — 更轻量的 ControlNet 替代
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线

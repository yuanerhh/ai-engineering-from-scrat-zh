# StyleGAN

> 大多数生成器同时将 `z` 搅入每一层。StyleGAN 将其分离：先将 `z` 映射到中间表示 `w`，再通过 AdaIN 在每个分辨率层级*注入* `w`。这一单一改变解开了潜在空间，使照片级真实人脸成为七年来的已解决问题。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 8 · 03（GAN）、Phase 4 · 08（归一化）、Phase 3 · 07（CNN）
**时长：** 约 45 分钟

## 问题背景

DCGAN 通过一叠转置卷积将 `z` 映射到图像。问题在于：`z` 控制一切——姿态、光照、身份、背景——全部纠缠在一起。沿 `z` 的一个轴移动，四者皆变。你无法让模型"同一个人，不同姿态"，因为表示没有这样分解。

Karras 等（2019，NVIDIA）提出：停止将 `z` 直接输入卷积层。将一个常量 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP，将 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率处注入 `w`：归一化每个卷积特征图，然后通过 `w` 的仿射投影进行缩放和偏移。为随机细节添加逐层噪声（皮肤毛孔、发丝）。

结果：`W` 对"高层风格"（姿态、身份）与"细节风格"（光照、颜色）有大致正交的轴。你可以通过对低分辨率层使用图像 A 的 `w`、对高分辨率层使用图像 B 的 `w` 来在两张图像之间交换风格。这解锁了编辑、跨域风格化，以及整个"StyleGAN 反演"系列研究。

## 核心概念

![StyleGAN：映射网络 + AdaIN + 逐层噪声](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯分布——它学习到数据适配的形状。

**合成网络。** 从一个学习到的常量 `4×4×512` 开始。每个分辨率块：`上采样 → 卷积 → AdaIN(w_i) → 噪声 → 卷积 → AdaIN(w_i) → 噪声`。分辨率翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。对每个特征图进行归一化，然后重设风格。这里的"风格"是特征图的一阶和二阶统计量。

**逐层噪声。** 添加到每个特征图的单通道高斯噪声，由每通道学习到的因子缩放。在不影响全局结构的情况下控制随机细节。

**截断技巧（Truncation trick）。** 推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是多个样本上的平均 `w`。`ψ < 1` 以多样性换取质量。几乎所有 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|------|------|------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长。 |
| StyleGAN2 | 2020 | 权重解调替代 AdaIN（修复水滴状伪影）；跳跃/残差架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无混叠卷积 + 等变核；消除纹理粘附到像素网格的问题。 |
| StyleGAN-XL | 2022 | 类别条件，1024²，ImageNet。 |
| R3GAN | 2024 | 强化正则化重新包装；在 FFHQ-1024 上以 1/20 参数量缩小与扩散的差距。 |

2026 年，StyleGAN3 仍是以下场景的默认选择：(a) 高帧率窄域照片级真实感，(b) 少样本域适应（在 100 张图像的新数据集上训练，冻结映射网络），(c) 基于反演的编辑（找到重建真实照片的 `w`，然后编辑该 `w`）。对于开放域文本到图像，它不是正确工具——扩散才是。

## 动手实现

`code/main.py` 在一维中实现了一个玩具"style-GAN lite"：一个映射 MLP、一个以学习常量向量为起点并用 `w` 派生的缩放/偏置对其调制的合成函数，以及逐层噪声。它表明通过仿射调制注入 `w` 与将 `z` 拼接到生成器输入相比效果相当或更好。

### 步骤一：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤二：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的缩放和偏置通过线性投影来自 `w`。

### 步骤三：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每通道的 sigma 是可学习的。

## 常见陷阱

- **水滴状伪影。** StyleGAN 1 在特征图中产生斑点状水滴，因为 AdaIN 将均值归零。StyleGAN 2 的权重解调通过对卷积权重进行缩放来修复它。
- **纹理粘附。** StyleGAN 1 和 2 的纹理跟随像素坐标而非对象坐标（在插值时可见）。StyleGAN 3 的无混叠卷积使用窗口化 sinc 滤波器修复了这一问题。
- **模式覆盖率。** 截断 `ψ < 0.7` 看起来干净，但从映射网络范围的狭窄锥体中采样；如果需要多样性，使用 `ψ = 1.0`。
- **反演有损失。** 将真实照片反演到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。多次迭代后结果会漂移。

## 生产使用

| 使用场景 | 方案 |
|---------|------|
| 照片级真实人脸（动漫、产品、窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片编辑人脸 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 人脸交换/重演 | StyleGAN + 编码器 + 混合 |
| 虚拟形象流水线 | StyleGAN3 带 ADA 用于低数据微调 |
| 从少量图像进行域适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要——使用扩散 |

对于答案是"一个人脸照片"的产品级演示，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和相同质量条件下的锐利度上胜过扩散。

## 上手实践

保存 `outputs/skill-stylegan-inversion.md`。该技能接受一张真实照片，输出：反演方法（e4e / ReStyle / HyperStyle）、预期潜在损失、编辑预算（在 `W` 中可以移动多远才会出现伪影），以及已知有效的编辑方向列表（年龄、表情、姿态）。

## 练习

1. **简单。** 分别在 `adain_on=True` 和 `adain_on=False` 下运行 `code/main.py`。比较固定潜在变量与扰动潜在变量下输出的分布范围。
2. **中等。** 实现混合正则化：对一个训练批次，计算 `w_a`、`w_b`，并在合成的前半部分应用 `w_a`、后半部分应用 `w_b`。解码器是否学到了解耦的风格？
3. **困难。** 使用预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标注样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告在身份漂移之前可以推进多远。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 映射网络（Mapping network） | "那个 MLP" | `f: Z → W`，8 层，将潜在几何与数据统计解耦。 |
| W 空间（W space） | "风格空间" | 映射网络的输出；大致解耦。 |
| AdaIN | "自适应实例归一化" | 归一化特征图，然后通过 `w` 投影进行缩放 + 偏移。 |
| 截断技巧（Truncation trick） | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以多样性换质量。 |
| 路径长度正则化（Path-length regularization） | "PL 正则" | 惩罚 `w` 单位变化引起的大图像变化；使 `W` 更平滑。 |
| 权重解调（Weight demodulation） | "StyleGAN2 修复" | 对卷积权重而非激活值归一化；消除水滴状伪影。 |
| 无混叠（Alias-free） | "StyleGAN3 技巧" | 窗口化 sinc 滤波器；消除纹理粘附到像素网格的问题。 |
| 反演（Inversion） | "为真实图像找 w" | 优化或编码 `x → w` 使 `G(w) ≈ x`。 |

## 生产注意：StyleGAN 在 2026 年仍然在生产中运行

StyleGAN3 在 4090 上生成 1024² FFHQ 人脸不到 10 毫秒——`num_steps = 1`，无 VAE 解码，无交叉注意力传播。在生产术语中，这是任何图像生成器的延迟下限。相同分辨率下 50 步 SDXL + VAE 解码流水线约需 3 秒。这是 **300 倍的差距**，对于窄域产品（虚拟形象服务、身份证件流水线、证件照生成），它在总拥有成本（TCO）上胜出。

两个实际运营后果：

- **无调度器，无批处理器。** 在目标占用率下静态批是最优的。连续批处理（对 LLM 和扩散至关重要）带来零收益，因为每个请求需要相同的 FLOP。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 从映射网络范围的狭窄锥体中采样。这是服务层对样本方差的唯一控制杠杆。峰值负载时降低 `ψ`，为高级用户提高 `ψ`。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 方案

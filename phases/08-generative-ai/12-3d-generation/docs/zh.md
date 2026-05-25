# 3D 生成

> 3D 是 2D 到 3D 迁移效果最强的模态。2023 年的突破是 3D 高斯溅射（3D Gaussian Splatting）。2024-2026 年的生成进展在其上叠加了多视角扩散 + 3D 重建，从单个提示词或照片生成对象和场景。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 4（视觉）、Phase 8 · 07（潜在扩散）
**时长：** 约 45 分钟

## 问题背景

3D 内容的难点：

- **表示。** 网格、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯。每种都有权衡。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL，2023）有约 1000 万个对象，大多质量较低。
- **内存。** 512³ 的体素网格有 1.28 亿个体素；有用的场景 NeRF 每条射线需要 100 万个样本。生成比重建更难。
- **监督信号。** 对于 2D 图像，你有像素。对于 3D，你通常只有少量 2D 视角，必须上升到 3D。

2026 年的技术栈将这两个问题分开。首先，用扩散模型生成*2D 多视角图像*。其次，将*3D 表示*（通常是高斯溅射）拟合到这些图像上。

## 核心概念

![3D 生成：多视角扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示：3D 高斯溅射（Kerbl 等，2023）

将场景表示为约 100 万个 3D 高斯的点云。每个高斯有 59 个参数：位置（3）、协方差（6，或四元数 4 + 尺度 3）、不透明度（1）、球谐颜色（3 度时 48，0 度时 3）。

渲染 = 投影 + alpha 合成。速度快（4090 上 1080p 约 100fps）。可微分。通过梯度下降对真实照片进行拟合。场景在消费级 GPU 上 5-30 分钟内拟合完成。

其上的两个 2023-2024 创新：
- **生成式高斯溅射。** LGM、LRM、InstantMesh 等模型直接从一张或几张图像预测高斯点云。
- **4D 高斯溅射。** 带每帧偏移的高斯用于动态场景。

### 多视角扩散

微调预训练图像扩散模型，从文本提示词或单张图像生成同一对象的多个一致视角。Zero123（Liu 等，2023）、MVDream（Shi 等，2023）、SV3D（Stability，2024）、CAT3D（谷歌，2024）。通常输出围绕对象的 4-16 个视角，通过高斯溅射或 NeRF 上升到 3D。

### 文本到 3D 流水线

| 模型 | 输入 | 输出 | 时间 |
|------|------|------|------|
| DreamFusion（2022） | 文本 | 通过 SDS 的 NeRF | 每个资产约 1 小时 |
| Magic3D | 文本 | 网格 + 纹理 | 约 40 分钟 |
| Shap-E（OpenAI，2023） | 文本 | 隐式 3D | 约 1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | 约 30 分钟 |
| LRM（Meta，2023） | 图像 | 三平面 | 约 5 秒 |
| InstantMesh（2024） | 图像 | 网格 | 约 10 秒 |
| SV3D（Stability，2024） | 图像 | 新视角 | 约 2 分钟 |
| CAT3D（谷歌，2024） | 1-64 张图像 | 3D NeRF | 约 1 分钟 |
| TripoSR（2024） | 图像 | 网格 | 约 1 秒 |
| Meshy 4（2025） | 文本 + 图像 | PBR 网格 | 约 30 秒 |
| Rodin Gen-1.5（2025） | 文本 + 图像 | PBR 网格 | 约 60 秒 |
| 腾讯混元3D 2.0（2025） | 图像 | 网格 | 约 30 秒 |

2025-2026 方向：带 PBR 材质、适合游戏引擎的直接文本到网格模型。多视角扩散中间步骤仍然是通用对象的最佳性能方案。

### NeRF（背景知识）

神经辐射场（Mildenhall 等，2020）。一个小型 MLP 接受 `(x, y, z, 视角方向)` 并输出 `(颜色, 密度)`。通过沿射线积分进行渲染。在质量上胜过基于网格的新视角合成，但渲染速度慢 100-1000 倍。对于大多数实时用途已被高斯溅射取代，但在研究中仍占主导。

## 动手实现

`code/main.py` 实现了玩具 2D"高斯溅射"拟合：将合成目标图像（平滑渐变）表示为 2D 高斯溅射之和。通过梯度下降优化位置、颜色和协方差以匹配目标。你将看到两个核心操作：前向渲染（溅射 + alpha 合成）和梯度下降拟合。

### 步骤一：2D 高斯溅射

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤二：通过求和溅射进行渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D 高斯溅射按深度对高斯排序并按顺序进行 alpha 合成。我们的 2D 玩具只是求和。

### 步骤三：梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常见陷阱

- **视角不一致。** 如果独立生成 4 个视角，它们对对象结构的描述不一致，3D 拟合会模糊。修复：带共享注意力的多视角扩散。
- **背面幻觉。** 单张图像 → 3D 必须虚构看不见的一面。质量差异很大。
- **高斯溅射爆炸。** 无约束训练会增长到 1000 万个溅射并过拟合。密集化 + 剪枝启发式（来自 3D-GS 原始论文）是必不可少的。
- **拓扑问题。** 来自隐式场（SDF）的网格经常有孔或自相交。在交付前运行重网格化（例如 Blender 的体素重网格化）。
- **训练数据许可证。** Objaverse 有混合许可证；商业使用因模型而异。

## 生产使用

| 任务 | 2026 年选择 |
|------|-----------|
| 从照片重建场景 | 高斯溅射（3DGS、Gsplat、Scaniverse） |
| 游戏用文本到 3D 对象 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | 混元3D 2.0、TripoSR、InstantMesh |
| 少量图像的新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D 高斯溅射 |
| 虚拟形象/着装人体 | Gaussian Avatar、HUGS |
| 研究/SOTA | 上周发布的最新成果 |

对于在游戏或电商流水线中交付生产级 3D：Meshy 4 或 Rodin Gen-1.5 输出可直接导入 Unity/Unreal 的 PBR 网格。

## 上手实践

保存 `outputs/skill-3d-pipeline.md`。该技能接受 3D 简报（输入：文本/一张图像/少量图像；输出：网格/溅射/NeRF；用途：渲染/游戏/VR），输出：流水线（多视角扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 分别用 4、16、64 个高斯运行 `code/main.py`。报告相对于目标的最终 MSE。
2. **中等。** 扩展为彩色高斯（RGB）。确认重建匹配目标颜色模式。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片的采集中重建真实对象。报告拟合时间和保留视角上的最终 SSIM。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 3D 高斯溅射（3DGS） | "3DGS" | 场景表示为 3D 高斯点云；可微分 alpha 合成渲染。 |
| NeRF | "神经辐射场" | 输出 3D 点处颜色 + 密度的 MLP；通过射线积分渲染。 |
| 三平面（Triplane） | "三个 2D 平面" | 将 3D 分解为三个 2D 轴对齐特征网格；比体积方法更便宜。 |
| SDS | "分数蒸馏采样" | 使用 2D 扩散分数作为伪梯度训练 3D 模型。 |
| 多视角扩散（Multi-view diffusion） | "同时多个视角" | 输出一批一致摄像机视角的扩散模型。 |
| PBR | "基于物理的渲染" | 带漫反射率、粗糙度、金属度、法线通道的材质。 |
| 密集化（Densification） | "增长溅射" | 3DGS 训练启发式：在高梯度区域分裂/克隆溅射。 |

## 生产注意：3D 尚无统一基础设施

与图像（潜在扩散 + DiT）和视频（时空 DiT）不同，2026 年 3D 没有单一主导运行时。生产决策树因表示而分叉：

- **NeRF / 三平面。** 推理是射线步进 + 每个采样的 MLP 前向传播。512² 渲染需要数百万次 MLP 前向传播。积极地批处理射线样本；SDPA/xformers 适用。
- **多视角扩散 + LRM 重建。** 两阶段流水线。阶段 1（多视角 DiT）是第 07 课那样的扩散服务器。阶段 2（LRM Transformer）是对视角的一次性前向传播。整体延迟配置是"扩散 + 一次性"——相应地选择每阶段的服务原语。
- **SDS / DreamFusion。** 每资产优化，而非推理。构建作业，而非请求处理器。

对于大多数 2026 年的产品，正确答案是"按请求运行多视角扩散模型，异步重建为 3DGS，服务 3DGS 用于实时查看"。这将工作负载清晰地分割为 GPU 推理服务器（快速）和离线优化器（慢速）。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视角扩散
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D

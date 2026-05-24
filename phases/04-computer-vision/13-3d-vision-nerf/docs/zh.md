# 3D 视觉 — 点云与 NeRF

> 3D 视觉有两种形态。点云是传感器的原始输出。NeRF 是学到的体积场。两者都在回答"空间中什么东西在哪里"。

**类型：** 学习 + 构建
**语言：** Python
**前置条件：** Phase 4 第 03 课（CNN），Phase 1 第 12 课（张量操作）
**时长：** 约 45 分钟

## 学习目标

- 区分显式表示（点云、网格、体素）和隐式表示（有符号距离场、NeRF），并了解各自的适用场景
- 理解 PointNet 的对称函数技巧，使神经网络对无序点集具有排列不变性
- 追踪 NeRF 的前向传播：光线投射、体积渲染、位置编码、MLP 密度+颜色头
- 使用 `nerfstudio` 或 `instant-ngp` 从少量有姿态的图像进行预训练的 3D 重建

## 问题背景

相机产生 2D 图像，激光雷达（LIDAR）产生一组无序的 3D 点，运动恢复结构（SfM）流程产生稀疏的 3D 关键点云，NeRF 从少数有姿态的图像中重建完整的 3D 场景。这些都是"视觉"，但都不像 CNN 所需的密集张量。

3D 视觉之所以重要，是因为几乎所有高价值机器人任务都在 3D 空间中进行：抓取、避障、导航、AR 遮挡、3D 内容捕获。只懂 2D 图像的视觉工程师无缘这一领域中增长最快的部分（AR/VR 内容、机器人、自动驾驶栈、用于房地产或建筑的 NeRF 三维重建）。

两种表示因不同原因而占主导地位。点云是传感器直接给出的数据。NeRF 及其后继者（3D 高斯泼溅、神经 SDF）是要求神经网络学习一个场景时所得到的结果。

## 核心概念

### 点云

点云是 R³ 中 N 个点的无序集合，每个点可选地带有特征（颜色、强度、法线）。

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

没有网格，没有连接关系。这使得神经网络难以处理：

- **排列不变性（Permutation invariance）** — 输出不能依赖于点的顺序。
- **可变 N** — 单一模型必须处理不同大小的点云。

PointNet（Qi 等，2017）用一个思路解决了这两个问题：对每个点应用共享 MLP，然后用对称函数（最大池化）聚合。结果是一个不依赖于顺序的固定大小向量。

```
f(P) = max_{p in P} MLP(p)
```

这就是 PointNet 的全部核心。更深的变体（PointNet++、Point Transformer）增加了层次采样和局部聚合，但对称函数技巧保持不变。

### PointNet 架构

```mermaid
flowchart LR
    PTS["N 个点<br/>(x, y, z)"] --> MLP1["共享 MLP<br/>(64, 64)"]
    MLP1 --> MLP2["共享 MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["最大池化<br/>（对称）"]
    MAX --> FEAT["全局特征<br/>(1024,)"]
    FEAT --> FC["MLP 分类器"]
    FC --> CLS["类别 logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

"共享 MLP"意味着同一个 MLP 独立地作用于每个点。为了效率，实现为在点维度上的 1×1 卷积。

### 神经辐射场（Neural Radiance Fields，NeRF）

NeRF（Mildenhall 等，2020）对"能否从 N 张照片重建一个 3D 场景？"这一问题的回答，是用一个本身就是场景的神经网络。该网络将 `(x, y, z, 观察方向)` 映射到 `(密度, 颜色)`。渲染新视角就是在这个网络上运行光线投射循环。

```
NeRF MLP：(x, y, z, theta, phi) -> (sigma, r, g, b)

渲染新视角像素 (u, v) 的过程：
  1. 从相机通过像素 (u, v) 投射一条光线
  2. 沿光线在距离 t_1, t_2, ..., t_N 处采样点
  3. 在每个点处查询 MLP
  4. 用 (1 - exp(-sigma * dt)) 加权合成颜色
  5. 求和得到渲染像素颜色
```

损失将渲染像素与训练照片中的真实像素进行比较。通过渲染步骤反向传播更新 MLP。无需 3D 真实标注，无需显式几何——场景存储在 MLP 权重中。

### NeRF 中的位置编码

在 `(x, y, z)` 上使用普通 MLP 无法表示高频细节，因为 MLP 在频谱上偏向低频。NeRF 通过在 MLP 前将每个坐标编码为傅里叶特征向量来解决这个问题：

```
gamma(p) = (sin(2^0 * pi * p), cos(2^0 * pi * p), sin(2^1 * pi * p), cos(2^1 * pi * p), ...)
```

频率层级最多到 L=10。这与 Transformer 用于位置的技巧相同，也再次出现在扩散模型时间条件中（第 10 课）。没有它，NeRF 看起来会很模糊。

### 体积渲染

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i` 是透射率（transmittance）——光线到达第 i 点时剩余的比例。`(1 - exp(-sigma_i * delta_i))` 是第 i 点的不透明度（opacity）。`c_i` 是颜色。最终像素是沿光线的加权求和。

### NeRF 的后继者

纯 NeRF 训练慢（数小时）、渲染慢（每张图像数秒）。此后的发展谱系：

- **Instant-NGP**（2022）— 哈希网格编码替代 MLP 的位置输入；训练仅需数秒。
- **Mip-NeRF 360** — 处理无界场景和抗锯齿。
- **3D 高斯泼溅（3D Gaussian Splatting）**（2023）— 用数百万个 3D 高斯函数替代体积场；训练仅需数分钟，实时渲染。当前的生产默认方案。

2026 年几乎所有真实的 NeRF 产品实际上都是 3D 高斯泼溅。但其背后的心智模型仍然是 NeRF。

### 数据集与基准

- **ShapeNet** — 3D CAD 模型作为点云的分类和分割。
- **ScanNet** — 用于分割的真实室内扫描。
- **KITTI** — 用于自动驾驶的室外激光雷达点云。
- **NeRF Synthetic** / **Blended MVS** — 用于视角合成的有姿态图像数据集。
- **Mip-NeRF 360** 数据集 — 无界真实场景。

## 动手实现

### 步骤一：PointNet 分类器

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — 为 Conv1d 转置
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

约 160 万个参数。每个点云处理 1024 个点。

### 步骤二：位置编码

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

乘以 `2^l * pi` 产生逐渐升高的频率。

### 步骤三：微型 NeRF MLP

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

与原始 NeRF（深度为 8 的两个 MLP 主干）相比非常精简，足以演示架构。

### 步骤四：沿光线的体积渲染

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) 沿光线的距离
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

一条光线，64 个采样点，合成为单个 RGB 像素和深度值。

## 生产实践

实际工作中：

- `nerfstudio`（Tancik 等）— 当前 NeRF/Instant-NGP/高斯泼溅的参考库，带命令行和 Web 查看器。
- `pytorch3d`（Meta）— 可微渲染、点云工具、网格操作。
- `open3d` — 点云处理、配准、可视化。

对于部署，3D 高斯泼溅已大量取代纯 NeRF，因为其渲染速度快 100 倍，重建质量相当。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 点云（Point cloud） | "激光雷达的 3D 点" | (x, y, z) 的无序集合 + 每点可选特征 |
| PointNet | "第一个点云神经网络" | 每点共享 MLP + 对称（最大）池化；在构造上具有排列不变性 |
| NeRF | "充当场景的 MLP" | 将 (x, y, z, 方向) 映射到 (密度, 颜色) 的网络；通过光线投射渲染 |
| 位置编码（Positional encoding） | "傅里叶特征" | 在多个频率上将每个坐标编码为 sin/cos，以克服 MLP 的低频偏置 |
| 体积渲染（Volumetric rendering） | "光线积分" | 使用透射率和 alpha 将光线上的采样点合成为单个像素 |
| Instant-NGP | "哈希网格 NeRF" | 用多分辨率哈希网格替换 NeRF 的坐标 MLP；速度快 100-1000 倍 |
| 3D 高斯泼溅（3D Gaussian splatting） | "数百万个高斯函数" | 场景 = 3D 高斯基元的集合；实时渲染，训练仅需数分钟 |
| SDF | "有符号距离场" | 返回到最近表面的有符号距离的函数；另一种隐式表示 |

## 延伸阅读

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) — 排列不变性分类器
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) — 使从照片进行 3D 重建成为神经网络问题的论文
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989) — 哈希网格，加速 1000 倍
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) — 在生产中取代 NeRF 的架构

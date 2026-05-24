# 从零实现 3D 高斯溅射

> 场景是数百万个 3D 高斯的云。每个高斯都有位置、朝向、尺度、不透明度，以及随观察方向变化的颜色。光栅化它们，通过光栅化做反向传播，完成。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 4 第 13 课（3D 视觉与 NeRF），Phase 1 第 12 课（张量操作），Phase 4 第 10 课（扩散基础，可选）
**时长：** 约 90 分钟

## 学习目标

- 解释为什么 3D 高斯溅射（3D Gaussian Splatting，3DGS）在 2026 年取代 NeRF 成为逼真 3D 重建的生产默认方案
- 说明每个高斯的六种参数（位置、旋转四元数、尺度、不透明度、球谐颜色、可选特征），以及各自占多少浮点数
- 从零实现 2D 高斯溅射光栅化器（使用 alpha 合成），并说明 3D 情况如何投影到同一个循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出为 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat` 格式

## 问题背景

NeRF 将场景存储为 MLP 的权重。每个渲染像素需要沿光线进行数百次 MLP 查询。训练需要数小时，渲染需要数秒，且权重无法编辑——如果想移动场景中的一把椅子，就必须重新训练。

3D 高斯溅射（Kerbl、Kopanas、Leimkühler、Drettakis，SIGGRAPH 2023）替代了这一切。场景是一组显式的 3D 高斯。渲染是 GPU 光栅化，速度超过 100fps。训练只需几分钟。编辑是直接的：平移一部分高斯就移动了椅子。到 2026 年，Khronos 组织批准了高斯溅射的 glTF 扩展，OpenUSD 26.03 推出了高斯溅射格式，Zillow 和 Apartments.com 用它渲染房地产，大多数关于 3D 重建的新研究论文都是核心 3DGS 思想的变体。

思维模型很简单，但数学有足够多的移动部件，以至于大多数介绍从光栅化开始，跳过了投影和球谐函数。本课从头构建整个系统——先是 2D 版本，然后扩展到 3D。

## 核心概念

### 高斯携带的信息

一个 3D 高斯是空间中的参数化斑块，具有以下属性：

```
位置          mu         (3,)    世界坐标中的中心
旋转          q          (4,)    编码朝向的单位四元数
尺度          s          (3,)    每轴对数尺度（渲染时指数化）
不透明度      alpha      (1,)    sigmoid 后的不透明度 [0, 1]
球谐系数      c_lm       (3 * (L+1)^2,)   视角相关颜色
```

旋转 + 尺度构建 3×3 协方差：`Sigma = R S S^T R^T`。这是高斯在 3D 中的形状。球谐函数（Spherical Harmonics，SH）让颜色随观察方向变化——镜面高光、微妙光泽、视角相关光晕——无需存储逐视角纹理。使用 SH 3 阶，每个颜色通道有 16 个系数，单颜色就需要每高斯 48 个浮点数。

一个场景通常有 100 万到 500 万个高斯。每个存储约 60 个浮点数（3 + 4 + 3 + 1 + 48 + 杂项）。五百万高斯的场景约 240 MB——远小于具有逐点纹理的等效点云，也比在高分辨率下重新渲染的 NeRF MLP 权重小一个数量级。

### 光栅化，而非光线行进

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>（位置、旋转、尺度、<br/>不透明度、球谐颜色）"] --> PROJ["投影到 2D<br/>（相机外参 + 内参）"]
    PROJ --> TILES["分配到图块<br/>（16×16 屏幕空间）"]
    TILES --> SORT["按深度排序<br/>（每图块）"]
    SORT --> ALPHA["前到后 alpha 合成"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全部对 GPU 友好。每个像素无需 MLP 查询。单张 RTX 3080 Ti 以 147fps 渲染 600 万个溅射。

### 投影步骤

世界位置 `mu`、3D 协方差 `Sigma` 的 3D 高斯投影为屏幕位置 `mu'`、2D 协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 观察变换（相机旋转 + 平移）
J = 透视投影在 mu' 处的雅可比矩阵
```

2D 高斯的足迹是一个椭圆，其轴是 `Sigma'` 的特征向量。椭圆内的每个像素都接收该高斯的贡献，权重为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha 合成规则

对于一个像素，覆盖它的高斯按从后到前排序（或等价地，使用翻转公式从前到后）。颜色与自 1980 年代以来所有半透明光栅化器相同的公式合成：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       到 i 的透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角相关颜色
```

这与 **NeRF 的体积渲染公式完全相同**，只是将沿光线的密集采样替换为显式稀疏高斯集合。这一等价性解释了为何渲染质量与 NeRF 相当——两者都在积分相同的辐射场方程。

### 为何可微分

每个步骤——投影、图块分配、alpha 合成、球谐评估——相对于高斯参数都是可微分的。给定真实图像，计算渲染像素损失，通过光栅化器反向传播，以梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 次迭代，高斯找到正确的位置、尺度和颜色。

### 致密化与剪枝

固定的高斯集合无法覆盖复杂场景。训练包含两种自适应机制：

- **克隆（Clone）**：当梯度幅度高但尺度小时，在当前位置克隆一个高斯——重建在此处需要更多细节。
- **分裂（Split）**：当梯度高且尺度大时，将一个大尺度高斯分裂为两个小高斯——一个大高斯太平滑，无法拟合该区域。
- **剪枝（Prune）**：删除不透明度低于阈值的高斯——它们没有贡献。

致密化每 N 次迭代运行一次。场景通常从约 10 万个初始高斯（从 SfM 点播种）增长到训练结束时的 100 万到 500 万个。

### 球谐函数一段话

视角相关颜色是单位球面上的函数 `c(direction)`。球谐函数是球面的傅里叶基。截断到 L 阶得到每通道 `(L+1)^2` 个基函数。对新视角评估颜色就是学到的球谐系数与在观察方向处评估的基之间的点积。0 阶 = 一个系数 = 常数颜色。3 阶 = 16 个系数 = 足以捕获朗伯着色、镜面反射和轻微反光。3DGS 论文默认使用 3 阶。

### 2026 年的生产技术栈

```
1. 采集         智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS   COLMAP 或 GLOMAP 生成相机位姿 + 稀疏点
3. 训练 3DGS   nerfstudio / gsplat / inria 官方 / PostShot（RTX 4090 上约 10-30 分钟）
4. 编辑         SuperSplat / SplatForge（清理浮动点，分割）
5. 导出         .ply -> glTF KHR_gaussian_splatting 或 .usd（OpenUSD 26.03）
6. 查看         Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 与生成式变体

- **4D 高斯溅射** — 高斯是时间的函数；用于体积视频（《超人》2026、A$AP Rocky 的"Helicopter"）。
- **生成式溅射** — 文本到溅射模型（World Labs 的 Marble）凭空生成整个场景。
- **3D 高斯无迹变换** — NVIDIA NuRec 用于自动驾驶仿真的变体。

## 动手实现

### 步骤一：2D 高斯

首先构建 2D 光栅化器。3D 情况在投影后归约到它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 对每个（高斯，像素）对计算二次型 `diff^T Sigma^-1 diff`。

### 步骤二：2D 溅射光栅化器

前到后 alpha 合成。2D 中深度没有意义，所以用每高斯一个可学习标量排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快——真实实现使用基于图块的 CUDA 核——但数学完全正确且完全可微分。

### 步骤三：可训练的 2D 溅射场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过正确的激活函数映射。这是每个 3DGS 实现的标准模式。

### 步骤四：将 2D 高斯拟合到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 步，64 个高斯会收敛到两个形状。这就是整个思路——对显式几何图元进行梯度下降。

### 步骤五：从 2D 到 3D

3D 扩展保持相同的循环。新增内容：

1. 每个高斯的旋转是四元数而非单个角度。
2. 协方差为 `R S S^T R^T`，`R` 由四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参和透视投影在 `mu` 处的雅可比矩阵。
4. 颜色变为球谐展开；在观察方向处评估。
5. 深度排序基于实际相机空间 z 而非可学习标量。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都在 GPU 上用基于图块的 CUDA 核实现这些。

### 步骤六：球谐函数评估

最高 3 阶的球谐基有每通道 16 项。评估：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

学到的 `sh_coeffs` 存储该高斯"在每个方向的颜色"。在渲染时，对当前观察方向评估，得到 3 维 RGB 向量。

## 生产使用

对于真实的 3DGS 工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。对于典型场景，在 RTX 4090 上运行需要 10-30 分钟。

2026 年重要的导出选项：

- `.ply` — 原始高斯云（可移植，文件最大）。
- `.splat` — PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting` — Khronos 标准，跨查看器可移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 原生，用于 NVIDIA Omniverse 和 Vision Pro 管线。

对于 4D/动态场景，`4DGS` 和 `Deformable-3DGS` 用随时间变化的均值和不透明度扩展了相同的机制。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 3DGS | "高斯溅射" | 将场景表示为数百万 3D 高斯的显式表示，每个高斯有位置、旋转、尺度、不透明度、球谐颜色 |
| 协方差（Covariance） | "高斯的形状" | `Sigma = R S S^T R^T`；一个高斯的朝向和各向异性尺度 |
| Alpha 合成（Alpha compositing） | "从后到前混合" | 与 NeRF 体积渲染相同的公式，现在作用于显式稀疏集合 |
| 致密化（Densification） | "克隆与分裂" | 在重建欠拟合处自适应添加新高斯 |
| 剪枝（Pruning） | "删除低不透明度" | 移除训练中不透明度趋近于零的高斯 |
| 球谐函数（Spherical harmonics） | "视角相关颜色" | 球面上的傅里叶基；将颜色存储为观察方向的函数 |
| Splatfacto | "nerfstudio 的 3DGS" | 2026 年训练 3DGS 的最简路径 |
| `KHR_gaussian_splatting` | "glTF 标准" | Khronos 2026 扩展，使 3DGS 在查看器和引擎间可移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产质量的 CUDA 光栅化器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练方案
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 年可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` 格式
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概览

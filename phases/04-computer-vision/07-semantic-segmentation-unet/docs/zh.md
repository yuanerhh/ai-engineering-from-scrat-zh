# 语义分割——U-Net

> 分割是在每个像素上进行的分类。U-Net 通过将下采样编码器与上采样解码器配对，并在两者之间接入跳跃连接，使其得以实现。

**类型：** 构建
**语言：** Python
**前置条件：** 第 4 阶段第 03 课（CNN）、第 4 阶段第 04 课（图像分类）
**时间：** ~75 分钟

## 学习目标

- 区分语义分割、实例分割和全景分割，并为给定问题选择正确的任务
- 在 PyTorch 中从零构建 U-Net，包含编码器块、瓶颈层、带转置卷积的解码器和跳跃连接
- 实现逐像素交叉熵、Dice 损失和目前医学及工业分割默认使用的组合损失
- 按类别读取 IoU 和 Dice 指标，诊断差分数是来自小目标召回、边界精度还是类别不平衡

## 问题所在

分类每张图像输出一个标签。检测每张图像输出少量框。分割每个像素输出一个标签。对于大小为 `H x W` 的输入，输出是形状为 `H x W`（语义）或 `H x W x N_instances`（实例）的张量。即每张图像数百万个预测，而非一个。

分割的结构解释了为什么它支撑着几乎每一个密集预测视觉产品：医学成像（肿瘤掩码）、自动驾驶（道路、车道、障碍物）、卫星图（建筑轮廓、农田边界）、文档解析（布局区域）、机器人（可抓取区域）。这些任务都无法通过在目标周围画框来解决；它们需要精确的轮廓。

架构问题很容易陈述，但不容易解决：你需要网络同时看到图像的全局上下文（这是什么场景）和局部像素细节（哪个像素是道路 vs 人行道）。标准 CNN 通过空间压缩来获得上下文，而这会丢弃细节。U-Net 是同时获得两者的设计。

## 核心概念

### 语义 vs 实例 vs 全景

```mermaid
flowchart LR
    IN["输入图像"] --> SEM["语义分割<br/>（像素 → 类别）"]
    IN --> INS["实例分割<br/>（像素 → 目标 ID，<br/>仅前景类别）"]
    IN --> PAN["全景分割<br/>（每个像素 → 类别 + ID）"]

    style SEM fill:#dbeafe,stroke:#2563eb
    style INS fill:#fef3c7,stroke:#d97706
    style PAN fill:#dcfce7,stroke:#16a34a
```

- **语义分割**说"这个像素是道路，那个像素是汽车。"相邻的两辆车合并成一个斑块。
- **实例分割**说"这个像素是汽车 #3，那个像素是汽车 #5。"忽略背景物体（"stuff" = 天空、道路、草地）。
- **全景分割**统一两者：每个像素获得一个类别标签，每个实例获得唯一 ID，stuff 和 things 都被分割。

本课涵盖语义分割。下一课（Mask R-CNN）涵盖实例分割。

### U-Net 形状

```mermaid
flowchart LR
    subgraph ENC["编码器（收缩）"]
        E1["64<br/>H x W"] --> E2["128<br/>H/2 x W/2"]
        E2 --> E3["256<br/>H/4 x W/4"]
        E3 --> E4["512<br/>H/8 x W/8"]
    end
    subgraph BOT["瓶颈"]
        B1["1024<br/>H/16 x W/16"]
    end
    subgraph DEC["解码器（扩张）"]
        D4["512<br/>H/8 x W/8"] --> D3["256<br/>H/4 x W/4"]
        D3 --> D2["128<br/>H/2 x W/2"]
        D2 --> D1["64<br/>H x W"]
    end
    E4 --> B1 --> D4
    E1 -. 跳跃连接 .-> D1
    E2 -. 跳跃连接 .-> D2
    E3 -. 跳跃连接 .-> D3
    E4 -. 跳跃连接 .-> D4
    D1 --> OUT["1x1 卷积<br/>类别"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style BOT fill:#fef3c7,stroke:#d97706
    style DEC fill:#dcfce7,stroke:#16a34a
```

编码器四次将空间分辨率减半并将通道数翻倍。解码器逆转：四次将空间分辨率翻倍并将通道数减半。跳跃连接在每个分辨率处将匹配的编码器特征与解码器特征拼接。最终 1x1 卷积在完整分辨率下将 `64 -> num_classes`。

为什么跳跃连接是必要的：当解码器尝试输出像素级预测时，它只看到了小的特征图。没有跳跃连接，它无法精确定位边缘，因为那些信息在编码器中被压缩掉了。跳跃连接将编码器在下采样过程中计算的高分辨率特征图传递给解码器。

### 转置卷积 vs 双线性上采样

解码器必须扩展空间维度。两个选项：

- **转置卷积**（`nn.ConvTranspose2d`）——可学习的上采样。历史上的 U-Net 默认值。如果步长和卷积核大小不能整除，可能产生棋盘格伪影。
- **双线性上采样 + 3x3 卷积**——平滑上采样后接卷积。伪影更少，参数更少，现在是现代默认值。

两者在实际中都有使用。对于第一个 U-Net，双线性更安全。

### 像素网格上的交叉熵

对于有 C 个类别的语义分割，模型输出为 `(N, C, H, W)`。目标为 `(N, H, W)`，包含整数类别 ID。交叉熵与分类情况完全相同，只是应用于每个空间位置：

```
损失 = 对 (n, h, w) 的平均，-log( softmax(logits[n, :, h, w])[target[n, h, w]] )
```

PyTorch 中的 `F.cross_entropy` 原生处理这种形状。不需要重塑。

### Dice 损失及其必要性

交叉熵对每个像素平等对待。当一个类别主导了帧（医学成像：99% 背景，1% 肿瘤），这是错误的。网络可以通过到处预测背景获得 99% 的准确率，但这毫无用处。

Dice 损失通过直接优化预测掩码与真实掩码之间的重叠来解决这个问题：

```
Dice(p, y) = 2 * sum(p * y) / (sum(p) + sum(y) + epsilon)
Dice_loss = 1 - Dice
```

其中 `p` 是一个类别的 sigmoid/softmax 概率图，`y` 是二进制真实掩码。只有当重叠完美时损失才为零。因为它基于比率，类别不平衡无关紧要。

实际中使用**组合损失**：

```
L = L_cross_entropy + lambda * L_dice       (lambda ~ 1)
```

交叉熵在训练初期提供稳定的梯度；Dice 在训练尾期专注于实际匹配掩码形状。这种组合是医学成像的默认值，在任何类别不平衡的数据集上都难以超越。

### 评估指标

- **像素准确率** — 正确预测的像素百分比。计算简单。在不平衡数据上与分类中的准确率一样失效。
- **逐类 IoU** — 每个类别掩码的交并比；对类别取平均 = mIoU。
- **Dice（像素上的 F1）** — 与 IoU 相似；`Dice = 2 * IoU / (1 + IoU)`。医学成像偏好 Dice，驾驶社区偏好 IoU；它们单调相关。
- **边界 F1** — 测量预测边界与真实边界的接近程度，惩罚即使是小的偏移。对于半导体检测等高精度任务很重要。

报告逐类 IoU，而非只有 mIoU。当九个类别在 85% 时，平均 IoU 会隐藏一个 15% 的类别。

### 输入分辨率权衡

U-Net 的编码器将分辨率减半四次，所以输入必须能被 16 整除。医学图像通常是 512x512 或 1024x1024。自动驾驶裁剪是 2048x1024。U-Net 的内存成本随 `H * W * C_max` 缩放，在 1024x1024 配 1024 个瓶颈通道时，前向传播已经使用了数 GB 的显存。

两种标准变通方案：
1. 将输入分块——带重叠地处理 256x256 的块并拼接。
2. 用空洞卷积替换瓶颈，保持更高的空间分辨率但扩大感受野（DeepLab 家族）。

对于第一个模型，256x256 输入配 64 通道基础 U-Net 在 8 GB 显存上可以舒适训练。

## 动手构建

### 步骤 1：编码器块

两个 3x3 卷积配批归一化和 ReLU。第一个卷积改变通道数；第二个保持不变。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

这个块全程复用。`bias=False` 因为 BN 的 beta 处理偏置。

### 步骤 2：下采样和上采样块

```python
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
```

仅空间形状检查（`shape[-2:]`）处理维度不能被 16 整除的输入；一个安全的 `F.interpolate` 在拼接前对齐张量。比较完整形状也会在通道数不同时触发，那应该是一个明确的错误，而非静默插值。

### 步骤 3：U-Net

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16 + base * 8, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u3 = Up(base * 4 + base * 2, base * 2)
        self.u4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)

net = UNet(in_channels=3, num_classes=2, base=32)
x = torch.randn(1, 3, 256, 256)
print(f"输出: {net(x).shape}")
print(f"参数: {sum(p.numel() for p in net.parameters()):,}")
```

输出形状 `(1, 2, 256, 256)`——与输入相同的空间大小，`num_classes` 个通道。`base=32` 时约 770 万参数。

### 步骤 4：损失函数

```python
def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, num_classes, lam=1.0):
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce.item(), "dice": dc.item()}
```

Dice 按类别计算再取平均（宏 Dice）。`eps` 防止对批次中不存在的类别除以零。

### 步骤 5：IoU 指标

```python
@torch.no_grad()
def iou_per_class(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious[c] = (inter / union) if union > 0 else torch.tensor(float("nan"))
    return ious
```

返回长度为 C 的向量。`nan` 标记批次中不存在的类别——计算 mIoU 时不对这些取平均。

### 步骤 6：端到端验证的合成数据集

在彩色背景上生成形状，使网络必须学习形状，而非像素颜色。

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader

def synthetic_segmentation(num_samples=200, size=64, seed=0):
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    for i in range(num_samples):
        bg = rng.uniform(0, 1, (3,))
        images[i] = bg
        masks[i] = 0
        num_shapes = rng.integers(1, 4)
        for _ in range(num_shapes):
            cls = int(rng.integers(1, 3))
            color = rng.uniform(0, 1, (3,))
            cx, cy = rng.integers(10, size - 10, size=2)
            r = int(rng.integers(4, 12))
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            if cls == 1:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            else:
                mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            images[i][mask] = color
            masks[i][mask] = cls
        images[i] += rng.normal(0, 0.02, images[i].shape)
        images[i] = np.clip(images[i], 0, 1)
    return images, masks


class SegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[i]).long()
        return img, mask
```

三个类别：背景（0）、圆形（1）、方形（2）。网络必须学会区分形状。

### 步骤 7：训练循环

```python
def train_one_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    loss_sum, total = 0.0, 0
    iou_sum = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, _ = combined_loss(logits, y, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        iou_sum += iou_per_class(logits, y, num_classes).nan_to_num(0)
    return loss_sum / total, iou_sum / len(loader)
```

在合成数据集上运行 10-30 个 epoch，观察形状类别的 mIoU 超过 0.9。注意 `nan_to_num(0)` 将批次中不存在的类别视为零；为了准确的逐类 IoU，在评估时按存在性掩码并使用 `torch.nanmean` 跨批次取平均，而非在这里取平均。

## 实际使用

对于生产环境，`segmentation_models_pytorch`（"smp"）将每个标准分割架构与任何 torchvision 或 timm 骨干封装在一起。三行代码：

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)
```

实际工作中值得了解的：
- **DeepLabV3+** 用空洞卷积替换基于最大池化的下采样，使瓶颈保持分辨率；在卫星和驾驶数据上边界更快。
- **SegFormer** 将卷积编码器换成分层 Transformer；在许多基准测试上是当前最先进水平。
- **Mask2Former** / **OneFormer** 在单个架构中统一语义、实例和全景分割。

所有三种都是 `smp` 或 `transformers` 中使用相同数据加载器的即插即用替换。

## 交付成果

本课产生：

- `outputs/prompt-segmentation-task-picker.md` — 一个提示词，在语义、实例和全景分割之间选择，并为给定任务命名架构。
- `outputs/skill-segmentation-mask-inspector.md` — 一个技能，报告类别分布、预测掩码统计数据，以及预测不足或边界模糊的类别。

## 练习

1. **（简单）** 为二元分割任务（前景 vs 背景）实现 `bce_dice_loss`。在合成双类数据集上验证，当前景占 5% 的像素时，组合损失比单独 BCE 收敛更快。
2. **（中等）** 用 `kernel_size=2, stride=2` 的 `nn.ConvTranspose2d` 上采样块替换 `nn.Upsample + conv` 块。在合成数据集上训练两者并比较 mIoU。观察转置卷积版本中棋盘格伪影出现的位置。
3. **（困难）** 取一个真实分割数据集（Oxford-IIIT Pets、Cityscapes mini 分割或医学子集），将 U-Net 训练到与 `smp.Unet` 参考的 2 个 IoU 点以内。报告逐类 IoU，识别哪些类别从添加 Dice 到损失中受益最多。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 语义分割（Semantic segmentation） | "标记每个像素" | 将像素分类为 C 个类别；同一类的实例合并 |
| 实例分割（Instance segmentation） | "标记每个目标" | 分离同一类的不同实例；仅前景 |
| 全景分割（Panoptic segmentation） | "语义 + 实例" | 每个像素获得一个类别；每个 thing 实例也获得唯一 ID |
| 跳跃连接（Skip connection） | "U-Net 桥" | 将编码器特征拼接到匹配分辨率的解码器特征中；保留高频细节 |
| 转置卷积（Transposed conv） | "反卷积" | 可学习的上采样；可能产生棋盘格伪影 |
| Dice 损失（Dice loss） | "重叠损失" | 1 - 2\|A ∩ B\| / (\|A\| + \|B\|)；直接优化掩码重叠，对类别不平衡鲁棒 |
| mIoU（平均交并比） | "平均交并比" | 对类别取平均的 IoU；分割的社区标准指标 |
| 边界 F1（Boundary F1） | "边界精度" | 仅在边界像素上计算的 F1 分数；对精度要求高的任务很重要 |

## 延伸阅读

- [U-Net：用于生物医学图像分割的卷积网络（Ronneberger 等，2015）](https://arxiv.org/abs/1505.04597) — 原始论文；每个人都复制的图在第 2 页
- [全卷积网络（Long 等，2015）](https://arxiv.org/abs/1411.4038) — 首次将分割变成端到端卷积问题的论文
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) — 生产分割的参考；每个标准架构加上每个标准损失
- [训练 SOTA 分割经验教训（kaggle.com 竞赛）](https://www.kaggle.com/code/iafoss/carvana-unet-pytorch) — 关于为什么 TTA、伪标签和类别权重在真实数据上重要的演练

# 图像分类

> 分类器是从像素到类别概率分布的函数。其他一切都是管道。

**类型：** 构建
**语言：** Python
**前置条件：** 第 2 阶段第 09 课（模型评估）、第 3 阶段第 10 课（迷你框架）、第 4 阶段第 03 课（CNN）
**时间：** ~75 分钟

## 学习目标

- 在 CIFAR-10 上构建端到端的图像分类管道：数据集、数据增强、模型、训练循环、评估
- 解释每个组件（DataLoader、损失函数、优化器、调度器、数据增强）的作用，并预测破坏其中任何一个在损失曲线上的表现
- 从零实现 mixup、cutout 和标签平滑，并说明何时值得添加它们
- 读取混淆矩阵（confusion matrix）和逐类精确率/召回率表，诊断超出聚合准确率的数据集和模型故障

## 问题所在

每一个落地的视觉任务在某种程度上都归结为图像分类。检测是对区域进行分类，分割是对像素进行分类，检索是按照与类别中心的相似度进行排名。把分类做对——数据集循环、数据增强策略、损失函数、评估——是贯穿本阶段所有其他任务的核心技能。

大多数分类 bug 不在模型里，而藏在管道中：错误的归一化、未打乱的训练集、破坏标签的数据增强、被训练数据污染的验证集分割、在第 30 个 epoch 后悄悄发散的学习率。一个在正确设置下能达到 93% CIFAR-10 准确率的 CNN，在有问题的设置下通常只有 70-75%，而损失曲线全程看起来都很合理。

本课手动接入整个管道，使每个部分都可检查。我们不使用 `torchvision.datasets` 中任何可能隐藏 bug 的内容。

## 核心概念

### 分类管道

```mermaid
flowchart LR
    A["数据集<br/>（图像 + 标签）"] --> B["数据增强<br/>（随机变换）"]
    B --> C["归一化<br/>（均值/标准差）"]
    C --> D["DataLoader<br/>（批次 + 打乱）"]
    D --> E["模型<br/>（CNN）"]
    E --> F["Logits<br/>（N, C）"]
    F --> G["交叉熵损失"]
    F --> H["Argmax<br/>（评估时）"]
    G --> I["反向传播"]
    I --> J["优化器步骤"]
    J --> K["调度器步骤"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个循环的每一行都可能藏有 bug。交叉熵接收原始 logits，而非 softmax 输出，因此在损失前的任何 `model(x).softmax()` 都会悄悄计算错误的梯度。数据增强只作用于输入，不作用于标签——除了 mixup，它对两者都做混合。`optimizer.zero_grad()` 每步必须调用一次；跳过它会累积梯度，看起来像极不稳定的学习率。这些 bug 的每一个都会使学习曲线变平，而不抛出任何错误。

### 交叉熵、logits 与 softmax

分类器为每张图像产生 `C` 个数字，称为 logits（对数几率）。应用 softmax 将它们转换为概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

交叉熵衡量正确类别的负对数概率：

```
CE(z, y) = -log( softmax(z)_y )
         = -z_y + log( sum_j exp(z_j) )
```

右侧形式是数值稳定的形式（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 将 softmax + NLL 融合为一个操作，直接接收原始 logits。先自己应用 softmax 几乎总是一个 bug——你计算的是 log(softmax(softmax(z)))，一个无意义的量。

### 为什么数据增强有效

CNN 对平移有归纳偏置（来自权重共享），但对裁剪、翻转、颜色抖动或遮挡没有内置不变性。教会它这些不变性的唯一方式是展示能体现它们的像素。训练时的每一次随机变换都是在说："这两张图像有相同的标签；学习忽略差异的特征。"

```
原始裁剪：  "向左的狗"
翻转：      "向右的狗"       <- 相同标签，不同像素
旋转(+15)： "轻微倾斜的狗"
颜色抖动：  "在较暖光线下的狗"
随机擦除：  "缺失一块的狗"
```

规则：数据增强必须保留标签。对数字进行 cutout 和旋转可能会将 "6" 变成 "9"；对于那个数据集，要使用较小的旋转范围，并选择尊重数字特定不变性的数据增强。

### Mixup 与 CutMix

普通数据增强变换像素但保留 one-hot 标签。**Mixup** 和 **CutMix** 通过对两者进行插值来打破这一点。

```
Mixup：
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

CutMix：
  将 x_j 的一个随机矩形粘贴到 x_i 中
  y = 按面积加权混合 y_i 和 y_j
```

为什么有帮助：模型不再记忆尖锐的 one-hot 目标，而是学习在类别之间插值。训练损失上升，测试准确率上升。这是任何分类器最便宜的鲁棒性升级。

### 标签平滑（label smoothing）

mixup 的近亲。不是训练 `[0, 0, 1, 0, 0]`，而是训练 `[eps/C, eps/C, 1-eps, eps/C, eps/C]`（eps 取 0.1 这样的小值）。阻止模型产生任意尖锐的 logits 并以几乎零代价改善校准（calibration）。自 PyTorch 1.10 起内置于 `nn.CrossEntropyLoss(label_smoothing=0.1)`。

### 超越准确率的评估

聚合准确率隐藏了不平衡。一个 90-10 二分类器如果总是预测多数类，得分为 90%。真正告诉你发生了什么的工具：

- **逐类准确率** — 每类一个数字；立即显示表现不佳的类别。
- **混淆矩阵** — C x C 网格，第 i 行第 j 列 = 真实类别 i 被预测为类别 j 的数量；对角线是正确的，非对角线是模型真实所处之地。
- **Top-1 / Top-5** — 正确类别是否在前 1 或前 5 个预测中；Top-5 对 ImageNet 很重要，因为"诺里奇梗"vs"诺福克梗"这样的类别是真正模糊的。
- **校准（ECE）** — 0.8 置信度的预测有 80% 的时间正确吗？现代网络系统性地过于自信；用温度缩放（temperature scaling）或标签平滑修复。

## 动手构建

### 步骤 1：确定性合成数据集

CIFAR-10 存在于磁盘上。为使本课可复现且快速，我们构建一个看起来像 CIFAR 的合成数据集——带有类特定结构的 32x32 RGB 图像，模型必须学习这些结构。完全相同的管道可以无改动地用于真实 CIFAR-10。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类有自己的颜色调色板和频率模式，加上高斯噪声迫使模型学习信号而非记忆像素。十个类，每类一千张图像，随机打乱。

### 步骤 2：归一化与数据增强

每个视觉管道都有的两种变换。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

裁剪前使用反射填充（reflect-pad）而非零填充，因为黑色边框是一个模型会学着忽略的信号，但这种忽略是无用的。

### 步骤 3：Mixup

在训练步骤内混合两张图像和两个标签。作为批次变换实现，使其位于前向传播旁边而非数据集内部。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是对软标签分布的交叉熵。当目标恰好是 one-hot 时，它退化为通常的情况。

### 步骤 4：训练循环

完整配方：对数据进行一次遍历，每批次一次梯度，每 epoch 步进一次调度器。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # 开启 mixup 时，训练准确率是对未混合标签 y 的近似
        # （模型看到的是软目标，不是 y）。把它当作粗略的进度信号；
        # 真实性能依赖验证准确率。
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次编写训练循环时检查的五个不变量：

1. 训练前 `model.train()`，评估前 `model.eval()`——切换 dropout 和 batchnorm 的行为。
2. `.backward()` 前调用 `.zero_grad()`。
3. 累积指标时调用 `.item()`，不让任何内容保持计算图存活。
4. 评估期间使用 `@torch.no_grad()`——节省内存和时间，防止细微错误。
5. 对原始 logits 取 argmax，而非 softmax——结果相同，少一步操作。

### 步骤 5：组合起来

使用上一课的 `TinyResNet`，训练几个 epoch，进行评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet 来自上一课（03-cnns-lenet-to-resnet）。
# 将导入路径调整到你存储上一课代码的位置。
from cnns_lenet_to_resnet import TinyResNet  # 示例占位符

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，五个 epoch 内即可达到近乎完美的验证准确率，这正是重点：管道是正确的，模型能学到可学习的内容。将数据集换成真实的 CIFAR-10，同一个循环无需改动即可训练到约 90%。

### 步骤 6：读取混淆矩阵

仅靠准确率永远无法告诉你模型在哪里失败。混淆矩阵能做到。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  精确率 {prec[i]:.3f}  召回率 {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类别，列是预测。类别 3 和类别 5 之间一簇非对角线计数意味着模型在混淆这两类，给你提供了针对性数据收集或类特定数据增强的起点。

## 实际使用

`torchvision` 将上述所有内容封装为惯用组件。对于真实 CIFAR-10，完整管道是四行代码加一个训练循环。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

需要注意两点：均值/标准差是**数据集特定的**——在 CIFAR-10 训练集上计算，而非 ImageNet——反射填充是社区默认的裁剪策略。在这里复制粘贴 ImageNet 统计数据是约 1% 的准确率漏损，直到有人对模型进行分析才会发现。

## 交付成果

本课产生：

- `outputs/prompt-classifier-pipeline-auditor.md` — 一个提示词，审计训练脚本是否符合上述五个不变量，并发现第一个违规。
- `outputs/skill-classification-diagnostics.md` — 一个技能，给定混淆矩阵和类别名称列表，总结逐类故障并提出单一最有影响力的修复建议。

## 练习

1. **（简单）** 在合成数据集上，分别使用和不使用 mixup 训练同一模型五个 epoch。绘制两者的训练和验证损失图。解释为什么使用 mixup 时训练损失更高，但验证准确率相近甚至更好。
2. **（中等）** 实现 Cutout——在每张训练图像中随机擦除一个 8x8 的正方形——并进行消融实验：无数据增强、hflip+crop、hflip+crop+cutout、hflip+crop+mixup。报告每种情况的验证准确率。
3. **（困难）** 构建 CIFAR-100 管道（100 个类，相同输入尺寸），并将 ResNet-34 的训练运行结果复现到发布准确率的 1% 以内。额外：扫描三个学习率和两个权重衰减，记录到本地 CSV，生成最终的混淆矩阵-顶部混淆表。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Logits（对数几率） | "原始输出" | softmax 前每张图像 C 个数字的向量；交叉熵期望这些，而非 softmax 后的值 |
| 交叉熵（Cross-entropy） | "损失函数" | 正确类别的负对数概率；在一个稳定的操作中融合 log-softmax 和 NLL |
| DataLoader | "批次处理器" | 用打乱、批处理和（可选的）多工作者加载包装数据集；被归咎于一半的训练 bug |
| 数据增强（Augmentation） | "随机变换" | 训练时保留标签的任何像素级变换；教会 CNN 天生没有的不变性 |
| Mixup / CutMix | "混合两张图像" | 混合输入和标签，使分类器学习平滑插值而非硬边界 |
| 标签平滑（Label smoothing） | "更软的目标" | 用 (1-eps, eps/(C-1), ...) 替换 one-hot；改善校准并略微提升准确率 |
| Top-k 准确率 | "Top-5" | 正确类别在 k 个最高概率预测中；用于类别真正模糊的数据集 |
| 混淆矩阵（Confusion matrix） | "错误所在之处" | C x C 表格，条目 (i, j) 计算真实类别 i 被预测为 j 的图像数量；对角线是正确的，非对角线告诉你需要修复什么 |

## 延伸阅读

- [CS231n: 训练神经网络](https://cs231n.github.io/neural-networks-3/) — 仍是对训练管道最清晰的单页介绍
- [图像分类的技巧袋（He 等，2019）](https://arxiv.org/abs/1812.01187) — 每个小技巧加在一起使 ResNet 在 ImageNet 上提升 3-4%
- [mixup: 超越经验风险最小化（Zhang 等，2017）](https://arxiv.org/abs/1710.09412) — 原始 mixup 论文；三页理论加上令人信服的实验
- [为什么温度缩放重要（Guo 等，2017）](https://arxiv.org/abs/1706.04599) — 证明现代网络过于自信并用一个标量参数修复它的论文

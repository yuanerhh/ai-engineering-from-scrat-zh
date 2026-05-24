# PyTorch 入门

> 你从活塞和曲轴开始构建了引擎。现在来学习人人实际驾驶的那一辆。

**类型：** 构建
**语言：** Python
**前置知识：** 课程 03.10（构建你自己的迷你框架）
**时间：** ~75 分钟

## 学习目标

- 使用 PyTorch 的 nn.Module、nn.Sequential 和 autograd 构建并训练神经网络
- 使用 PyTorch 张量、GPU 加速和标准训练循环（zero_grad、forward、loss、backward、step）
- 将你的从零开始的迷你框架组件转换为 PyTorch 等价物
- 对同一任务上的纯 Python 框架和 PyTorch 进行训练速度分析和比较

## 问题所在

你有一个可运行的迷你框架：线性层、ReLU、dropout、批归一化、Adam、DataLoader 和训练循环。它在纯 Python 中训练一个 4 层网络解决圆形分类问题。

但它也比 PyTorch 在同样问题上慢 500 倍。

你的迷你框架使用嵌套 Python 循环每次处理一个样本。PyTorch 将同样的操作分发到在 GPU 上运行的优化 C++/CUDA 内核。在一块 NVIDIA A100 上，PyTorch 在 ImageNet（128 万张图片）上训练 ResNet-50（2560 万参数）大约需要 6 小时。你的框架在同样任务上大约需要 3,000 小时——如果内存够用的话。

速度不是唯一的差距。你的框架没有 GPU 支持，没有自动微分（你为每个模块手写了 backward()），没有序列化，没有分布式训练，没有混合精度，没有不靠 print 调试梯度流的方法。

PyTorch 填补了所有这些缺口。而且它使用的是与你从零构建的完全相同的概念模型：Module、forward()、parameters()、backward()、optimizer.step()。概念一一对应，语法几乎相同。区别在于 PyTorch 在你从头设计的同一接口后面包裹了十年的系统工程。

## 概念

### 为什么 PyTorch 赢了

2015 年，TensorFlow 要求你在运行任何东西之前先定义一个静态计算图。你构建图，编译它，然后向其中馈送数据。调试意味着盯着图的可视化。改变架构意味着从头重建图。

PyTorch 在 2017 年以不同的哲学发布：即时执行（eager execution）。你写 Python，它立即运行。`y = model(x)` 实际上现在就计算 y，而不是"向图中添加一个稍后计算 y 的节点"。这意味着标准 Python 调试工具可以工作——print() 可用，pdb 可用，forward pass 中的 if/else 可用。

到 2020 年，市场已经发声。PyTorch 在机器学习研究论文中的份额从 7%（2017 年）增长到超过 75%（2022 年）。Meta、Google DeepMind、OpenAI、Anthropic 和 Hugging Face 都使用 PyTorch 作为主要框架。TensorFlow 2.x 作为回应采用了即时执行——默认承认 PyTorch 的设计是正确的。

教训：开发体验会复利累积。一个慢 10% 但调试快 50% 的框架每次都会赢。

### 张量（Tensors）

张量是具有三个关键属性的多维数组：形状（shape）、数据类型（dtype）和设备（device）。

```python
import torch

x = torch.zeros(3, 4)           # 形状: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # 2 张 224x224 的 RGB 图像批次
x = torch.tensor([1, 2, 3])     # 从 Python 列表创建
```

**形状（Shape）** 是维度。标量是 shape ()，向量是 (n,)，矩阵是 (m, n)，图像批次是 (batch, channels, height, width)。

**数据类型（Dtype）** 控制精度和内存。

| dtype | 位数 | 范围 | 使用场景 |
|-------|------|-------|---------|
| float32 | 32 | ~7 位十进制数字 | 默认训练 |
| float16 | 16 | ~3.3 位十进制数字 | 混合精度 |
| bfloat16 | 16 | 与 float32 相同范围，精度较低 | LLM 训练 |
| int8 | 8 | -128 到 127 | 量化推理 |

**设备（Device）** 决定计算在哪里发生。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

每次操作都要求所有张量在同一设备上。这是初学者遇到的 #1 PyTorch 错误：`RuntimeError: Expected all tensors to be on the same device`。修复：在计算前将所有内容移到同一设备。

**重塑（Reshaping）** 是常数时间操作——它改变元数据，不改变数据。

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # 重塑为 (2, 12)——必须是连续的
x.reshape(6, 4)    # 重塑为 (6, 4)——总是有效
x.permute(2, 0, 1) # 重新排序维度
x.unsqueeze(0)     # 添加维度: (1, 2, 3, 4)
x.squeeze()        # 移除大小为 1 的维度
```

### 自动微分（Autograd）

你的迷你框架需要你为每个模块实现 backward()。PyTorch 不需要。它将对张量的每次操作记录到有向无环图（计算图）中，然后反向遍历该图自动计算梯度。

```mermaid
graph LR
    x["x（叶子节点）"] --> mul["*"]
    w["w（叶子节点，requires_grad）"] --> mul
    mul --> add["+"]
    b["b（叶子节点，requires_grad）"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w
```

与你的框架的关键区别：PyTorch 使用基于磁带的自动微分（tape-based autodiff）。每次操作在前向传播期间附加到"磁带"。调用 `.backward()` 反向重放磁带。

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

自动微分的三条规则：

1. 只有 `requires_grad=True` 的叶子张量才会积累梯度
2. 梯度默认积累——每次反向传播前调用 `optimizer.zero_grad()`
3. `torch.no_grad()` 禁用梯度追踪（评估时使用）

### nn.Module

`nn.Module` 是 PyTorch 中每个神经网络组件的基类。你在课程 10 中已经构建了这个抽象。PyTorch 的版本增加了自动参数注册、递归模块发现、设备管理和状态字典序列化。

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

当你在 `__init__` 中将 `nn.Module` 或 `nn.Parameter` 赋值为属性时，PyTorch 会自动注册它。`model.parameters()` 递归收集每个已注册的参数。这就是为什么你不需要像在迷你框架中那样手动收集权重。

关键构建块：

| 模块 | 功能 | 参数数量 |
|------|------|---------|
| nn.Linear(in, out) | Wx + b | in*out + out |
| nn.Conv2d(in_ch, out_ch, k) | 2D 卷积 | in_ch*out_ch*k*k + out_ch |
| nn.BatchNorm1d(features) | 归一化激活值 | 2 * features |
| nn.Dropout(p) | 随机置零 | 0 |
| nn.ReLU() | max(0, x) | 0 |
| nn.GELU() | 高斯误差线性单元 | 0 |
| nn.Embedding(vocab, dim) | 查找表 | vocab * dim |
| nn.LayerNorm(dim) | 每样本归一化 | 2 * dim |

### 损失函数和优化器

PyTorch 内置了你构建的所有内容的生产就绪版本。

**损失函数**（来自 `torch.nn`）：

| 损失 | 任务 | 输入 |
|------|------|------|
| nn.MSELoss() | 回归 | 任意形状 |
| nn.CrossEntropyLoss() | 多分类 | Logits（非 softmax） |
| nn.BCEWithLogitsLoss() | 二元分类 | Logits（非 sigmoid） |
| nn.L1Loss() | 鲁棒回归 | 任意形状 |
| nn.CTCLoss() | 序列对齐 | 对数概率 |

注意：`CrossEntropyLoss` 内部结合了 `LogSoftmax` + `NLLLoss`。传入原始 logits，不要传 softmax 输出。这是一个常见错误，会悄无声息地产生错误梯度。

**优化器**（来自 `torch.optim`）：

| 优化器 | 使用场景 | 典型学习率 |
|--------|---------|-----------|
| SGD(params, lr, momentum) | CNN、精心调优的流程 | 0.01--0.1 |
| Adam(params, lr) | 默认起点 | 1e-3 |
| AdamW(params, lr, weight_decay) | Transformer、微调 | 1e-4--1e-3 |
| LBFGS(params) | 小规模、二阶方法 | 1.0 |

### 训练循环

每个 PyTorch 训练循环都遵循相同的 5 步模式。你在课程 10 中已经知道这些了。

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop 每轮（Each Epoch）
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end
```

标准模式：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

批次循环内的五行代码。训练了 GPT-4、Stable Diffusion 和 LLaMA 的五行代码。架构会变，数据会变，这五行不变。

### Dataset 和 DataLoader

PyTorch 的 `Dataset` 是一个有两个方法的抽象类：`__len__` 和 `__getitem__`。`DataLoader` 将其包装为批处理、打乱和多进程数据加载。

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

`num_workers=4` 启动 4 个进程，在 GPU 训练当前批次的同时并行加载数据。在磁盘受限的工作负载（大型图像、音频）中，仅这一点就可以将训练速度提升一倍。

### GPU 训练

将模型移至 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

这会递归地将每个参数和缓冲区移至 GPU。然后在训练期间移动每个批次：

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**混合精度（Mixed precision）** 通过在 float16 中运行前向/反向传播（同时在 float32 中保留主权重）在现代 GPU（A100、H100、RTX 4090）上将内存减半、吞吐量翻倍：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 对比：迷你框架 vs PyTorch vs JAX

| 特性 | 迷你框架（课程 10） | PyTorch | JAX |
|------|-------------------|---------|-----|
| 自动微分 | 手动 backward() | 基于磁带的 autograd | 函数式变换 |
| 执行方式 | 即时（Python 循环） | 即时（C++ 内核） | 追踪 + JIT 编译 |
| GPU 支持 | 无 | 有（CUDA, ROCm, MPS） | 有（CUDA, TPU） |
| 速度（MNIST MLP） | ~300s/轮 | ~0.5s/轮 | ~0.3s/轮 |
| 模块系统 | 自定义 Module 类 | nn.Module | 无状态函数（Flax/Equinox） |
| 调试 | print() | print(), pdb, breakpoint() | 困难（JIT 追踪破坏 print） |
| 生态系统 | 无 | Hugging Face, Lightning, timm | Flax, Optax, Orbax |
| 学习曲线 | 你自己构建了 | 适中 | 陡峭（函数式范式） |
| 生产使用 | 玩具问题 | Meta, OpenAI, Anthropic, HF | Google DeepMind, Midjourney |

## 构建

使用纯 PyTorch 原语在 MNIST 上训练一个 3 层 MLP。不使用高级封装，不用 `torchvision.datasets`。我们自己下载并解析原始数据。

### 步骤 1：从原始文件加载 MNIST

MNIST 以 4 个 gzip 压缩文件形式分发：训练图像（60,000 x 28 x 28）、训练标签、测试图像（10,000 x 28 x 28）和测试标签。我们下载它们并解析二进制格式。

```python
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
```

### 步骤 2：定义模型

一个 3 层 MLP：784 -> 256 -> 128 -> 10。ReLU 激活。Dropout 用于正则化。不用批归一化以保持简单。

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
```

输出层产生 10 个原始 logits（每个数字一个）。不用 softmax——`CrossEntropyLoss` 内部处理。

参数数量：784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146。按现代标准很小。GPT-2 small 有 1.24 亿。这几秒内就能训练完。

### 步骤 3：训练循环

标准的 forward-loss-backward-step 模式。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
```

注意评估时的 `torch.no_grad()`。这禁用了 autograd，减少了内存使用并加速推理。没有它，PyTorch 会构建一个你永远不会使用的计算图。

### 步骤 4：连接所有内容

```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
```

10 轮后的预期输出：~97.8% 测试准确率。CPU 训练时间：~30 秒。GPU：~5 秒。同架构的迷你框架：~45 分钟。

## 实际应用

### 快速对比：迷你框架 vs PyTorch

| 迷你框架（课程 10） | PyTorch |
|------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` 然后 `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| 无 GPU | `model.to("cuda")` |
| 每个模块手动实现 backward | Autograd 自动处理一切 |

接口几乎完全相同。区别在于底层的所有内容。

### 保存和加载模型

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

始终保存 `state_dict()`（参数字典），不要保存模型对象。保存模型对象使用 pickle，重构代码时会失效。状态字典是可移植的。

### 学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

PyTorch 内置了 15+ 种调度器：StepLR、ExponentialLR、CosineAnnealingLR、OneCycleLR、ReduceLROnPlateau。全部接入同一个优化器接口。

## 交付物

本课程产出两个文件：

- `outputs/prompt-pytorch-debugger.md`——诊断常见 PyTorch 训练失败的提示词
- `outputs/skill-pytorch-patterns.md`——PyTorch 训练模式的技能参考

## 练习

1. **添加批归一化。** 在每个线性层之后（激活函数之前）插入 `nn.BatchNorm1d`。与仅用 dropout 的版本比较测试准确率和训练速度。批归一化应该在更少的轮次内达到 98%+。

2. **实现学习率查找器。** 以指数增加的学习率（从 1e-7 到 1.0）训练一个 epoch。绘制损失 vs LR 图。最优 LR 就在损失开始上升之前。用它为 MNIST 模型选择更好的 LR。

3. **使用混合精度移植到 GPU。** 为训练循环添加 `torch.amp.autocast` 和 `GradScaler`。测量有无混合精度的吞吐量（每秒样本数）。在 A100 上预期 ~2x 加速。

4. **构建自定义 Dataset。** 下载 Fashion-MNIST（与 MNIST 格式相同，但是服装类别）。实现带 `__getitem__` 和 `__len__` 的 `FashionMNISTDataset(Dataset)` 类。训练同样的 MLP 并比较准确率。Fashion-MNIST 更难——预期 ~88% vs ~98%。

5. **用 SGD + 动量替换 Adam。** 用 `SGD(params, lr=0.01, momentum=0.9)` 训练。比较收敛曲线。然后添加 `CosineAnnealingLR` 调度器，看看 SGD 到第 10 轮是否能追上 Adam。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|---------|
| 张量（Tensor） | "多维数组" | 每次操作都内置了自动微分支持的类型感知、设备感知数组 |
| 自动微分（Autograd） | "自动反向传播" | 基于磁带的系统，在前向传播期间记录操作，然后反向重放以计算精确梯度 |
| nn.Module | "一个层" | 任何可微分计算块的基类——注册参数，支持嵌套，处理 train/eval 模式 |
| state_dict | "模型权重" | 将参数名称映射到张量的 OrderedDict——训练模型的可移植、可序列化表示 |
| .backward() | "计算梯度" | 反向遍历计算图，为每个有 requires_grad=True 的叶子张量计算并积累梯度 |
| .to(device) | "移至 GPU" | 递归地将所有参数和缓冲区转移到指定设备（CPU、CUDA、MPS） |
| DataLoader | "数据流水线" | 从 Dataset 批处理、打乱并可选地并行化数据加载的迭代器 |
| 混合精度（Mixed precision） | "用 float16" | 用 float16 前向/反向传播提速，同时保留 float32 主权重以保持数值稳定性 |
| 即时执行（Eager execution） | "立即运行" | 操作在调用时立即执行，不推迟到后来的编译步骤——这是区分 PyTorch 与 TF 1.x 的核心设计选择 |
| zero_grad | "重置梯度" | 在下一次反向传播之前将所有参数梯度设为零，因为 PyTorch 默认积累梯度 |

## 延伸阅读

- Paszke et al.，"PyTorch: An Imperative Style, High-Performance Deep Learning Library"（2019）——解释 PyTorch 设计权衡的原始论文
- PyTorch 教程："Learning PyTorch with Examples"——从张量到 nn.Module 的官方路径
- PyTorch 性能调优指南——混合精度、DataLoader workers、固定内存和其他生产优化
- Horace He，"Making Deep Learning Go Brrrr"——为什么 GPU 训练快，以及 PyTorch 特定的优化策略

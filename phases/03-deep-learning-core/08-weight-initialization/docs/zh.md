# 权重初始化与训练稳定性

> 初始化错误，训练就无从开始。初始化正确，50 层网络就能像 3 层一样平滑训练。

**类型：** 构建
**语言：** Python
**前置要求：** 第03.04课（激活函数），第03.07课（正则化）
**时长：** ~90 分钟

## 学习目标

- 实现零初始化、随机初始化、Xavier/Glorot 和 Kaiming/He 初始化策略，并测量它们对 50 层激活量级的影响
- 推导为什么 Xavier 使用 Var(w) = 2/(fan_in + fan_out)，Kaiming 使用 Var(w) = 2/fan_in
- 演示零初始化的对称性问题，解释为什么仅随机缩放是不够的
- 将正确的初始化策略与激活函数匹配：sigmoid/tanh 用 Xavier，ReLU/GELU 用 Kaiming

## 问题背景

将所有权重初始化为零。什么都不学习。每个神经元计算相同的函数，接收相同的梯度，进行相同的更新。经过 10000 个 epoch，你的 512 神经元隐藏层仍然是同一个神经元的 512 份拷贝。你付了 512 个参数的代价，却只得到 1 个。

初始化太大。激活值在网络中爆炸，第 10 层时值达到 1e15，第 20 层时溢出到无穷大。梯度在反方向走同样的路径。

从标准正态分布随机初始化。3 层时有效。50 层时，信号坍缩到零或爆炸到无穷大，取决于随机缩放是略小还是略大。"有效"和"损坏"之间的界线极其细微。

权重初始化是深度学习中最被低估的决策。架构发论文，优化器上博客，初始化只是脚注。但搞错了，其他一切都无关紧要——你的网络在训练开始前就已经死了。

## 核心概念

### 对称性问题

层中的每个神经元都有相同的结构：将输入乘以权重，加偏置，应用激活。如果所有权重从相同的值开始（零是极端情况），每个神经元计算相同的输出。反向传播时，每个神经元接收相同的梯度。更新步骤时，每个神经元改变相同的量。

你被卡住了。网络有数百个参数，但它们都同步移动。这叫做对称性，随机初始化是打破它的暴力方法。每个神经元在权重空间的不同点开始，因此各自学习不同的特征。

但"随机"是不够的。随机性的*尺度*决定了网络是否能训练。

### 方差通过各层的传播

考虑一个有 fan_in 个输入的单层：

```
z = w1*x1 + w2*x2 + ... + w_n*x_n
```

如果每个权重 wi 从方差为 Var(w) 的分布中抽取，每个输入 xi 的方差为 Var(x)，输出方差为：

```
Var(z) = fan_in * Var(w) * Var(x)
```

如果 Var(w) = 1，fan_in = 512，输出方差是输入方差的 512 倍。经过 10 层：512^10 = 1.2e27。你的信号爆炸了。

如果 Var(w) = 0.001，输出方差每层缩减 0.001 * 512 = 0.512 倍。经过 10 层：0.512^10 = 0.00013。你的信号消失了。

目标：选择 Var(w) 使得 Var(z) = Var(x)。信号量级跨层保持恒定。

### Xavier/Glorot 初始化

Glorot 和 Bengio（2010）推导了 sigmoid 和 tanh 激活的解。为了在前向和反向传播中保持方差恒定：

```
Var(w) = 2 / (fan_in + fan_out)
```

实践中，权重从以下分布中抽取：

```
w ~ Uniform(-limit, limit)  其中 limit = sqrt(6 / (fan_in + fan_out))
```

或：

```
w ~ Normal(0, sqrt(2 / (fan_in + fan_out)))
```

这有效是因为 sigmoid 和 tanh 在零附近近似线性，而正确初始化的激活值就在零附近。方差在数十层中保持稳定。

### Kaiming/He 初始化

ReLU 杀死一半的输出（所有负值变为零）。有效的 fan_in 减半，因为平均有一半的输入被置零。Xavier 初始化没有考虑到这一点——它低估了所需的方差。

He 等人（2015）调整了公式：

```
Var(w) = 2 / fan_in
```

权重从以下分布中抽取：

```
w ~ Normal(0, sqrt(2 / fan_in))
```

系数 2 补偿了 ReLU 将一半激活置零的效果。没有它，信号每层收缩约 0.5 倍。经过 50 层：0.5^50 = 8.8e-16。Kaiming 初始化防止了这种情况。

### Transformer 初始化

GPT-2 引入了不同的模式。残差连接将每个子层的输出加到其输入上：

```
x = x + sublayer(x)
```

每次加法都会增加方差。有 N 个残差层时，方差按 N 成比例增长。GPT-2 将残差层的权重缩放 1/sqrt(2N)，其中 N 是层数。这保持了累积信号量级的稳定。

Llama 3（4050 亿参数，126 层）使用类似的方案。没有这种缩放，残差流会通过 126 层注意力和前馈块无限增长。

```mermaid
flowchart TD
    subgraph "零初始化"
        Z1["第1层<br/>所有权重 = 0"] --> Z2["第2层<br/>所有神经元相同"]
        Z2 --> Z3["第3层<br/>仍然相同"]
        Z3 --> ZR["结果: 1 个有效神经元<br/>无论宽度多少"]
    end

    subgraph "Xavier 初始化"
        X1["第1层<br/>Var = 2/(fan_in+fan_out)"] --> X2["第2层<br/>信号稳定"]
        X2 --> X3["第50层<br/>信号稳定"]
        X3 --> XR["结果: 用 sigmoid/tanh 训练"]
    end

    subgraph "Kaiming 初始化"
        K1["第1层<br/>Var = 2/fan_in"] --> K2["第2层<br/>信号稳定"]
        K2 --> K3["第50层<br/>信号稳定"]
        K3 --> KR["结果: 用 ReLU/GELU 训练"]
    end
```

### 50 层中的激活量级

```mermaid
graph LR
    subgraph "平均激活量级"
        direction LR
        L1["第1层"] --> L10["第10层"] --> L25["第25层"] --> L50["第50层"]
    end

    subgraph "结果"
        R1["随机 N(0,1): 第5层爆炸"]
        R2["随机 N(0,0.01): 第10层消失"]
        R3["Xavier + Sigmoid: 第50层 ~1.0"]
        R4["Kaiming + ReLU: 第50层 ~1.0"]
    end
```

### 选择正确的初始化

```mermaid
flowchart TD
    Start["使用什么激活函数？"] --> Act{"激活函数类型？"}

    Act -->|"Sigmoid / Tanh"| Xavier["Xavier/Glorot<br/>Var = 2/(fan_in + fan_out)"]
    Act -->|"ReLU / Leaky ReLU"| Kaiming["Kaiming/He<br/>Var = 2/fan_in"]
    Act -->|"GELU / Swish"| Kaiming2["Kaiming/He<br/>（与 ReLU 相同）"]
    Act -->|"Transformer 残差"| GPT["按 1/sqrt(2N) 缩放<br/>N = 层数"]

    Xavier --> Check["验证：激活量级<br/>在所有层中保持在 0.5 到 2.0 之间"]
    Kaiming --> Check
    Kaiming2 --> Check
    GPT --> Check
```

## 实现

### 第一步：初始化策略

四种初始化权重矩阵的方式。每种返回一个二维列表（矩阵），fan_in 列，fan_out 行。

```python
import math
import random


def zero_init(fan_in, fan_out):
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
```

### 第二步：激活函数

我们需要 sigmoid、tanh 和 ReLU 来测试每种初始化策略与其目标激活函数的匹配。

```python
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)
```

### 第三步：通过 50 层的前向传播

将随机数据通过深层网络前向传播，测量每层的平均激活量级。

```python
def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    random.seed(42)
    layer_magnitudes = []

    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes
```

### 第四步：实验

运行所有组合：零初始化、随机 N(0,1)、随机 N(0,0.01)、带 sigmoid 的 Xavier、带 tanh 的 Xavier、带 ReLU 的 Kaiming。打印关键层的量级。

```python
def run_experiment():
    configs = [
        ("零初始化 + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("随机 N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("随机 N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"{'策略':<30} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("-" * 80)

    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        row = f"{name:<30}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'EXPLODED':>10}"
            elif val < 1e-6:
                row += f" {'VANISHED':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)
```

### 第五步：对称性演示

展示零初始化产生相同的神经元。

```python
def symmetry_demo():
    random.seed(42)
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("\n对称性演示（4 个神经元，零初始化）:")
    for i, out in enumerate(outputs):
        print(f"  神经元 {i}: 输出 = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"  全部相同: {all_same}")
    print(f"  有效参数: 1（非 {len(weights) * len(weights[0])}）")
```

### 第六步：逐层量级报告

打印 50 层中激活量级的可视化条形图。

```python
def magnitude_report(name, magnitudes):
    print(f"\n{name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " EXPLODED"
            elif mag < 1e-6:
                bar = "." + " VANISHED"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"  第{i+1:3d}层: {bar} ({mag:.6f})")
```

## 工程应用

PyTorch 提供这些作为内置函数：

```python
import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

nn.init.zeros_(layer.bias)
```

当你调用 `nn.Linear(512, 256)` 时，PyTorch 默认使用 Kaiming 均匀初始化。这就是为什么大多数简单网络"就是能用"——PyTorch 已经做出了正确的选择。但当你构建自定义架构或深度超过 20 层时，你需要理解正在发生什么，并可能需要覆盖默认值。

对于 transformer，HuggingFace 模型通常在其 `_init_weights` 方法中处理初始化。GPT-2 的实现将残差投影按 1/sqrt(N) 缩放。如果你从头构建 transformer，你需要自己添加这部分。

## 交付物

本课产出：
- `outputs/prompt-init-strategy.md` — 诊断权重初始化问题并推荐正确策略的提示

## 练习

1. 添加 LeCun 初始化（Var = 1/fan_in，为 SELU 激活设计）。用 LeCun 初始化 + tanh 运行 50 层实验，与 Xavier + tanh 比较。

2. 实现 GPT-2 残差缩放：在加到残差流之前，将每一层的输出乘以 1/sqrt(2*N)。有无缩放各运行 50 层，测量残差量级增长速度。

3. 创建"初始化健康检查"函数：接收网络层维度和激活类型，推荐正确的初始化，并在当前初始化会导致问题时发出警告。

4. 用 fan_in = 16 和 fan_in = 1024 运行实验。Xavier 和 Kaiming 会自适应 fan_in，但随机初始化不会。展示随着层变大，"有效"和"损坏"之间的差距如何拉大。

5. 实现正交初始化（生成随机矩阵，计算其 SVD，使用正交矩阵 U）。在 50 层 ReLU 网络上与 Kaiming 比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 权重初始化（Weight initialization） | "随机设置起始权重" | 选择初始权重值的策略，决定网络是否能训练 |
| 打破对称性（Symmetry breaking） | "让神经元不同" | 使用随机初始化确保神经元学习不同的特征，而非计算相同的函数 |
| Fan-in | "神经元的输入数量" | 传入连接的数量，决定输入方差如何在加权求和中积累 |
| Fan-out | "神经元的输出数量" | 传出连接的数量，与反向传播时维持梯度方差相关 |
| Xavier/Glorot 初始化 | "sigmoid 初始化" | Var(w) = 2/(fan_in + fan_out)，为通过 sigmoid 和 tanh 激活保持方差而设计 |
| Kaiming/He 初始化 | "ReLU 初始化" | Var(w) = 2/fan_in，考虑了 ReLU 将一半激活置零的效果 |
| 方差传播（Variance propagation） | "信号如何随层增长或收缩" | 基于权重尺度，分析激活方差如何逐层变化的数学分析 |
| 残差缩放（Residual scaling） | "GPT-2 的初始化技巧" | 将残差连接权重按 1/sqrt(2N) 缩放，防止方差通过 N 个 transformer 层增长 |
| 死亡网络（Dead network） | "什么都不训练" | 初始化不当导致所有梯度为零或所有激活饱和的网络 |
| 激活爆炸（Exploding activations） | "值趋向无穷大" | 当权重方差太高时，激活量级通过各层指数级增长 |

## 延伸阅读

- Glorot & Bengio，"Understanding the difficulty of training deep feedforward neural networks"（2010）——带方差分析的原始 Xavier 初始化论文
- He et al.，"Delving Deep into Rectifiers"（2015）——为 ReLU 网络引入了 Kaiming 初始化
- Radford et al.，"Language Models are Unsupervised Multitask Learners"（2019）——带残差缩放初始化的 GPT-2 论文
- Mishkin & Matas，"All You Need is a Good Init"（2016）——逐层顺序单位方差初始化，分析公式的实证替代方案

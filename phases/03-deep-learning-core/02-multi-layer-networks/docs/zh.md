# 多层网络与前向传播

> 一个神经元画一条线。叠加它们，你可以画出任何形状。

**类型：** 构建
**语言：** Python
**前置要求：** 第1阶段（数学基础），第03.01课（感知机）
**时长：** ~90 分钟

## 学习目标

- 从零构建带有 Layer 和 Network 类的多层网络，完成完整的前向传播
- 追踪数据通过网络每一层时的矩阵维度，识别形状不匹配问题
- 解释堆叠非线性激活函数如何使网络学习弯曲的决策边界
- 用手动调整的 sigmoid 权重，以 2-2-1 架构解决 XOR 问题

## 问题背景

单个神经元是一个画线机器，就这一件事。一条穿过数据的直线。AI 中的所有真实问题——图像识别、语言理解、下围棋——都需要曲线。将神经元堆叠成层，才能得到曲线。

1969 年，Minsky 和 Papert 证明了这一限制是致命的：单层网络无法学习 XOR。不是"难以学习"——而是数学上根本不可能。XOR 真值表将 [0,1] 和 [1,0] 放在一侧，[0,0] 和 [1,1] 放在另一侧。没有任何直线能分隔它们。

这使神经网络研究资金断绝超过十年。解决方法事后看来显而易见：停止使用单层。将神经元堆叠成层。让第一层将输入空间切割成新特征，让第二层将这些特征组合成单条直线无法做出的决策。

这种堆叠就是多层网络。它是当今所有生产级深度学习模型的基础。前向传播——数据从输入经过隐藏层流向输出——是在任何其他事情能运作之前，你需要首先构建的东西。

## 核心概念

### 层的类型：输入层、隐藏层、输出层

多层网络有三种类型的层：

**输入层**——并非真正意义上的层。它保存你的原始数据。两个特征意味着两个输入节点。这里不发生任何计算。

**隐藏层**——工作发生的地方。每个神经元接收前一层的所有输出，应用权重和偏置，然后通过激活函数传递结果。"隐藏"是因为这些值在训练数据中从未被直接观察到。

**输出层**——最终答案。对于二元分类，一个带 sigmoid 的神经元。对于多分类，每个类别一个神经元。

```mermaid
graph LR
    subgraph Input["输入层"]
        x1["x1"]
        x2["x2"]
    end
    subgraph Hidden["隐藏层（3个神经元）"]
        h1["h1"]
        h2["h2"]
        h3["h3"]
    end
    subgraph Output["输出层"]
        y["y"]
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> y
    h2 --> y
    h3 --> y
```

这是一个 2-3-1 网络：两个输入，三个隐藏神经元，一个输出。每个连接携带一个权重。每个神经元（输入层除外）携带一个偏置。

每一层产生一个称为隐藏状态（hidden state）的数值向量。对于文本，隐藏状态会增加维度——将一个词编码为 768 个数字以捕获语义含义。对于图像，则会降低维度——将数百万像素压缩成可处理的表示。学习就活在隐藏状态里。

### 神经元与激活函数

每个神经元做三件事：

1. 将每个输入乘以其对应权重
2. 对所有乘积求和并加上偏置
3. 将求和结果通过激活函数

现在使用 sigmoid 作为激活函数：

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Sigmoid 将任意数值压缩到 (0, 1) 范围内。大正数趋近于 1，大负数趋近于 0，零映射到 0.5。这条平滑曲线使学习成为可能——不同于感知机的硬阶跃，sigmoid 在每处都有梯度。

### 前向传播：数据如何流动

前向传播将输入数据逐层推送通过网络，直到到达输出。前向传播过程中不发生任何学习。它是纯计算：乘法、加法、激活，循环往复。

```mermaid
graph TD
    X["输入: [x1, x2]"] --> WH["乘以权重矩阵 W1 (2x3)"]
    WH --> BH["加偏置向量 b1 (3,)"]
    BH --> AH["对每个元素应用 sigmoid"]
    AH --> H["隐藏层输出: [h1, h2, h3]"]
    H --> WO["乘以权重矩阵 W2 (3x1)"]
    WO --> BO["加偏置向量 b2 (1,)"]
    BO --> AO["应用 sigmoid"]
    AO --> Y["输出: y"]
```

每一层依次发生三个操作：

```
z = W * input + b       （线性变换）
a = sigmoid(z)           （激活）
```

一层的输出成为下一层的输入。这就是整个前向传播。

### 矩阵维度

追踪维度是深度学习中最重要的调试技能。以下是 2-3-1 网络的维度：

| 步骤 | 操作 | 维度 | 结果形状 |
|------|------|------|---------|
| 输入 | x | -- | (2,) |
| 隐藏线性变换 | W1 * x + b1 | W1: (3, 2), b1: (3,) | (3,) |
| 隐藏激活 | sigmoid(z1) | -- | (3,) |
| 输出线性变换 | W2 * h + b2 | W2: (1, 3), b2: (1,) | (1,) |
| 输出激活 | sigmoid(z2) | -- | (1,) |

规则：第 k 层的权重矩阵 W 的形状为 (当前层神经元数, 上一层神经元数)。行数对应当前层，列数对应上一层。如果形状不匹配，就有 bug。

### 通用近似定理

1989 年，George Cybenko 证明了一个了不起的结论：具有单个隐藏层和足够多神经元的神经网络，可以以任意精度近似任何连续函数。

这并不意味着单个隐藏层总是最好的。它意味着该架构在理论上是有能力的。实践中，更深的网络（更多层，每层神经元更少）学习同样的函数所需的总参数量，远少于浅而宽的网络。这就是深度学习有效的原因。

直觉：隐藏层中的每个神经元学习一个"波峰"或特征。在正确位置放置足够多的波峰，可以近似任何平滑曲线。神经元越多，波峰越多，近似越好。

```mermaid
graph LR
    subgraph FewNeurons["4 个隐藏神经元"]
        A["粗略近似"]
    end
    subgraph MoreNeurons["16 个隐藏神经元"]
        B["较好近似"]
    end
    subgraph ManyNeurons["64 个隐藏神经元"]
        C["接近完美拟合"]
    end
    FewNeurons --> MoreNeurons --> ManyNeurons
```

### 可组合性

神经网络是可组合的。你可以堆叠、链式连接、并行运行它们。Whisper 模型使用一个编码器网络处理音频，使用一个独立的解码器网络生成文本。现代大语言模型（LLM）是纯解码器架构。BERT 是纯编码器架构。T5 是编码器-解码器架构。架构选择决定了模型能做什么。

## 实现

纯 Python，不使用 numpy。所有矩阵运算从零手写。

### 第一步：Sigmoid 激活函数

```python
import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
```

对 [-500, 500] 的截断防止溢出。`math.exp(500)` 很大但有限。`math.exp(1000)` 是无穷大。

### 第二步：Layer 类

深度学习中最重要的操作是矩阵乘法（matmul）。每一层、每个注意力头、每次前向传播——全是 matmul。线性层接收输入向量，乘以权重矩阵，加上偏置向量：y = Wx + b。这个简单方程占神经网络计算量的 90%。

Layer 持有一个权重矩阵和一个偏置向量。其 forward 方法接收输入向量，返回激活后的输出。

```python
class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
```

权重矩阵的形状为 (n_neurons, n_inputs)。每行是一个神经元跨所有输入的权重。forward 方法遍历神经元，计算加权和加偏置，应用 sigmoid，并收集结果。

### 第三步：Network 类

网络是层的列表。前向传播将它们链接起来：第 k 层的输出送入第 k+1 层。

```python
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
```

这就是整个前向传播——四行逻辑。数据进入，流经每一层，从另一端输出。

### 第四步：用手动调整的权重解决 XOR

第 01 课中，我们通过组合 OR、NAND 和 AND 感知机解决了 XOR。现在用 Layer 和 Network 类完成同样的事情。2-2-1 架构：两个输入，两个隐藏神经元，一个输出。

```python
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")
```

大权重值（20，-20）使 sigmoid 表现得像阶跃函数。第一个隐藏神经元近似 OR，第二个近似 NAND，输出神经元将它们组合成 AND，即 XOR。

### 第五步：圆形分类

一个更难的问题：将二维点分类为圆心在原点、半径为 0.5 的圆内或圆外。这需要弯曲的决策边界——对单个感知机来说不可能。

```python
import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])
```

使用随机权重时，网络分类效果不好。但前向传播仍然可以运行。这就是重点——前向传播只是计算。学习正确的权重是反向传播的任务，将在第 03 课介绍。

```python
correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"Accuracy with random weights: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
```

随机权重给出较差的准确率——通常比猜测多数类别还要差。训练后（第 03 课），同样的带 8 个隐藏神经元的架构，将画出一条弯曲边界，将内部与外部分开。

## 工程应用

PyTorch 用四行代码完成以上所有工作：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)
```

`nn.Linear(2, 8)` 就是你的 Layer 类：形状为 (8, 2) 的权重矩阵，形状为 (8,) 的偏置向量。`nn.Sigmoid()` 就是你的 sigmoid 函数，逐元素应用。`nn.Sequential` 就是你的 Network 类：按顺序链接各层。

区别在于速度和规模。PyTorch 在 GPU 上运行，处理数百万样本的批次，并自动计算反向传播的梯度。但前向传播逻辑与你刚才从零构建的完全相同。

## 交付物

本课产出一个用于设计网络架构的可复用提示：

- `outputs/prompt-network-architect.md`

当你需要决定层数、每层神经元数以及激活函数时，使用它。

## 练习

1. 构建一个 2-4-2-1 网络（两个隐藏层），在 XOR 数据上用随机权重运行前向传播。打印中间隐藏层的输出，观察表示在每一层如何转换。

2. 将圆形分类器的隐藏层大小从 8 改为 2，再改为 32。每次用随机权重运行前向传播。隐藏神经元数量会改变输出范围或分布吗？为什么？

3. 在 Network 类上实现一个 `count_parameters` 方法，返回可训练权重和偏置的总数。在 784-256-128-10 网络（经典 MNIST 架构）上测试它。有多少参数？

4. 构建一个 3-4-4-2 网络的前向传播。输入归一化到 0-1 的 RGB 颜色值，观察两个输出。这是一个简单的双类别颜色分类器架构。

5. 将 sigmoid 替换为"leaky step"函数：当 z < 0 时返回 0.01 * z，否则返回 1.0。用第四步中相同的手动调整权重在 XOR 上运行前向传播。它还能正确工作吗？为什么平滑的 sigmoid 比硬截断更受欢迎？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 前向传播（Forward pass） | "运行模型" | 将输入通过每一层推送——乘以权重，加偏置，激活——以产生输出 |
| 隐藏层（Hidden layer） | "中间部分" | 输入层和输出层之间的任何层，其值在训练数据中从未直接被观察到 |
| 多层网络（Multi-layer network） | "深度神经网络" | 顺序堆叠的神经元层，每层的输出作为下一层的输入 |
| 激活函数（Activation function） | "非线性函数" | 在线性变换后应用的函数，为决策边界引入曲线 |
| Sigmoid | "S 形曲线" | σ(z) = 1/(1+e^(-z))，将任意实数压缩到 (0,1)，处处光滑可微 |
| 权重矩阵（Weight matrix） | "参数" | 形状为 (当前层神经元数, 上一层神经元数) 的矩阵，包含可学习的连接强度 |
| 偏置向量（Bias vector） | "偏移量" | 矩阵乘法后加入的向量，让神经元在所有输入为零时也能激活 |
| 通用近似定理（Universal approximation） | "神经网络能学任何东西" | 具有足够多神经元的单隐藏层可以近似任何连续函数——但"足够多"可能意味着数十亿 |
| 线性变换（Linear transformation） | "矩阵乘法步骤" | z = W * x + b，激活前的计算，将输入映射到新空间 |
| 决策边界（Decision boundary） | "分类器切换的地方" | 输入空间中网络输出跨越分类阈值的曲面 |

## 延伸阅读

- Michael Nielsen，"Neural Networks and Deep Learning"，第 1-2 章（http://neuralnetworksanddeeplearning.com/）——对前向传播和网络结构最清晰的免费解释，带有交互可视化
- Cybenko，"Approximation by Superpositions of a Sigmoidal Function"（1989）——通用近似定理的原始论文，出人意料地易读
- 3Blue1Brown，"But what is a neural network?"（https://www.youtube.com/watch?v=aircAruvnKk）——20 分钟可视化讲解层、权重和前向传播，建立正确的心智模型
- Goodfellow, Bengio, Courville，"Deep Learning"，第 6 章（https://www.deeplearningbook.org/）——多层网络的标准参考，免费在线

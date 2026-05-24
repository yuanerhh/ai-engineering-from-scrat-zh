# 链式法则与自动微分

> 链式法则是每个神经网络学习的引擎。

**类型：** 实践
**语言：** Python
**前置要求：** 阶段 1，第 04 课（导数与梯度）
**时间：** 约 90 分钟

## 学习目标

- 构建一个极简自动微分引擎（Value 类），通过记录运算并使用反向模式自动微分计算梯度
- 通过拓扑排序实现计算图的前向传播和反向传播
- 仅使用从零构建的自动微分引擎，在 XOR 问题上构建并训练多层感知机
- 通过梯度检验（与数值有限差分对比）验证自动微分的正确性

## 问题

你能计算简单函数的导数。但神经网络不是简单函数。它是数百个函数的复合：矩阵乘法、加偏置、激活函数、再矩阵乘法、softmax、交叉熵损失。输出是函数的函数的函数。

要训练网络，你需要损失函数对每个权重的梯度。对于数百万个参数，手动推导是不可能的。用数值方法（有限差分）又太慢。

链式法则给你数学。自动微分给你算法。两者结合，能让你在与单次前向传播成比例的时间内，计算任意复合函数的精确梯度。

这就是 PyTorch、TensorFlow 和 JAX 的工作原理。你将从零构建一个迷你版本。

## 概念

### 链式法则

如果 `y = f(g(x))`，则 `y` 对 `x` 的导数为：

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

沿链路相乘各导数。每个环节贡献其局部导数。

例子：`y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

对于更深的复合，链条继续延伸：

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

神经网络中的每一层都是这条链中的一个环节。

### 计算图

计算图让链式法则变得可视化。每个运算变成一个节点。数据沿图向前流动，梯度向后流动。

**前向传播（计算值）：**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (乘法)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (加法)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["输出 y"]
```

**反向传播（计算梯度）：**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1，因为 c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

反向传播在每个节点应用链式法则，将梯度从输出传播回输入。

### 前向模式 vs 反向模式

有两种方式通过图应用链式法则。

**前向模式**从输入出发，将导数向前推。它计算 `dx/dx = 1` 并通过每个运算传播。适合输入少、输出多的情况。

```
前向模式：设种子 dx/dx = 1，向前传播

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**反向模式**从输出出发，将梯度向后拉。它计算 `dy/dy = 1` 并逆向通过每个运算传播。适合输入多、输出少的情况。

```
反向模式：设种子 dy/dy = 1，向后传播

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

神经网络有数百万个输入（权重）和一个输出（损失）。反向模式只需一次反向传播就能计算所有梯度。这就是反向传播使用反向模式的原因。

| 模式 | 种子 | 方向 | 适用场景 |
|------|------|------|----------|
| 前向 | `dx_i/dx_i = 1` | 输入到输出 | 输入少，输出多 |
| 反向 | `dy/dy = 1` | 输出到输入 | 输入多，输出少（神经网络） |

### 对偶数与前向模式

前向模式可以用对偶数优雅地实现。对偶数的形式为 `a + b*epsilon`，其中 `epsilon^2 = 0`。

```
对偶数：(值, 导数)

(2, 1) 表示：值为 2，对 x 的导数为 1

算术规则：
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

将输入变量的导数种子设为 1，导数就会通过每次运算自动传播。

### 构建自动微分引擎

自动微分引擎需要三件事：

1. **值包装。** 将每个数字封装在存储其值和梯度的对象中。
2. **图记录。** 每次运算记录其输入和局部梯度函数。
3. **反向传播。** 对图进行拓扑排序，然后逆向遍历，在每个节点应用链式法则。

这正是 PyTorch 的 `autograd` 所做的事。`torch.Tensor` 类封装值，在 `requires_grad=True` 时记录运算，在调用 `.backward()` 时计算梯度。

### PyTorch 自动微分的内部原理

当你编写 PyTorch 代码时：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch 在内部：

1. 为 `x` 创建一个 `Tensor` 节点，设置 `requires_grad=True`
2. 每次运算（`**`、`*`、`+`）创建一个新节点并记录反向函数
3. `y.backward()` 触发通过记录图的反向模式自动微分
4. 每个节点的 `grad_fn` 计算局部梯度并将其传递给父节点
5. 梯度通过加法（而非替换）累积在 `.grad` 属性中

计算图是动态的（按运行定义）。每次前向传播都会构建一个新图。这就是 PyTorch 支持在模型内使用控制流（if/else、循环）的原因。

## 动手实现

### 第一步：Value 类

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

每个 `Value` 存储其数值、梯度（初始为零）、反向函数，以及指向产生它的子节点的指针。

### 第二步：带梯度追踪的算术运算

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

每次运算都创建一个闭包，知道如何计算局部梯度并乘以上游梯度（`out.grad`）。`+=` 处理一个值被多次使用的情况。

### 第三步：反向传播

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

拓扑排序确保每个节点的梯度在传播给其子节点之前已完全计算。种子梯度为 1.0（dy/dy = 1）。

### 第四步：完整引擎所需的更多运算

基本 Value 类处理加法、乘法和 relu。真正的自动微分引擎需要更多。以下是构建神经网络所需的运算：

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**为什么每个运算都重要：**

| 运算 | 反向规则 | 用途 |
|------|----------|------|
| `__sub__` | 复用加法 + 取负 | 损失计算（pred - target） |
| `__pow__` | n * x^(n-1) | 多项式激活函数，MSE（误差²） |
| `__truediv__` | 复用乘法 + pow(-1) | 归一化，学习率缩放 |
| `exp` | exp(x) * 上游梯度 | Softmax，对数似然 |
| `log` | (1/x) * 上游梯度 | 交叉熵损失，对数概率 |
| `tanh` | (1 - tanh²) * 上游梯度 | 经典激活函数 |

巧妙之处：`__sub__` 和 `__truediv__` 通过已有运算实现，免费获得正确的梯度，因为链式法则通过底层的加/乘/幂运算自动复合。

### 第五步：从零构建迷你 MLP

有了完整的 Value 类，你可以构建神经网络。不用 PyTorch，不用 NumPy，只用 Value 和链式法则。

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

`Neuron` 计算 `tanh(w1*x1 + w2*x2 + ... + b)`。`Layer` 是神经元列表。`MLP` 将层堆叠起来。每个权重都是一个 `Value`，因此调用 `loss.backward()` 会将梯度传播到每个参数。

**在 XOR 上训练：**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 个输入，4 个隐藏神经元，1 个输出

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR 模式（使用 -1/1 对应 tanh）

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\n训练后的预测结果：")
for x, y in zip(xs, ys):
    print(f"  输入={x}  目标={y:2d}  预测={model(x).data:6.3f}")
```

这就是 micrograd。一个纯 Python 实现的完整神经网络训练循环，配备自动微分。每个商业深度学习框架都在大规模地做同样的事情。

### 第六步：梯度检验

怎么知道你的自动微分是正确的？将其与数值导数对比。这就是梯度检验。

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

在复杂表达式上测试：

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"自动微分: {ad:.8f}")
print(f"数值计算: {num:.8f}")
print(f"差异:     {diff:.2e}")
# 差异应小于 1e-5
```

添加新运算时，梯度检验不可或缺。如果反向传播有 bug，数值检验会发现它。每个严肃的深度学习实现在开发期间都会运行梯度检验。

**何时使用梯度检验：**

| 场景 | 是否做梯度检验？ |
|------|----------------|
| 向自动微分引擎添加新运算 | 是，始终做 |
| 调试不收敛的训练循环 | 是，先检查梯度 |
| 生产训练 | 否，太慢（每个参数需两次前向传播） |
| 自动微分代码的单元测试 | 是，自动化它 |

### 第七步：与手动计算对比验证

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

手动验证：`y = relu(x1*x2 + 1)`。由于 `x1*x2 + 1 = 7 > 0`，relu 等同于恒等函数。`dy/dx1 = x2 = 3`，`dy/dx2 = x1 = 2`。引擎结果完全匹配。

## 实际使用

### 与 PyTorch 对比验证

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

梯度相同。你的引擎计算出与 PyTorch 相同的结果，因为数学是一样的：基于链式法则的反向模式自动微分。

### 更复杂的表达式

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## 交付产出

本节课产出：
- `outputs/skill-autodiff.md` — 构建和调试自动微分系统的技能文档
- `code/autodiff.py` — 可扩展的极简自动微分引擎

这里构建的 Value 类是阶段 3 中神经网络训练循环的基础。

## 练习

1. **添加 `__pow__`。** 向 Value 类添加 `__pow__`，使其能计算 `x ** n`。验证 `d/dx(x^3)` 在 `x=2` 处等于 `12.0`。

2. **添加 tanh 激活函数。** 验证 `tanh'(0) = 1` 且 `tanh'(2) ≈ 0.0707`。

3. **单个神经元的计算图。** 为 `y = relu(w1*x1 + w2*x2 + b)` 构建计算图，计算全部五个梯度，并与 PyTorch 对比验证。

4. **实现前向模式自动微分。** 使用对偶数实现前向模式。创建 `Dual` 类，验证其与反向模式引擎给出相同的导数。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| 链式法则（Chain rule）| "导数相乘" | 复合函数的导数等于各函数局部导数之积，在正确的点求值 |
| 计算图（Computational graph）| "网络图" | 有向无环图，节点是运算，边承载前向的值或后向的梯度 |
| 前向模式（Forward mode）| "向前推导数" | 将导数从输入传播到输出的自动微分，每个输入变量需一次传播 |
| 反向模式（Reverse mode）| "反向传播" | 将梯度从输出传播到输入的自动微分，每个输出变量需一次传播 |
| 自动微分（Autograd）| "自动梯度" | 记录值上的运算、构建图、通过链式法则计算精确梯度的系统 |
| 对偶数（Dual numbers）| "值加导数" | 形如 a + b*epsilon（epsilon²=0）的数，能在算术运算中携带导数信息 |
| 拓扑排序（Topological sort）| "依赖顺序" | 对图节点排序使每个节点都在其所有依赖之后，梯度正确传播所必需 |
| 梯度累积（Gradient accumulation）| "相加而非替换" | 当一个值参与多次运算时，其梯度是所有传入梯度贡献的总和 |
| 动态图（Dynamic graph）| "按运行定义" | 每次前向传播时重新构建的计算图，允许在模型中使用 Python 控制流（PyTorch 风格） |
| 梯度检验（Gradient checking）| "数值验证" | 将自动微分梯度与数值有限差分梯度对比，验证正确性，调试必备 |
| 多层感知机（MLP）| "全连接网络" | 具有一个或多个隐藏层的神经网络，每个神经元计算加权和加偏置后应用激活函数 |
| 神经元（Neuron）| "加权和加激活" | 基本单元：output = activation(w1*x1 + w2*x2 + ... + b)，权重和偏置是可学习参数 |

## 延伸阅读

- [3Blue1Brown：反向传播微积分](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 神经网络中链式法则的可视化讲解
- [PyTorch 自动微分机制](https://pytorch.org/docs/stable/notes/autograd.html) — 真实系统的工作原理
- [Baydin 等，机器学习中的自动微分综述](https://arxiv.org/abs/1502.05767) — 全面参考资料

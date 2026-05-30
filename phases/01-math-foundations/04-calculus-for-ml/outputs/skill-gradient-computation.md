---
name: skill-gradient-computation
description: 计算常见 ML 损失函数的梯度并选择合适的求导方法
version: 1.0.0
phase: 1
lesson: 4
tags: [calculus, gradients, backpropagation]
---

# ML 梯度计算

神经网络中损失函数、激活函数和层操作的梯度计算实用参考。

## 决策清单

1. 函数是否由简单基本函数（幂函数、指数、对数、三角函数）复合而成？使用解析求导和链式法则。
2. 函数是否为自定义或黑盒操作？使用数值微分：`(f(x+h) - f(x-h)) / (2h)`，其中 h = 1e-7。
3. 函数是否由 PyTorch/JAX 的张量操作构建？交给自动微分处理。用数值方法验证结果。
4. 是否需要标量损失对权重矩阵的梯度？沿计算图逐节点应用链式法则。
5. 是否存在不可微操作（argmax、取整、采样）？使用直通估计器（straight-through estimator）或重参数化技巧（reparameterization trick）。

## 各方法的适用场景

| 方法 | 适用场景 | 开销 |
|---|---|---|
| 解析法（手推） | 简单函数、验证自动微分输出 | 运行时零开销 |
| 数值法（有限差分） | 调试、梯度检验、黑盒函数 | 每 n 个参数需 2n 次前向传播 |
| 自动微分 | 任何可微计算图（默认方式） | 一次反向传播 |
| 符号法（SymPy、Mathematica） | 推导论文中的闭合形式梯度 | 仅编译时开销 |

## 常用导数速查表

| 函数 | f(x) | f'(x) | ML 使用场景 |
|---|---|---|---|
| MSE 损失 | (1/n) sum(y_hat - y)^2 | (2/n)(y_hat - y) | 回归 |
| 交叉熵（二分类） | -(y log(p) + (1-y) log(1-p)) | p - y（接 sigmoid 后） | 二分类 |
| 交叉熵（多分类） | -log(p_true_class) | p - one_hot(y)（接 softmax 后） | 多分类 |
| Sigmoid | 1 / (1 + e^(-x)) | sigma(x) * (1 - sigma(x)) | 输出门、二值输出 |
| Tanh | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh(x)^2 | 隐藏层激活（旧式） |
| ReLU | max(0, x) | x > 0 时为 1，x < 0 时为 0 | 默认隐藏层激活 |
| Leaky ReLU | max(0.01x, x) | x > 0 时为 1，x < 0 时为 0.01 | 避免神经元死亡 |
| GELU | x * Phi(x) | Phi(x) + x * phi(x) | Transformer |
| Softmax_i | e^(x_i) / sum(e^(x_j)) | i=j 时为 s_i(1 - s_i)，i≠j 时为 -s_i*s_j | 输出层（Jacobian） |
| Log-softmax | x_i - log(sum(e^(x_j))) | 第 i 个元素为 1 - softmax(x_i) | 数值稳定的交叉熵 |
| 线性层 | y = Wx + b | dL/dW = dL/dy * x^T，dL/db = dL/dy | 每一层 |
| L2 正则化 | lambda * sum(w^2) | 2 * lambda * w | 权重衰减 |
| L1 正则化 | lambda * sum(\|w\|) | lambda * sign(w) | 稀疏性 |

## 常见错误

- 忘记批次平均损失（MSE、交叉熵）中的 1/n 因子。梯度按批量大小缩放。
- 将 softmax 梯度计算为向量，而实际上它是 Jacobian 矩阵。对于交叉熵 + softmax 的组合，梯度可以简化为 (p - y)，从而避免计算完整的 Jacobian。
- 链式法则的方向弄反。应从损失向后推导：dL/dW = dL/dy * dy/dW。
- 数值微分时 h 取值过大（h = 0.1）或过小（h = 1e-15）。float64 时请使用 h = 1e-7。
- 忘记 ReLU 在 x = 0 处的梯度未定义。实践中将其设为 0 或 0.5。

## 梯度检验方法

```
对每个参数 w：
  numeric_grad = (loss(w + h) - loss(w - h)) / (2h)
  auto_grad = 反向传播值
  relative_error = |numeric - auto| / max(|numeric|, |auto|, 1e-8)
  assert relative_error < 1e-5
```

相对误差超过 1e-3 说明存在问题。在 1e-5 到 1e-3 之间时需要进一步排查。

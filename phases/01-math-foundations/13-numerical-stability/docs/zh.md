# 数值稳定性

> 浮点数是一个有漏洞的抽象。它会在训练期间咬你一口，而你不会预料到这一点。

**类型：** 实践
**语言：** Python
**前置条件：** 第 1 阶段，第 01-04 课
**时间：** ~120 分钟

## 学习目标

- 使用减最大值技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性抵消
- 用中心有限差分法对解析梯度进行数值梯度校验
- 解释为什么训练时 bfloat16 优于 float16，以及损失缩放如何防止梯度下溢

## 问题所在

你的模型训练了三个小时，然后损失变成了 NaN。你加了一行打印语句。第 9000 步时 logit 还是正常的。第 9001 步时它们变成了 `inf`。到第 9002 步，每个梯度都是 `nan`，训练彻底死亡。

或者：你的模型训练完成了，但精度比论文中宣称的低了 2%。你检查了所有东西。架构匹配。超参数匹配。数据匹配。问题在于论文使用了 float32，而你使用了没有正确缩放的 float16。三十二位累积的舍入误差悄悄吃掉了你的精度。

或者：你从头实现了交叉熵损失。它在小 logit 上运行正常。当 logit 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 大于 float32 所能表示的最大值。每个 ML 框架都用两行代码的技巧处理这个问题。你不知道这个技巧的存在。

数值稳定性不是一个理论问题。它是决定一次训练成功还是悄悄失败的关键。你将要调试的每一个严重 ML bug，最终都会追溯到浮点数。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。一个浮点数有三个部分：符号位、指数和尾数（有效数字）。

```
Float32 布局（共 32 位）：
[1 符号位] [8 指数位] [23 尾数位]

值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度（有效数字的位数）。指数决定范围（数字可以多大或多小）。

```
格式       位数   指数位  尾数位  十进制位数  大致范围
float64    64     11      52      ~15-16     +/- 1.8e308
float32    32     8       23      ~7-8       +/- 3.4e38
float16    16     5       10      ~3-4       +/- 65,504
bfloat16   16     8       7       ~2-3       +/- 3.4e38
```

float32 给你大约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但区分不了 1.00000001 和 1.00000002。超过 7 位之后，一切都是舍入噪声。

float16 给你大约 3 位。它能表示的最大数是 65,504。对于 logit、梯度和激活值经常超过这个数的 ML 来说，这是令人不安的小。

bfloat16 是 Google 对 float16 范围问题的解答。它与 float32 有相同的 8 位指数（相同的范围，最大到 3.4e38），但只有 7 位尾数（精度比 float16 低）。对于训练神经网络，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法用二进制浮点精确表示。在二进制中，它是一个循环小数：

```
0.1 的二进制 = 0.0001100110011001100110011...（无限循环）
```

Float32 将其截断为 23 位尾数。存储的值大约是 0.100000001490116。类似地，0.2 存储为大约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 的影响：

1. 像 `if loss < threshold` 这样的损失比较可能给出错误答案
2. 累加许多小值（数千步的梯度更新）会偏离真实求和
3. 如果用 `==` 比较浮点数，校验和及可重现性测试会失败

修复方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当两个几乎相等的浮点数相减时，有效数字相互抵消，留下的舍入噪声被提升为前导数字。

```
a = 1.0000001    （float32 中存储为 1.00000011920929）
b = 1.0000000    （float32 中存储为 1.00000000000000）

真实差值：  0.0000001
计算结果：  0.00000011920929

相对误差：19.2%
```

仅一次减法就产生了 19% 的相对误差。在 ML 中，以下情况会发生这种问题：

- 计算均值较大的数据方差：当 E[x] 很大时计算 `E[x^2] - E[x]^2`
- 相减几乎相等的对数概率
- 使用过小 epsilon 计算有限差分梯度

修复方法：重新整理公式以避免相减大而近似相等的数。对于方差，使用 Welford 算法或先对数据中心化。对于对数概率，全程在对数空间中工作。

### 上溢和下溢

上溢发生在结果太大无法表示时。下溢发生在结果太小时（比最小可表示正数更接近零）。

```
Float32 边界：
  最大值：    3.4028235e+38
  最小正规数：1.175e-38
  最小非规数：1.401e-45
  上溢：任何 > 3.4e38 的值变成 inf
  下溢：任何 < 1.4e-45 的值变成 0.0
```

`exp()` 函数是 ML 中上溢的主要来源：

```
exp(88.7)  = 3.40e+38   （刚好在 float32 范围内）
exp(89.0)  = inf         （上溢）
exp(-87.3) = 1.18e-38   （刚好在下溢之上）
exp(-104)  = 0.0         （下溢为零）
```

`log()` 函数在另一个方向触发：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （正常）
log(1e-46) = -inf        （输入下溢为 0，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。没有正确技巧的情况下，组合 `log(exp(x))` 是一个雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 就会上溢。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都会下溢为零，`log(0)` 就是 `-inf`。

技巧：在求指数之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

原理：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不可能上溢。求和中至少有一项是 1，所以和至少为 1，`log(1) = 0`。不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加减 c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （提取 exp(c)）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

令 `c = max(x)`，上溢即被消除。

这个技巧在 ML 中随处可见：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要减最大值技巧

Softmax 将 logit 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不使用技巧时，logit [100, 101, 102] 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些超出了 float32（最大 ~3.4e38）？是的：
exp(88.7) 已经在 float32 的极限。
exp(100) 在 float32 中 = inf。
```

使用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算是安全的。这不是优化，而是正确性的要求。

### NaN 和 Inf：检测与预防

`nan`（非数字）和 `inf`（无穷大）会病毒式地在计算中传播。梯度更新中的一个 `nan` 会使权重变成 `nan`，这会使之后的每个输出都变成 `nan`。训练在一步之内就死亡。

`inf` 的来源：
- 大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- `float32` 累积时溢出

`nan` 的来源：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何涉及已有 `nan` 的算术

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 则返回 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 则返回 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 则返回 True
```

预防策略：

1. 对 `exp()` 的输入进行截断：`exp(clamp(x, -80, 80))`
2. 在分母中加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 调试时在每次前向传播后检查 `nan`/`inf`

### 数值梯度校验

解析梯度（来自反向传播）可能有 bug。数值梯度校验通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度，远好于仅有 O(h) 精度的前向差分 `(f(x+h) - f(x)) / h`。

选择 h：太大时近似不准确，太小时灾难性抵消会破坏结果。`h = 1e-5` 到 `1e-7` 是典型值。

校验：计算解析梯度和数值梯度之间的相对差异。

```
相对误差 = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- 相对误差 < 1e-7：完美，梯度正确
- 相对误差 < 1e-5：可接受，可能正确
- 相对误差 > 1e-3：有问题
- 相对误差 > 1：梯度完全错误

实现新的层或损失函数时务必检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 用于此目的。

### 混合精度训练

现代 GPU 有专用硬件（Tensor Core），计算 float16 矩阵乘法比 float32 快 2-8 倍。混合精度训练利用这一点：

```
1. 维护 float32 主权重副本
2. 前向传播使用 float16（快速）
3. 用 float32 计算损失（防止溢出）
4. 反向传播使用 float16（快速）
5. 将梯度缩放到 float32
6. 更新 float32 主权重
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 将低于 ~6e-8 的任何值下溢为零。模型停止学习，因为所有梯度更新都是零。

解决方法是损失缩放：

```
1. 将损失乘以大的缩放因子（例如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度扩大 1024 倍（推到 float16 下溢之上）
4. 在更新权重之前将梯度除以 1024
5. 净效果：相同的更新，但没有下溢
```

动态损失缩放自动调整缩放因子。从大值开始（65536）。如果梯度溢出到 `inf`，将其减半。如果 N 步没有溢出，则加倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中胜出

```
float16:   [1 符号位] [5 指数位]  [10 尾数位]
bfloat16:  [1 符号位] [8 指数位]  [7 尾数位]
```

float16 有更多精度（10 位尾数对比 7 位），但范围有限（最大 ~65,504）。bfloat16 精度较低，但范围与 float32 相同（最大 ~3.4e38）。

对于训练神经网络：

- 训练峰值期间，激活值和 logit 经常超过 65,504。float16 溢出；bfloat16 可以处理。
- float16 需要损失缩放，而 bfloat16 通常不需要，因为其范围覆盖了梯度幅度谱。
- bfloat16 是 float32 的简单截断：丢弃尾数的低 16 位。转换简单且在指数上无损。

float16 适用于值有界且精度更重要的推理。bfloat16 适用于范围更重要的训练。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）都原生支持 bfloat16。

### 梯度裁剪

梯度爆炸发生在梯度通过多层呈指数增长时（在 RNN、深层网络和 Transformer 中很常见）。一个大梯度可以在一步内破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立地截断每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的，是标准选择。

典型值：Transformer 用 `max_norm=1.0`，强化学习用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度裁剪不是一个临时方案。它是一个安全机制。没有它，一个异常批次产生的梯度就足以毁掉数周的训练。

### 归一化层作为数值稳定器

批归一化、层归一化和 RMS 归一化通常被视为帮助训练收敛的正则化器。它们也是数值稳定器。

没有归一化，激活值可以通过各层指数增长或缩小：

```
第 1 层：值在 [0, 1]
第 5 层：值在 [0, 100]
第 10 层：值在 [0, 10,000]
第 50 层：值在 [0, inf]
```

归一化在每一层重新定位和缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常是 1e-5）在所有激活值相同时防止除以零。学习参数 `gamma` 和 `beta` 让网络恢复它需要的任何尺度。

这使值在整个网络中保持在数值安全范围内，防止前向传播中的上溢和反向传播中的梯度爆炸。

### 常见 ML 数值 Bug

**Bug：损失在几个 epoch 后变成 NaN。**
原因：logit 增长过大，softmax 上溢。或学习率过高导致权重发散。
修复：使用稳定 softmax（减最大值），降低学习率，添加梯度裁剪。

**Bug：损失卡在 log(类别数)。**
原因：模型输出接近均匀概率。通常意味着梯度正在消失或模型根本没有学习。
修复：检查数据标签是否正确，验证损失函数，检查死亡 ReLU。

**Bug：验证精度比预期低 1-3%。**
原因：混合精度没有适当的损失缩放。梯度下溢静默地将小更新清零。
修复：启用动态损失缩放，或改用 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死亡 ReLU 神经元（所有输入为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个 GPU 上工作，但在另一个 GPU 上给出不同结果。**
原因：非确定性浮点累加顺序。GPU 并行规约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度惩罚。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：未使用减最大值技巧直接将原始 logit 传入 `exp()`。
修复：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度幅度或高于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

## 实践

### 第 1 步：演示浮点精度限制

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差值: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素 softmax 与稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"朴素:  {softmax_naive(safe_logits)}")
print(f"稳定: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 第 3 步：实现稳定 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"朴素:  {logsumexp_naive(safe):.6f}")
print(f"稳定: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 第 4 步：实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"朴素:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度校验

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  参数 {i}: 解析={a:.8f} 数值={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 应用

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪范数: {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保留: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("好的", [1.0, 2.0, 3.0])
check_tensor("坏的", [1.0, float('nan'), 3.0])
check_tensor("糟糕", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况演示见 `code/numerical.py`。

## 交付物

本课程产出：
- `code/numerical.py`：稳定 softmax、log-sum-exp、交叉熵、梯度校验和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：诊断训练中 NaN/Inf 及数值问题的提示

这些稳定实现将在第 3 阶段构建训练循环时和第 4 阶段实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 用朴素公式 `E[x^2] - E[x]^2` 在 float32 中计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford 在线算法计算。将两个结果与真实方差（0.6667）比较误差。

2. **精度搜寻。** 找出最小的正 float32 值 `x` 使得 Python 中 `1.0 + x == 1.0`。这就是机器精度。验证它与 `numpy.finfo(numpy.float32).eps` 匹配。

3. **Log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值非常负（-1000）。验证它在朴素版本失败的地方给出正确结果。

4. **对神经网络层做梯度校验。** 实现一个单线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：在 [1e-9, 1e-3] 范围内创建随机梯度，转换为 float16，测量有多少比例变为零。然后应用损失缩放（乘以 1024），转换为 float16，缩放回来，再次测量零的比例。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| 机器精度 | "精度极限" | 使得 `1.0 + e != 1.0` 的最小值 e。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致精度损失" | 相减几乎相等的浮点数时，有效数字抵消，舍入噪声主导结果。 |
| 上溢 | "数字太大" | 结果超过最大可表示值，变成 inf。exp(89) 在 float32 中上溢。 |
| 下溢 | "数字太小" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 在 float32 中下溢。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，防止上溢和下溢。用于 softmax、交叉熵和对数概率数学。 |
| 稳定 softmax | "不爆炸的 softmax" | 在求指数之前减去 max(logits)。数学上与朴素版本相同，不可能上溢。 |
| 梯度校验 | "验证你的反向传播" | 将反向传播的解析梯度与有限差分数值梯度进行比较，以捕捉实现 bug。 |
| 混合精度 | "float16 前向，float32 反向" | 对速度关键操作使用低精度浮点，对数值敏感操作使用高精度浮点。典型加速 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播之前将损失乘以大常数，使梯度保持在 float16 可表示范围内，然后在权重更新之前除以同一常数。 |
| bfloat16 | "脑浮点" | Google 的 16 位格式，有 8 位指数（与 float32 相同范围）和 7 位尾数（比 float16 精度低）。适合训练。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数字" | 来自未定义操作（0/0、inf-inf、sqrt(-1)）的特殊浮点值。通过所有后续算术传播。 |
| Inf | "无穷大" | 来自上溢或除以零的特殊浮点值。可以组合产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。速度慢但可靠，用于验证。 |

## 延伸阅读

- [每个计算机科学家都应该了解的浮点算术（Goldberg 1991）](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，内容密集但完整
- [混合精度训练（Micikevicius 等，2018）](https://arxiv.org/abs/1710.03740) -- 介绍 float16 训练损失缩放的 NVIDIA 论文
- [AMP：自动混合精度（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实践指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为何为 TPU 选择此格式
- [Kahan 求和（维基百科）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和舍入误差的算法

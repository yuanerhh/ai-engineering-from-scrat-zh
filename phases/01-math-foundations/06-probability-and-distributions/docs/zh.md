# 概率与分布

> 概率是 AI 表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 1，第 01-04 课
**时间：** 约 75 分钟

## 学习目标

- 从零实现伯努利、类别、泊松、均匀和正态分布的 PMF 与 PDF
- 计算期望值和方差，并用中心极限定理解释为何高斯分布无处不在
- 构建带数值稳定技巧（减去最大 logit）的 softmax 和 log-softmax 函数
- 从 logits 计算交叉熵损失，并将其与负对数似然关联起来

## 问题

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 5 万个候选词中选出下一个词。扩散模型通过从学到的分布中采样来生成图像。这些都是概率在发挥作用。

模型的每次预测都是一个概率分布。每个损失函数都在衡量预测分布与真实分布之间的差距。每次训练步骤都在调整参数，让一个分布看起来更像另一个。没有概率，你读不懂任何一篇 ML 论文，调不好任何一个模型，也无法理解为什么训练损失会变成 NaN。

## 概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到 0 到 1 之间的数字。

```
抛硬币：
  S = {正, 反}
  P(正) = 0.5，  P(反) = 0.5

掷单骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了所有概率：
1. 对任意事件 A，P(A) >= 0
2. P(S) = 1（总会发生某件事）
3. 当 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

其他所有内容（贝叶斯定理、期望、分布）都从这三条规则推导出来。

### 条件概率与独立性

P(A|B) 是在 B 发生的条件下 A 发生的概率。

```
P(A|B) = P(A 且 B) / P(B)

例子：一副扑克牌
  P(K | 人头牌) = P(K 且人头牌) / P(人头牌)
               = (4/52) / (12/52)
               = 4/12 = 1/3
```

当知道一件事对另一件事没有任何影响时，两个事件独立：

```
独立：   P(A|B) = P(A)
等价于：P(A 且 B) = P(A) * P(B)
```

掷硬币各次是独立的。不放回抽牌则不是。

### 概率质量函数 vs 概率密度函数

离散随机变量有概率质量函数（PMF），每个结果都有可以直接读取的具体概率。

```
PMF: P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数（PDF）。单个点处的密度不是概率，概率来自对某个区间内密度的积分。

```
PDF: f(x)

P(a <= X <= b) = f(x) 从 a 到 b 的积分

f(x) 可以大于 1（密度，不是概率）
f(x) 从负无穷到正无穷的积分 = 1
```

这个区别在 ML 中很重要。分类输出是 PMF（离散选择），VAE 的潜在空间使用 PDF（连续）。

### 常见分布

**伯努利分布：** 一次试验，两个结果。用于二元分类建模。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p，  方差 = p(1-p)
```

**类别分布：** 一次试验，k 个结果。用于多分类建模（softmax 输出）。

```
P(X = i) = p_i，  其中 sum(p_i) = 1
例子：P(猫) = 0.7，  P(狗) = 0.2，  P(鸟) = 0.1
```

**均匀分布：** 所有结果等可能。用于随机初始化。

```
离散：P(X = k) = 1/n，k 在 {1, ..., n} 中
连续：f(x) = 1/(b-a)，x 在 [a, b] 中
```

**正态（高斯）分布：** 钟形曲线。由均值 (mu) 和方差 (sigma^2) 参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态：mu = 0，sigma = 1
  68% 的数据在 1 个标准差内
  95% 在 2 个标准差内
  99.7% 在 3 个标准差内
```

**泊松分布：** 固定时间段内稀有事件的计数。用于事件率建模。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda，  方差 = lambda
```

### 期望值与方差

期望值是加权平均结果。

```
离散：  E[X] = sum(x_i * P(X = x_i))
连续：  E[X] = x * f(x) 的积分
```

方差衡量围绕均值的散布程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

在 ML 中，期望值表现为损失函数（数据分布上的平均损失）。方差反映模型稳定性。梯度的高方差意味着训练噪声大。

### 联合分布与边缘分布

联合分布 P(X, Y) 同时描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 雨伞）：

| | Y=0（不带伞） | Y=1（带伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一个变量求和得到：

```
P(X = x) = 对所有 y 求和 P(X = x, Y = y)
```

上表中的行列合计就是边缘分布。

### 为什么正态分布无处不在

中心极限定理：许多独立随机变量之和（或均值）收敛到正态分布，与原始分布无关。

```
掷 1 次骰子：均匀分布（平坦）
2 次骰子的平均：三角形（有峰值）
30 次骰子的平均：几乎完美的钟形曲线

这对任何起始分布都成立。
```

这就是为什么：
- 测量误差近似正态（许多小的独立误差来源）
- 神经网络的权重初始化使用正态分布
- SGD 中的梯度噪声近似正态（许多样本梯度之和）
- 正态分布是给定均值和方差下的最大熵分布

### 对数概率

原始概率会引起数值问题。将许多小概率相乘会很快下溢到零。

```
P(句子) = P(词1) * P(词2) * ... * P(词n)
        = 0.01 * 0.003 * 0.02 * ...
        -> 0.0（约 30 个词后下溢）
```

对数概率解决了这个问题。乘法变成了加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 有限数（无下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0（因为 0 < P <= 1）
- 越负 = 越不可能
- 交叉熵损失是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将其转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) 对所有 j)

特性：
  - 所有输出在 (0, 1) 范围内
  - 所有输出之和为 1
  - 保持输入的相对顺序
  - exp() 放大 logits 之间的差异
```

Softmax 技巧：在指数化之前减去最大 logit 以防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

结果相同，无溢出。
```

Log-softmax 结合了 softmax 和 log，以保证数值稳定性。PyTorch 内部将其用于交叉熵损失。

### 采样

采样是指从分布中抽取随机值。在 ML 中：
- Dropout 随机采样哪些神经元归零
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个词
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要逆变换采样、拒绝采样或重参数化技巧（用于 VAE）等方法。

## 动手实现

### 第一步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(K | 人头牌) = {p_king_given_face:.4f}")
```

### 第二步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第三步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"骰子：E[X] = {mu:.4f}，Var(X) = {var:.4f}，SD = {var**0.5:.4f}")
```

### 第四步：从分布采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第五步：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第六步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第七步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现及所有可视化见 `code/probability.py`。

## 实际使用

借助 NumPy 和 SciPy，上述所有内容都只需一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零构建了这些。现在你知道这些库调用在做什么了。

## 练习

1. 对指数分布实现逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF 对比来验证。

2. 为两个有偏骰子构建联合分布表。计算边缘分布，检验两个骰子是否独立。

3. 计算一个 5 分类分类器的交叉熵损失，该分类器输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别是索引 3。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，接收一组对数概率，返回最可能的序列、总对数概率和等效的原始概率。用一个包含 50 个词、每个词概率为 0.01 的句子测试。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| 样本空间（Sample space）| "所有可能性" | 实验所有可能结果的集合 S |
| PMF | "概率函数" | 给出每个离散结果精确概率的函数，总和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数，对区间积分才得到概率 |
| 条件概率（Conditional probability）| "给定某事的概率" | P(A\|B) = P(A 且 B) / P(B)，贝叶斯思维的基础 |
| 独立性（Independence）| "互不影响" | P(A 且 B) = P(A) * P(B)，知道一个事件对另一个无任何影响 |
| 期望值（Expected value）| "平均值" | 所有结果的概率加权和，损失函数就是期望值 |
| 方差（Variance）| "散布程度" | 与均值偏差平方的期望，高方差 = 嘈杂、不稳定的估计 |
| 正态分布（Normal distribution）| "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))，因 CLT 而无处不在 |
| 中心极限定理（Central Limit Theorem）| "均值趋向正态" | 许多独立样本的均值收敛到正态分布，与来源分布无关 |
| 联合分布（Joint distribution）| "两个变量合在一起" | P(X, Y) 描述 X 和 Y 每种组合结果的概率 |
| 边缘分布（Marginal distribution）| "对另一个变量求和" | P(X) = sum_y P(X, Y)，从联合分布恢复单个变量的分布 |
| 对数概率（Log probability）| "概率的对数" | log P(x)，将乘积变为求和，防止长序列中的数值下溢 |
| Softmax | "将分数转为概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))，将实值 logits 映射为有效概率分布 |
| 交叉熵（Cross-entropy）| "损失函数" | -sum(p_真实 * log(p_预测))，衡量两个分布的差距，越低越好 |
| Logits | "模型原始输出" | softmax 之前的未归一化分数，得名于 logistic 函数 |
| 采样（Sampling）| "抽取随机值" | 根据概率分布生成值，模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown：什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) — 均值趋向正态的可视化证明
- [Stanford CS229 概率回顾](https://cs229.stanford.edu/section/cs229-prob.pdf) — 简明参考，涵盖本课及更多内容
- [对数求和指数技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) — 数值稳定性为何重要以及如何实现

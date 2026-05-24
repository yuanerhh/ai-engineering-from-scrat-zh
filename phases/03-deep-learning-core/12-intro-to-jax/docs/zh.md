# JAX 入门

> PyTorch 对张量进行原地修改，TensorFlow 构建计算图，JAX 编译纯函数。最后一个改变了你对深度学习的思考方式。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 03 课程 01-10，基础 NumPy
**时间：** ~90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的即时变异（eager mutation）与 JAX 的函数式编译模型之间的关键设计差异
- 应用 jit 编译和 vmap 向量化，与朴素 Python 相比加速训练循环
- 用 JAX 训练一个简单网络，并对比显式状态管理与 PyTorch 面向对象方式的差异

## 问题所在

你知道如何用 PyTorch 构建神经网络。定义一个 `nn.Module`，调用 `.backward()`，执行优化器步骤。它有效，数百万人在用。

但 PyTorch 有一个刻在其 DNA 中的限制：它在 Python 中逐个即时追踪操作。每个 `tensor + tensor` 都是一次独立的内核启动。每个训练步骤重新解释相同的 Python 代码。这在你需要在 2,048 个 TPU 上训练一个 5400 亿参数模型之前还好用。届时开销会把你拖垮。

Google DeepMind 使用 JAX 训练 Gemini。Anthropic 使用 JAX 训练 Claude。这些不是小操作——它们是地球上规模最大的神经网络训练运行。他们选择 JAX，因为它将你的训练循环视为可编译程序，而非 Python 调用序列。

JAX 是具有三大超能力的 NumPy：自动微分、JIT 编译到 XLA 以及自动向量化。你编写处理单个样本的函数，JAX 给你一个处理批次、计算梯度、编译为机器代码并在多个设备上运行的函数——所有这些都无需修改原始函数。

## 概念

### JAX 哲学

JAX 是一个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。取而代之：

| PyTorch | JAX |
|---------|-----|
| 有状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 进行 JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是风格偏好，这是编译器约束。JIT 编译需要纯函数——相同输入始终产生相同输出，无副作用。这一限制就是使 100 倍加速成为可能的原因。

### jax.numpy：熟悉的表面

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名，相同的广播规则，相同的切片语义。但数组存在于 GPU/TPU 上，每次操作都可被编译器追踪。

一个关键区别：JAX 数组是不可变的。不能 `a[0] = 5`，而是：`a = a.at[0].set(5)`。这一周会感觉别扭，然后就会豁然开朗——不可变性是使 `grad`、`jit` 和 `vmap` 可以组合的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到张量（`.grad`）。JAX 将梯度附加到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受一个函数并返回一个计算梯度的新函数。没有 `.backward()` 调用，没有存储在张量上的计算图。梯度只是另一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数、三阶导数、雅可比矩阵、黑塞矩阵——全都通过组合 `grad` 实现。PyTorch 也可以做到这些（`torch.autograd.functional.hessian`），但那是附加上去的。在 JAX 中，这是基础。

约束：`grad` 只适用于纯函数。里面不能有 print 语句（它们在追踪时运行，不是执行时）。不能改变外部状态。不能在没有显式密钥管理的情况下生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 追踪函数——它记录发生了哪些操作，但不执行它们。然后将该追踪交给 XLA（加速线性代数），Google 用于 TPU 和 GPU 的编译器。XLA 融合操作，消除冗余内存复制，并生成优化的机器代码。

后续调用完全跳过 Python。编译后的代码以 C++ 速度在加速器上运行。

JIT 有帮助的场景：
- 训练步骤（相同的计算重复数千次）
- 推理（相同的模型，不同的输入）
- 任何以类似形状输入调用超过一次的函数

JIT 有害的场景：
- 依赖值的 Python 控制流（`if x > 0`，其中 x 是追踪数组）
- 一次性计算（编译开销超过运行时）
- 调试（追踪隐藏了实际执行）

控制流限制是真实的。`jax.lax.cond` 替换 `if/else`，`jax.lax.scan` 替换 `for` 循环。这些不是可选的——它们是编译的代价。

### vmap：自动向量化

你编写处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升为处理批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 意味着：不对 `params` 进行批处理（共享），在 `x` 的轴 0 上批处理。没有手动 `for` 循环，没有重塑，没有穿线批次维度。JAX 找出批次维度并向量化整个计算。

这不是语法糖。`vmap` 生成的融合向量化代码比 Python 循环快 10-100 倍。而且它可以与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度，一行代码。在 PyTorch 中没有技巧几乎不可能实现。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 将函数复制到所有可用设备（GPU/TPU）并分割批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 跨设备同步梯度。

Google 使用 `pmap`（及其继任者 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型：编写单设备版本，用 `pmap` 包装，完成。

### Pytrees：通用数据结构

JAX 在"pytrees"上操作——列表、元组、字典和数组的嵌套组合。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 将 `f` 应用于每个叶子。这就是优化器如何一次更新所有参数：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法，没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象中：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params 被传入。没有存储，没有改变。这使每个函数都可测试、可组合、可编译。它也意味着你自己管理 params——或者使用 Flax 或 Equinox 这样的库。

### JAX 生态系统

JAX 给你原语。库给你人体工程学：

| 库 | 角色 | 风格 |
|----|------|------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 Pytree，Pythonic |
| **Optax**（DeepMind） | 优化器 + LR 调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点 | 保存/恢复 pytree |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具 |

Optax 是标准优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，使组合变得微不足道：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX vs PyTorch

| 因素 | JAX | PyTorch |
|------|-----|---------|
| TPU 支持 | 一流（Google 同时构建了两者） | 社区维护（torch_xla） |
| GPU 支持 | 好（通过 XLA 的 CUDA） | 同类最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 容易（即时，逐行） |
| 生态系统 | 研究导向（Flax, Equinox） | 庞大（HuggingFace, torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（无处不在） |
| 大规模训练 | 卓越（XLA, pmap, mesh） | 好（FSDP, DeepSpeed） |
| 原型速度 | 较慢（函数式开销） | 较快（修改后即运行） |
| 生产推理 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 使用者 | DeepMind（Gemini），Anthropic（Claude） | Meta（Llama），OpenAI（GPT），Stability AI |

诚实的答案：除非有特定原因，否则使用 PyTorch。那些原因是——TPU 访问、需要逐样本梯度、在大规模多设备上训练，或在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每次随机操作都需要一个显式的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

起初会感到烦人。但它保证了跨设备和编译的可重复性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的属性。

## 构建

### 步骤 1：设置和数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐层分别有 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤 2：初始化参数

没有类，只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 初始化，手动实现。三个 PRNG 密钥从一个种子分割。每个权重都是嵌套字典中的不可变数组。

### 步骤 3：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。params 输入，预测输出。没有 `self`，没有存储状态。`loss_fn` 从头计算交叉熵——softmax、log、负均值。

### 步骤 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次通过中返回损失值和梯度。`@jax.jit` 装饰器将两个函数都编译到 XLA。第一次调用后，每个训练步骤都不接触 Python。

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 轮，~97% 测试准确率。第一轮很慢（JIT 编译），第 2-10 轮很快。

注意缺失的内容：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合函数调用。梯度被计算、由 Adam 变换，并应用于参数——全在 `train_step` 内部。

## 实际应用

### Flax：Google 标准

Flax 是最常见的 JAX 神经网络库。它加回了 `nn.Module`，但带有显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

与 PyTorch 结构相同，但 `params` 与模型分离。`model.init()` 创建 params，`model.apply(params, x)` 运行前向传播。模型对象没有状态。

### Equinox：Pythonic 替代方案

Equinox（Patrick Kidger 开发）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一个 pytree。不需要 `.apply()`。参数就是模型的叶子。这更接近 JAX 的思维方式。

### Optax：可组合优化器

Optax 将梯度变换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——全部作为变换链组合。每个变换看到梯度，修改它，并传递给下一个。没有单一的优化器类。

## 交付物

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用很慢（编译）。基准测试前先预热。
- 避免在 JIT 内部对 JAX 数组进行 Python 循环。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部有效，普通 `print()` 无效。
- 使用 `jax.profiler` 或 TensorBoard 进行分析。XLA 编译可能隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 禁用。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课程产出：**
- `outputs/prompt-jax-optimizer.md`——选择正确 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md`——涵盖 JAX 函数式模式的技能

## 练习

1. 为 MLP 添加 dropout。在 JAX 中，dropout 需要一个 PRNG 密钥——通过前向传播穿线一个密钥并为每个 dropout 层分割它。比较有无 dropout 的测试准确率。

2. 使用 `jax.vmap` 计算一批 32 张 MNIST 图像的逐样本梯度。计算每个样本的梯度范数。哪些样本有最大的梯度，为什么？

3. 将手动前向函数替换为适用于任意层数的通用 `mlp_forward(params, x)`。使用 `jax.tree.leaves` 自动确定深度。

4. 对有无 `@jax.jit` 的训练步骤进行基准测试。计时 100 步各自的用时。在你的硬件上加速比是多少？第一次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。有无裁剪地训练。绘制训练期间的梯度范数以观察效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|---------|
| XLA | "使 JAX 快速的东西" | 加速线性代数——融合操作并从计算图生成优化 GPU/TPU 内核的编译器 |
| JIT | "即时编译" | JAX 在第一次调用时追踪函数，编译到 XLA，然后在后续调用时运行编译版本 |
| 纯函数（Pure function） | "无副作用" | 输出仅依赖于输入的函数——没有全局状态、没有修改、没有不带显式密钥的随机性 |
| vmap | "自动批处理" | 将处理单个样本的函数变换为处理批次的函数，无需重写 |
| pmap | "自动并行" | 跨多个设备复制函数并分割输入批次 |
| Pytree | "数组的嵌套字典" | JAX 可以遍历和变换的列表、元组、字典和数组的任意嵌套结构 |
| 追踪（Tracing） | "记录计算" | JAX 以抽象值执行函数以构建计算图，不计算实际结果 |
| 函数式自动微分（Functional autodiff） | "函数的梯度" | 通过变换函数而非将梯度存储附加到张量来计算导数 |
| Optax | "JAX 的优化器库" | 可组合的梯度变换库——Adam、SGD、裁剪、调度——链式组合在一起 |
| Flax | "JAX 的 nn.Module" | Google 用于 JAX 的神经网络库，在保持状态显式的同时添加层抽象 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/——官方文档，有关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs"（Bradbury et al., 2018）——解释设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/——Google 用于 JAX 的神经网络库
- Patrick Kidger，"Equinox: neural networks in JAX via callable PyTrees and filtered transformations"（2021）——Flax 的 Pythonic 替代方案
- DeepMind，"Optax: composable gradient transformation and optimisation"——标准优化器库
- "You Don't Know JAX"（Colin Raffel, 2020）——JAX 陷阱和模式的实践指南，来自 T5 的作者之一

# GPU 配置与云端

> 用 CPU 训练适合学习，真正的训练需要 GPU。

**类型：** 实践
**语言：** Python
**前置要求：** 阶段 0，第 01 课
**时间：** 约 45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 可用性
- 配置 Google Colab 使用 T4 GPU 进行免费云端实验
- 对比 CPU 与 GPU 的矩阵乘法性能，测量加速比
- 利用 fp16 经验公式估算你的显存能容纳的最大模型

## 问题

阶段 1-3 的大多数课程在 CPU 上运行没问题。但一旦你开始训练 CNN、Transformer 或 LLM（阶段 4+），就需要 GPU 加速。在 CPU 上需要 8 小时的训练，在 GPU 上只需 10 分钟。

你有三种选择：本地 GPU、云端 GPU，或 Google Colab（免费）。

## 概念

```
你的选择：

1. 本地 NVIDIA GPU
   费用：$0（你已有）
   配置：安装 CUDA + cuDNN
   适合：日常使用、大型数据集

2. Google Colab（免费层）
   费用：$0
   配置：无需配置
   适合：快速实验、家里没有 GPU

3. 云端 GPU（Lambda、RunPod、Vast.ai）
   费用：$0.20-2.00/小时
   配置：SSH + 安装
   适合：正式训练、大型模型
```

## 动手实现

### 选项 1：本地 NVIDIA GPU

检查是否有 GPU：

```bash
nvidia-smi
```

安装带 CUDA 支持的 PyTorch：

```python
import torch

print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 运行时 > 更改运行时类型 > T4 GPU
3. 运行 `!nvidia-smi` 验证

将本课程的 Notebook 直接上传到 Colab 运行。

### 选项 3：云端 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没关系。

大多数课程在 CPU 上可以运行。需要 GPU 的课程会明确说明并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用: {device}")
```

## 动手实现：GPU vs CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"加速比: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，比较 CPU 与 GPU 的时间
2. 如果没有 GPU，在 Google Colab 上运行并对比
3. 查看你的显存容量，估算能容纳的最大模型（经验公式：fp16 每个参数占 2 字节）

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| CUDA | "GPU 编程" | NVIDIA 的并行计算平台，让你在 GPU 上运行代码 |
| VRAM | "GPU 内存" | GPU 上的显存，与系统内存独立。限制模型大小。|
| fp16 | "半精度" | 16 位浮点数，使用 fp32 一半的内存，精度损失极小 |
| Tensor Core | "快速矩阵硬件" | GPU 上专门用于矩阵乘法的核心，比普通核心快 4-8 倍 |

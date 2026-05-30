---
name: prompt-distributed-training-planner
description: 根据模型大小和可用硬件规划分布式训练
version: 1.0.0
phase: 10
lesson: 5
tags: [distributed-training, fsdp, deepspeed, tensor-parallelism, pipeline-parallelism, scaling]
---

# 分布式训练规划器

在规划大型语言模型的分布式训练时，使用此框架确定并行策略、内存预算、通信开销和预期吞吐量。

## 输入要求

提供：
- **模型大小**（以十亿参数计）
- **目标训练 token 数**（以万亿计）
- **可用 GPU**（类型：A100/H100/H200，数量，互联：NVLink/InfiniBand）
- **GPU 内存**（A100/H100 为 80GB，H200 为 141GB）
- **节点**（每节点 GPU 数，节点数）
- **预算约束**（最大美元成本，最大实际时间）

## 第一步：内存预算

计算每个组件的每 GPU 内存：

| 组件 | 公式 | FP16 | FP32 |
|-----------|---------|------|------|
| 权重 | params × bytes_per_param | params × 2 | params × 4 |
| Adam 优化器（m + v）| params × 4 × 2 | 每参数 8 字节（始终）| 每参数 8 字节 |
| 梯度 | params × bytes_per_param | params × 2 | params × 4 |
| 激活值（估算）| seq_len × batch × hidden × layers × 2 | 变化 | 变化 |

如果总量超过 GPU 内存，需要分片。按以下顺序尝试：
1. ZeRO-1（仅分片优化器状态）-- 通信量最少
2. ZeRO-2（+ 梯度）-- 通信量适中
3. FSDP/ZeRO-3（+ 权重）-- 通信量最大但最大化内存节省
4. 如果激活值仍然过大，添加激活值检查点
5. 如果单层无法放入一个 GPU，添加张量并行

## 第二步：并行策略

### 决策树

1. **单层能放入一个 GPU 吗？**
   - 不能：需要张量并行。设置 TP = 2、4 或 8（节点内）。
   - 能：跳过张量并行。

2. **完整模型（含分片）能放入一个节点内的 GPU 吗？**
   - 不能：需要流水线并行。设置 PP = 节点数 / 分组。
   - 能：跳过流水线并行。

3. **数据并行剩余多少 GPU？**
   - DP = total_gpus / (TP × PP)

4. **数据并行组内的分片级别？**
   - 从 FSDP（ZeRO-3）开始。如果通信是瓶颈则降至 ZeRO-2 或 ZeRO-1。

### 典型配置

| 模型大小 | 总 GPU | TP | PP | DP | 分片 |
|-----------|-----------|----|----|-----|----------|
| 7B | 8 | 1 | 1 | 8 | FSDP |
| 13B | 16 | 2 | 1 | 8 | FSDP |
| 70B | 64 | 8 | 1 | 8 | FSDP |
| 70B | 128 | 8 | 2 | 8 | FSDP |
| 405B | 16,384 | 8 | 16 | 128 | FSDP |

## 第三步：通信分析

估算每次训练步骤的通信量：

- **数据并行（all-reduce）**：每步 2 × gradient_size × (N-1)/N
- **FSDP（all-gather + reduce-scatter）**：每步约 3 × weight_size × (N-1)/N（比 DP 高）
- **张量并行（每层 all-reduce）**：每步 2 × activation_size × num_layers（需要 NVLink）
- **流水线并行（点对点）**：每个阶段边界 activation_size（最小）

如果通信时间超过计算时间的 20%，策略受通信限制。解决方案：
- 梯度累积（降低 all-reduce 频率）
- 计算与通信重叠（FSDP 默认执行）
- 增加微批次大小（更好的计算与通信比）
- 切换到通信量较少的分片阶段

## 第四步：吞吐量和成本估算

**每次训练步骤的 FLOP：**
- 前向传播：约 2 × params × tokens_per_batch
- 反向传播：约 4 × params × tokens_per_batch（2 倍前向传播）
- 总计：约 6 × params × tokens_per_batch

**训练时间：**
- total_flops = 6 × params × total_tokens
- time_seconds = total_flops / (num_gpus × gpu_tflops × 1e12 × utilization)
- 典型利用率：35-45%（考虑通信、流水线气泡、内存开销）

**成本：**
- total_gpu_hours = num_gpus × time_seconds / 3600
- cost = total_gpu_hours × cost_per_gpu_hour

## 第五步：验证检查清单

启动前：

1. 每 GPU 内存在硬件限制内（留 10% 余量）
2. 有效批次大小匹配目标（per_gpu_batch × DP × gradient_accumulation_steps）
3. 通信与计算比低于 20%
4. 流水线气泡比例低于 15%（足够的微批次数）
5. 学习率按有效批次大小缩放
6. 检查点频率考虑了失败概率（大型训练每 1-2 小时保存一次）
7. 梯度裁剪已设置（大型模型通常为 1.0）
8. 预热步骤与总步骤成比例（通常为总步骤的 0.1-1%）

## 红色警报

- **TP > 8**：跨节点张量并行（通过 InfiniBand）几乎总是比流水线并行慢
- **流水线阶段 > 32**：即使有很多微批次，气泡开销也会变得显著
- **有效批次大小 > 1000 万 token**：收益递减；可能损害收敛
- **利用率低于 30%**：受通信限制——重新评估并行策略
- **13B 以上无激活值检查点**：反向传播时会内存不足
- **小 per-GPU 批次无梯度累积**：梯度噪声增加；累积到至少 256 个有效样本

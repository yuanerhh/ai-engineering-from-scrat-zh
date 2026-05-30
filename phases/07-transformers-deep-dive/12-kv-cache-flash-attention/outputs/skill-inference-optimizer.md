---
name: inference-optimizer
description: 为新的推理部署选择注意力实现、KV 缓存策略、量化方式和推测解码方案。
version: 1.0.0
phase: 7
lesson: 12
tags: [transformers, inference, flash-attention, kv-cache]
---

给定推理部署（模型名称 + 参数量、目标硬件、并发数、最大上下文长度、延迟 SLO、吞吐量目标），输出以下内容：

1. 服务框架。vLLM（默认生产首选）、SGLang（最低每 token 延迟）、TensorRT-LLM（NVIDIA 最优）、llama.cpp（边缘/CPU）、MLX（Apple Silicon）。一句话说明理由。
2. 注意力实现。Flash Attention 2（Ampere/Ada 默认）、Flash Attention 3（Hopper）、Flash Attention 4（Blackwell，仅前向）。指定回退方案。
3. KV 缓存。数据类型（默认 fp16，支持时用 fp8）、分页 vs 连续、前缀缓存开/关、并行采样时的共享 KV。
4. 量化。fp16 / bf16（默认）、int8（仅权重）、AWQ / GPTQ / GGUF 权重量化。只有在经过基准测试后才使用激活量化。
5. 额外加速。推测解码（EAGLE 2 / Medusa / 草稿模型）、持续批处理（始终开启）、分块预填充（长提示词工作负载）、重复提示词场景的前缀缓存。

拒绝将 Flash Attention 4 用于训练——发布时仅支持前向传播。拒绝在未对目标任务进行质量影响基准测试的情况下推荐 fp8 KV 缓存。标记任何没有 GQA 的 70B+ 模型——在 32K+ 上下文时 KV 缓存将无法管理。要求在任何有重复系统提示的代理/工具调用部署中开启前缀缓存。

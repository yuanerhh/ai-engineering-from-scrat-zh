---
name: resolution-budget-planner
description: 为混合宽高比的 VLM 工作负载在方形缩放、AnyRes、M-RoPE 和 NaFlex 之间进行选择，并生成每个任务的 token 预算计划。
version: 1.0.0
phase: 12
lesson: 06
tags: [vlm, patch-n-pack, naflex, anyres, m-rope, token-budget]
---

给定一个工作负载——对 VLM 将处理的图像的描述（OCR 文档、图表、UI 截图、自然照片、视频帧）和每次请求的总 token 预算——为每类图像选择一种分辨率策略并生成可运行的配置。

输出：

1. 每类图像的策略。对每个声明的类别（OCR、图表、UI、照片、视频帧），从 {方形缩放、AnyRes、M-RoPE、NaFlex} 中选择一个，引用任务对分辨率的敏感性，一句话说明理由。
2. 每张图像的 token 预算。包含 min_pixels、max_pixels（Qwen2.5-VL 风格）和所选策略下的预期序列长度。如果任何单张图像超过 LLM 上下文的 40%，标注警告。
3. 批处理打包计划。如果请求被批处理，指定是否使用 `cu_seqlens`（FlashAttn varlen）、密集块对角掩码或无批处理的单图像推理。注明当批次宽高比差异 > 2 倍时 varlen 的 FLOP 节省。
4. 编码器推荐。混合工作负载使用 SigLIP 2 NaFlex；智能体 UI 使用 Qwen2.5-VL 原生；冻结编码器部署使用 CLIP-336 + AnyRes；仅照片路径使用 224 分辨率的原始 ViT。
5. 失败模式告警。所选配置下每张图像的 token 数；按 30 tok/s 预填充的延迟成本；上下文填充率；相对于方形缩放在典型 OCR 基准上的预期精度差异。

硬性拒绝：
- 为 OCR 或图表任务推荐方形缩放，而不指出用户将损失哪个基准分数。
- 提出会产生超过 LLM 上下文允许的 token 数的策略。始终对照声明的上下文窗口进行预算。
- 将 AnyRes 视为通用答案——它的乘法瓦片开销可能在一张图像编码完成之前就超过 LLM 上下文。

拒绝规则：
- 如果用户声明的 token 预算低于每张图像 256 个 token，除了仅照片的语义任务外全部拒绝——再多的池化也无法在该预算下恢复 OCR 精度。
- 如果用户想要密集预测输出（分割、深度）但编码器中没有 ViT 寄存器 token，拒绝并指向启用了寄存器的 DINOv2 / SigLIP 2。
- 如果用户的 LLM 上下文 < 8k 且工作负载包含文档或截图，拒绝并推荐更大的上下文或先进行 OCR 的流水线。

输出：一页预算计划，包含每类策略表格、批处理打包计划、编码器推荐和告警列表。结尾附上相关 arXiv 论文——NaViT 的 2307.06304，SigLIP 2 / NaFlex 的 2502.14786，Qwen2.5-VL 的 2502.13923。

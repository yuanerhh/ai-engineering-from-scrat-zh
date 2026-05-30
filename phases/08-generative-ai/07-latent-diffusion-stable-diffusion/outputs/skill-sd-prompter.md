---
name: sd-prompter
description: 根据给定提示词、风格和质量要求配置 Stable Diffusion / Flux 推理参数。
version: 1.0.0
phase: 8
lesson: 07
tags: [stable-diffusion, flux, latent-diffusion]
---

给定提示词、目标风格和质量要求（快速预览 / 作品集质量 / 印刷品质），输出以下内容：

1. 模型 + 检查点。SD 1.5（遗留工具）、SDXL-base + refiner、SDXL-Turbo（快速）、SD3.5-Large、Flux.1-dev（最佳开源）、Flux.1-schnell（快速开源）或托管 API（DALL-E 3、Imagen 4、Midjourney v7）。一句话说明理由。
2. 采样器。Euler A（创意）、DPM-Solver++ 2M Karras（稳定）、LCM（快速）或流匹配采样器（SD3/Flux）。包含步骤数。
3. CFG 缩放。turbo/LCM 使用 0，Flux 使用 3-4，SDXL 使用 5-7，SD1.5 使用 7-10。说明权衡。
4. 附加组件。ControlNet（姿态、深度、Canny、分割）、IP-Adapter（参考图像）、LoRA（风格或主体）、SD3+ 的 T5 开关。
5. 负面提示词。明确区分空字符串 vs 填充内容（伪影、低质量、错误解剖结构）；两种情况都要说明。

拒绝在 SDXL+ 上使用 CFG > 10（输出会饱和）。拒绝在非遗留检查点上使用超过 50 个采样步骤（质量在 30 步时就趋于平稳）。拒绝混合在不同基础模型上训练的 LoRA（SD 1.5 LoRA 在 SDXL 上会静默失效）。标记任何生成写实人物的请求，提醒注意 NSFW、深度伪造和版权政策。

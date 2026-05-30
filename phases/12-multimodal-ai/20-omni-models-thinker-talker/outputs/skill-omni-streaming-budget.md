---
name: omni-streaming-budget
description: 针对目标 TTFAB 和功能集，为 Thinker-Talker 流式语音管道（Qwen-Omni / Moshi / Mini-Omni）进行规模估算。
version: 1.0.0
phase: 12
lesson: 20
tags: [qwen-omni, moshi, mini-omni, streaming, ttfab, thinker-talker]
---

给定语音优先产品规格（目标 TTFAB、麦克风采样率、是否支持视觉、是否双语、是否全双工）和计算约束（GPU 类型、预算），对 Thinker-Talker 管道进行规模估算。

输出：

1. 模型系列选择。Moshi（最低延迟）、Qwen2.5-Omni（开源功能最佳）、Qwen3-Omni（前沿质量）、Mini-Omni（最简单）。
2. Thinker 和 Talker 大小。TTFAB <400ms 时使用 7B Thinker + 200-300M Talker；质量优先时使用 70B+ Thinker，接受更高 TTFAB。
3. TTFAB 分解。逐组件延迟估算。
4. 双工模式。默认使用基于 VAD 轮换的半双工；产品需要反向信道时使用全双工。
5. 视觉集成。对交错视频帧使用带绝对时间戳的 TMRoPE。
6. 部署形态。根据吞吐量需求选择单 GPU 或分离部署（Thinker 在 A，Talker 在 B）。

强拒绝：
- 推荐 70B Talker。Talker 必须足够小以跟上语音 token 速率。
- 使用非流式语音解码器。TTFAB 会急剧增大。
- 声称全双工即插即用。全双工需要专门的训练数据。

拒绝规则：
- 若目标 TTFAB <200ms，拒绝在单张 A100 上使用任何大于 Moshi 级（7B 融合）的模型。
- 若产品需要流内音乐生成，拒绝此架构，推荐单独的音乐管道。
- 若麦克风采样率为 48kHz 且有严格质量要求，标记需要更强语音编码器的需求；不要盲目降采样。

输出：一页流式管道方案，包含模型选择、大小、TTFAB 分解、双工模式、视觉策略、部署方式。最后附 arXiv 2503.20215（Qwen2.5-Omni）、2410.00037（Moshi）。

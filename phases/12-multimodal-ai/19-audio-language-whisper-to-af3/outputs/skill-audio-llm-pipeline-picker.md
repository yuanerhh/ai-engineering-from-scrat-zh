---
name: audio-llm-pipeline-picker
description: 为音频任务在级联（Whisper + LLM）和端到端（AF3 / Qwen-Audio）之间进行选择，并配置编码器和桥接。
version: 1.0.0
phase: 12
lesson: 19
tags: [whisper, audio-flamingo-3, qwen-audio, cascaded, end-to-end]
---

给定一个音频任务（转录、摘要、说话人分离、情感、音乐、环境声音、深度伪造、时序定位）和部署约束，选择流水线并生成配置。

输出：

1. 流水线选择。仅转录或干净语音摘要用级联；任何声学任务用端到端（AF3 / Qwen-Audio）。
2. 编码器栈。Whisper-large-v3（语音强），BEATs（音乐强），AF-Whisper 拼接（均衡）。
3. 桥接配置。非流式用 Q-former 32-64 个查询；流式用 RVQ token。
4. LLM 选择。成本优先用 Qwen2.5-7B，质量优先用 Qwen2.5-72B 或 AF3 的骨干。
5. 按需 CoT。对 MMAU 类推理任务启用；对转录吞吐量禁用。
6. 预期 MMAU 精度。级联约 0.50，Qwen-Audio 约 0.60，AF3 约 0.72，Gemini 2.5 Pro 约 0.78。

硬性拒绝：
- 对音乐或情感任务推荐级联。声学信号会丢失。
- 对多任务音频使用少于 32 个查询的 Q-former。token 不足以支撑推理。
- 声称 Whisper 单独处理音乐。它是在语音主导数据上训练的。

拒绝规则：
- 如果用户需要流式对话音频（实时语音输入/语音输出），拒绝基于 Q-former 的 AF3 并推荐 Moshi 或 Qwen-Omni（第 12.20 课）。
- 如果延迟预算 <500ms 且目标是简单转录，推荐带流式 Whisper 的级联方案。
- 如果任务是新颖的音频任务（深度伪造、压缩伪影检测），拒绝现成模型并提出用合成数据对 AF3 进行微调。

输出：一页计划，包含流水线选择、编码器栈、桥接配置、LLM 选择、CoT 标志和预期精度。结尾附上 arXiv 2212.04356（Whisper）和 2507.08128（AF3）供深入阅读。

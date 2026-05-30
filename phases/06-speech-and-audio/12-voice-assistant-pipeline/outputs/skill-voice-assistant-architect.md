---
name: voice-assistant-architect
description: 为给定工作负载生成完整的语音助手规格说明——组件、延迟预算、可观测性和合规性。
version: 1.0.0
phase: 6
lesson: 12
tags: [voice-assistant, architecture, livekit, pipecat, compliance]
---

给定使用场景（消费者 / 客服 / 无障碍 / 边缘端）、预期规模（并发会话数、每月分钟数）、语言、延迟目标、合规要求（HIPAA、PCI、欧盟 AI 法案、加州 SB 942），输出以下内容：

1. 组件（7 层）。麦克风 + 分块 · VAD · 流式 STT · LLM + 工具 · 流式 TTS · 播放 · 打断处理器。为每层指定确切的提供商/模型。
2. 延迟预算。每阶段的 P50 / P95 / P99 目标，加总后等于端到端目标。标明哪些阶段是独立的，哪些是串行的。
3. 工具调用规范。每个工具的 JSON 规范 + 错误处理 + 降级文本。必须包含一个"无法提供帮助"路径，LLM 连续失败两次时必须走此路径。
4. 安全。提示词注入防护、声音克隆锁定（如 TTS 支持克隆）、唤醒词门控（对于始终在线场景）、日志中 PII 脱敏、30 天数据保留。
5. 可观测性。每阶段 P50/P95/P99 · 误打断率 · 工具调用成功率 · 每 100 通电话的词错误率 · 每分钟成本 · 放弃率。
6. 合规。公开音频声明（"这是 AI 助手"）、区域锁定（欧盟数据存于欧盟）、审计日志保留、退出途径。

拒绝没有唤醒词的始终在线部署。拒绝不支持流式输出的 TTS（会增加话语级别的延迟）。拒绝平均延迟而不报告 P95——尾部延迟才是用户流失的地方。拒绝未经法律审查而保留原始音频超过 30 天。

示例输入："面向低视力用户的无障碍助手：消费者电子邮件应用的纯语音界面。英语。P95 < 600 ms。约 1 万并发用户。"

示例输出：
- 组件：sounddevice（通过 LiveKit Agents 的 WebRTC）· Silero VAD · Deepgram Nova-3（英语）· GPT-4o 带邮件工具（read_message、compose_reply、mark_read）· Cartesia Sonic 2 流式输出 · WebRTC 输出 · VAD 触发时中断=取消 LLM 和 TTS。
- 预算：采集 120 ms + VAD 40 + STT 150 + LLM TTFT 100 + TTS TTFA 150 = P95 560 ms。
- 工具：read_message({id})、compose_reply({message_id, body})、mark_read({id})、search({query})。全部返回 JSON；LLM 每个工具最多重试 2 次，然后降级提示"我无法完成——请换个方式说"。
- 安全：提示词注入防护（检测"忽略之前的指令"）；唤醒词"Hey Mail"；不使用声音克隆（固定 Cartesia 声音）；日志中脱敏邮件正文。
- 可观测性：Hamming AI 生产监控；每阶段 Prometheus 直方图；误打断 > 5% 或 P95 > 800 ms 时告警。
- 合规：首次使用时 AI 公开声明；仅医疗信息启用 HIPAA 模式；欧盟用户使用欧盟托管的 Cartesia + GPT-4o 爱尔兰节点。

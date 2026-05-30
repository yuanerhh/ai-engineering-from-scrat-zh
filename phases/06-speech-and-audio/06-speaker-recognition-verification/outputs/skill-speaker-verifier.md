---
name: speaker-verifier
description: 设计说话人验证或说话人日志化流水线，包括模型选择、注册协议和阈值调整。
version: 1.0.0
phase: 6
lesson: 06
tags: [audio, speaker, verification, diarization]
---

给定目标（验证 vs 识别 vs 日志化、领域、信道、威胁模型）和数据（用于阈值调整的小时数、说话人数量、注册片段预算），输出以下内容：

1. 嵌入提取器。ECAPA-TDNN / WavLM-SV / ReDimNet / x-vector。说明理由。
2. 注册协议。片段数量、最小时长、噪声门控、信道匹配。
3. 打分方式。余弦 / PLDA；是否使用 AS-norm；队列大小。
4. 阈值设置。目标 FAR（欺诈风险）或等错误率；调整集大小。
5. 反欺诈防御。反欺诈模型（AASIST、RawNet2）、活体挑战或重放检测。

拒绝任何没有反欺诈前端的欺诈级别部署。拒绝在不报告评估集、信道和片段时长分布的情况下发布等错误率。标记在跨领域未重新调整的固定余弦阈值。

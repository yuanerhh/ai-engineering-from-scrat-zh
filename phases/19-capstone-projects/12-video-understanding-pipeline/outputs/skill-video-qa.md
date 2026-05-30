---
name: video-qa
description: 构建一个视频理解管道，具备场景分割、多向量索引、时序定位和带时间戳的引用。
version: 1.0.0
phase: 19
lesson: 12
tags: [capstone, video, multimodal, gemini, qwen-vl, molmo, transnet, qdrant]
---

给定 100 小时视频，构建一个摄取管道和查询系统，以自然语言问题配合 (start, end) 时间戳及帧预览进行回答。

构建计划：

1. 摄取视频（YouTube URL 或 MP4）；如有需要降采样至 720p。
2. 使用 TransNetV2 或 PySceneDetect 进行场景分割；输出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。
3. 使用 Whisper-v3-turbo（faster-whisper）进行 ASR，生成词级时间戳；按场景切片。
4. 使用 Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2 进行 VLM 描述；输出描述文本 + 帧嵌入。
5. Qdrant 多向量索引，每个场景包含三个命名向量（caption_emb、frame_emb、transcript_emb），载荷为 {video_id, scene_id, start_ms, end_ms, keyframe_url}。
6. 查询：三路并行密集查询；倒数排名融合合并；top-k=5 个场景。
7. 时序定位（TimeLens adapter 或 VideoITG）在最优场景内精确 (start, end)。
8. VLM 合成（Gemini 2.5 Pro），输入查询 + 前 3 个场景片段 + 转录；要求引用 `(video_id, start_ms, end_ms)`。
9. 在 ActivityNet-QA、NeXT-GQA 以及 100 个手工标注的自定义问题集上进行评估。按问题类别（描述性、计数、动作类型）分别报告总体和细分准确率。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 时序定位 IoU | 在保留定位集上的 IoU |
| 20 | 问答准确率 | NeXT-GQA 和 100 个自定义问题集 |
| 20 | 摄取吞吐量 | 每美元索引的视频小时数 |
| 20 | UI 和引用用户体验 | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 计数和动作类型准确率单独报告 |

硬性拒绝条件：

- 每个场景仅使用单一向量的管道。多向量是展示类别区分所必需的。
- 没有 (start, end) 引用的答案。
- 只报告总体准确率而不提供计数/动作子集细分。
- VLM 合成未直接接收场景帧（仅文本输入会失去视觉定位能力）。

拒绝规则：

- 拒绝处理许可证来源不明的视频；每个 video_id 必须有许可证标签。
- 拒绝在测量吞吐量以上的摄取速率下声称"实时"响应。
- 拒绝将计数/动作幻觉率隐藏在总体准确率中。

输出：一个包含场景分割 + ASR + 描述管道、多向量 Qdrant 集合、时序定位适配器、带时间戳深层链接的 Next.js 15 查看器、三项基准评估结果（ActivityNet-QA、NeXT-GQA、自定义）的代码库，以及一份说明你观察到的三种计数或动作类型失败类别以及减少每种失败的检索或合成变更的报告。

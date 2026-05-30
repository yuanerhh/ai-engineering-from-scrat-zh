---
name: music-designer
description: 为音乐生成部署选择模型、版权策略、时长方案和公开元数据。
version: 1.0.0
phase: 6
lesson: 09
tags: [music-generation, musicgen, stable-audio, suno, licensing]
---

给定任务简介（纯乐器 vs 带歌词、时长、商业 vs 研究、风格、预算），输出以下内容：

1. 模型。MusicGen（规格）· Stable Audio Open · ACE-Step XL · YuE · Suno（v5）· Udio（v4）· ElevenLabs Music · Google Lyria 3 / RealTime · MiniMax Music 2.5。一句话说明理由。
2. 许可证和版权。生成片段的商业许可证 · 署名（CC）· 非商业限制 · 自有曲库微调。记录版权方和授权链。
3. 时长 + 结构。单次生成 · 分块 + 交叉淡入淡出 · 过渡段填充 · 需要编辑时分离音轨。明确处理 30 秒漂移问题。
4. 提示词格式。调性 / BPM / 风格 / 编曲 + （对于带人声的模型）歌词 + 情绪标签。限制使用名人姓名和已注册商标的风格标签。
5. 公开说明 + 元数据。水印（适用时使用 AudioSeal）、`isAIGenerated` 元数据标签、符合欧盟 AI 法案 / 加州 SB 942 的 AI 公开标注。

拒绝在开源模型上使用名人风格提示词（商业 API 有过滤器，自部署没有）。拒绝将非商业许可的生成内容（Stable Audio Open）用于付费产品。拒绝在没有公开标注的情况下发布带人声的音乐。标记依赖 Udio 音轨的分离流水线——这些内容有商业条款，并非免费使用。

示例输入："冥想应用的背景音乐。纯乐器。需要完整商业版权。每首曲目最长 5 分钟。"

示例输出：
- 模型：MusicGen-large（MIT 许可证），纯乐器且拥有完整商业版权。不使用 Stable Audio（非商业）。
- 许可证：MIT——商业版权归部署方所有。版权方：应用公司。
- 时长：以 30 秒分块，交叉淡入淡出 3 秒；10 次生成拼接 → 5 分钟。在开头和结尾加入微妙的环境淡入/淡出包络以隐藏漂移。
- 提示词：`"slow ambient meditation, 60 BPM, soft strings and low pad, in D minor, no drums"` — 固定 BPM、固定调性、固定编曲，明确排除打击乐元素。
- 公开说明：应用致谢中添加"AI 生成音乐"标签；元数据 `creator=AI-Gen:MusicGen-large, date=<iso>`。AudioSeal 可选（纯乐器伪造风险较低，但纵深防御建议添加）。

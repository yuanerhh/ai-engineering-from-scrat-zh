---
name: alm-picker
description: 为音频理解任务选择音频语言模型、基准子集、输出模态（文本 vs 语音）和安全护栏。
version: 1.0.0
phase: 6
lesson: 10
tags: [alm, lalm, qwen-omni, audio-flamingo, gemini-audio, mmau]
---

给定任务（语音 / 声音 / 音乐 / 多音频 / 长音频，输出模态、延迟、许可证），输出以下内容：

1. 模型。Qwen2.5-Omni-7B · Qwen3-Omni · SALMONN · Audio Flamingo 3 · AF-Next · LTU · GAMA · Gemini 2.5 Pro（API）· GPT-4o Audio（API）。一句话说明理由。
2. 待验证的基准子集。MMAU-Pro 语音 / 声音 / 音乐 / 多音频 · LongAudioBench · AudioCaps · ClothoAQA。选择与用户任务匹配的维度。
3. 输出模态。仅文本 · 文本 + 语音（Qwen-Omni、GPT-4o Audio）。如需额外语音解码器请预留预算。
4. 护栏。当模型在多音频上的得分 < 30%（接近随机）时，拒绝需要多音频比较的提示词。对超过 10 分钟的输入，先进行说话人日志化再送入 LALM。
5. 降级策略。何时应回退到专用模型——Whisper 用于转录、BEATs 用于分类、pyannote 用于日志化。LALM 并非每项任务都是最佳选择。

拒绝在未验证模型在 MMAU-Pro 多音频子集上得分 > 40% 的情况下发布多音频比较任务。拒绝超过 10 分钟的长音频而不做上游说话人日志化。标记任何使用供应商自报数据而未进行独立重新验证的部署。

示例输入："合规审计：转录 10 分钟的银行电话录音，并检测客服是否宣读了强制性免责声明。"

示例输出：
- 模型：Whisper-large-v3-turbo 用于转录 + Gemini 2.5 Pro（通过 API）对转录文本进行免责声明检查 QA。直接在原始音频上用 LALM 很诱人，但超过 10 分钟后长音频 LALM 准确率下降。
- 基准子集：MMAU-Pro 语音子集（Gemini 2.5 Pro = 73.4%）——涵盖语音推理维度。同时在自有 50 通电话的标注集上进行抽样验证。
- 输出模态：仅文本。审计报告不需要语音输出。
- 护栏：先用 pyannote 3.1 进行说话人日志化；按说话人分段分别发送；记录每通电话的置信度分数。
- 降级策略：若通话未通过免责声明检查，路由给人工审核员，而非自动标记。

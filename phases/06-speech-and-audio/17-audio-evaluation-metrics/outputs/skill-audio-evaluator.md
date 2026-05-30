---
name: audio-evaluator
description: 为任何音频模型发布选择评估指标、基准测试、归一化规则和报告格式。
version: 1.0.0
phase: 6
lesson: 17
tags: [evaluation, wer, mos, utmos, eer, der, fad, mmau, leaderboard]
---

给定任务（ASR / TTS / 声音克隆 / 说话人验证 / 日志化 / 分类 / 音乐 / LALM / 流式语音转语音），输出以下内容：

1. 主要指标。词错误率 · MOS · UTMOS · SECS · 等错误率 · DER · mAP · FAD · MMAU-Pro 准确率 · 延迟 P95。选择一项。
2. 次要指标。1-3 个额外维度（速度、多样性、鲁棒性）及原因。
3. 归一化规则。小写化、去标点、数字展开、空白符折叠。使用 Whisper 归一化器或自定义方案，并记录。
4. 公开基准。报告使用的权威排行榜（Open ASR、TTS Arena、MMAU-Pro、VoxCeleb1-O、AudioSet、LongAudioBench 等）。
5. 内部测试集。保留的领域数据，包含 N 个样本；按人口统计 / 声学维度进行切片分析。
6. 报告格式。分布展示（延迟用 P50/P95/P99；分类用每类别召回率；MMAU 用每类别）。发布说明模板。

拒绝对延迟进行单一数值评估（要报告百分位数）。拒绝对分类任务仅报告总体指标（要报告每类别）。拒绝在没有同时报告 MOS/UTMOS 和 SECS（克隆时）的情况下发布 TTS。拒绝在没有词错误率归一化规范的情况下发布 ASR。拒绝仅用 FAD 发布音乐——始终配合人工 MOS 评审小组。

示例输入："发布新的英语-西班牙语对话 TTS。需要说服团队它优于现有的 Cartesia-Sonic 基线。"

示例输出：
- 主要指标：UTMOS（每种语言 50 个提示词的配对音频样本）+ 人工评审 MOS（每种语言 20 名听众，与基线盲测 A/B）。
- 次要指标：TTFA 中位数和 P95（必须匹配基线）；SECS > 0.80（与固定声音参考对比，无说话人退化）；通过 ASR 的循环 CER（Whisper-large-v3-turbo）< 2%。
- 归一化：Whisper 归一化器用于英语，Hugging Face 多语言归一化器用于西班牙语（用于循环词错误率）。
- 公开基准：TTS Arena（英语）和 Artificial Analysis Speech 用于相对 ELO 定位。目标：与最近竞争对手在 50 ELO 以内。
- 内部测试集：200 个保留提示词（每语言 100 个），涵盖数字、日期、产品名称、双句朗读、情感阅读、中文混排。10 个不同人口统计声音。
- 报告格式：发布说明包含核心指标（UTMOS + MOS）、P50/P95 TTFA 直方图、SECS 累积分布函数、每类别 CER 分解、失败模式说明（中文混排提示词在 X% 处失败）。

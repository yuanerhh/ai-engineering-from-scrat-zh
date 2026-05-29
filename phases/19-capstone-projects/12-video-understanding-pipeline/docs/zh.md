# 顶点项目 12 — 视频理解流水线（场景分析、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 交付了视频 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 长上下文原生处理数小时的视频。TimeLens-100K 在规模上定义了时间定位。2026 年的流水线已趋于稳定：场景分割，每场景字幕 + 嵌入，字幕对齐，多向量索引，以及返回带时间戳和帧预览的（start, end）答案的查询。这个顶点项目要求摄入 100 小时视频，达到公开基准，并对计数和动作类问题的幻觉率进行测量。

**类型：** 顶点项目  
**语言：** Python（流水线），TypeScript（UI）  
**前置条件：** 第 4 阶段（计算机视觉），第 6 阶段（语音），第 7 阶段（Transformer），第 11 阶段（LLM 工程），第 12 阶段（多模态），第 17 阶段（基础设施）  
**涵盖阶段：** P4 · P6 · P7 · P11 · P12 · P17  
**预计时间：** 30 小时

## 问题背景

长视频问答是 2026 年规模下最耗带宽的多模态问题。Gemini 2.5 Pro 可以原生读取一段 2 小时的视频，但将 100 小时视频摄入一个可查询的语料库仍然需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、使用 VLM 进行每场景字幕生成（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、字幕对齐（带词级时间戳的 Whisper-v3-turbo），以及并排存储字幕、帧嵌入和字幕的多向量索引。查询流水线返回带时间戳和帧预览的（start, end）答案。

基准测试是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的 100 个问题自定义集。对计数和动作类问题的幻觉是已知的难点失败类别；这个顶点项目明确对此进行测量。

## 核心概念

摄入时三条流水线并行运行。**场景分割**将视频切割成场景。**VLM 字幕生成**为每个场景生成字幕，并从关键帧生成帧嵌入。**ASR 对齐**生成词级时间戳。三个流以（scene_id, 时间范围）连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：字幕嵌入、帧嵌入、字幕（转录文本）嵌入。

查询时，自然语言问题对三种向量都发起查询；结果用 RRF 合并；时间定位适配器（TimeLens 风格）在 top-k 场景内细化（start, end）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + top-k 场景 + 裁剪帧，返回带引用时间戳和帧预览的答案。

幻觉测量很重要。计数（"有多少人进入房间？"）和动作类（"厨师是先倒再搅拌吗？"）问题出了名地不可靠。单独报告这些问题的精度，与描述性问题区分开来。

## 架构图

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年最先进）或 PySceneDetect
- ASR：faster-whisper Whisper-v3-turbo（带词级时间戳）
- VLM 字幕生成器 + 答案生成器：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：Qdrant 多向量支持（字幕/帧/字幕）
- UI：Next.js 15 带 HTML5 视频播放器和场景缩略图
- 评测：ActivityNet-QA、NeXT-GQA、自定义 100 个问题手工标注集
- 幻觉基准：带手工标签的计数和动作类子集

## 构建步骤

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。必要时缩小到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 个场景。

3. **ASR 处理。** 对音频运行 Whisper-v3-turbo；导出词级时间戳；切分为每场景字幕切片。

4. **VLM 字幕生成。** 对每个场景，用关键帧和简短字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成字幕 + 帧嵌入。

5. **多向量索引。** Qdrant 集合，带三个命名向量。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三次稠密查询；用倒数排名融合合并；top-k=5 个场景。

7. **时间定位。** 在 top 场景上运行 TimeLens 风格适配器，在场景内细化（start, end）窗口。

8. **VLM 合成。** 用查询 + top-3 场景片段（图像或短片段）+ 字幕调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评测。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 个问题的自定义集。报告总体精度 + 按类别分解（计数、动作、描述性）。

## 使用示例

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付物

`outputs/skill-video-qa.md` 是交付物。给定一个 YouTube URL 或上传的视频，流水线索引场景并用带时间戳引用的方式回答问题。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 时间定位 IoU | 保留定位集上的交并比 |
| 20 | 问答精度 | NeXT-GQA 和自定义 100 个问题 |
| 20 | 摄入吞吐量 | 每花费一美元处理的视频小时数 |
| 20 | UI 和引用体验 | 时间戳链接、缩略图条带、跳转到帧 |
| 15 | 幻觉率 | 计数和动作类精度单独报告 |
| **100** | | |

## 练习题

1. 在字幕生成步骤上将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在 50 个场景的人工评分样本上报告字幕质量差值。

2. 将每场景帧嵌入减少为一个池化向量而不是多向量。测量检索回退情况。

3. 构建"严格计数"模式：合成器提取每个被计数实例及其时间戳，用户点击验证。测量用户验证是否减少幻觉。

4. 对三种 VLM 选择基准测试摄入成本：每美元处理的视频小时数。找出甜蜜点。

5. 添加说话人分隔转录：对音频运行 pyannote 说话人分隔，嵌入每个说话人的字幕。展示"Alice 说了 X 的哪些内容？"查询。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 场景分割 | "镜头检测" | 在镜头边界将视频切割成场景 |
| 多向量索引 | "字幕 + 帧 + 字幕" | 每种表示带命名向量的 Qdrant 集合 |
| 时间定位 | "这究竟发生在什么时候" | 细化查询答案的（start, end）窗口 |
| 帧嵌入 | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似度 |
| RRF 融合 | "倒数排名融合" | 跨多个排名列表的合并策略；经典的混合检索技巧 |
| 计数幻觉 | "误计" | VLM 在"有多少个 X"问题上的已知失败模式 |
| ActivityNet-QA | "视频问答基准" | 长视频问答精度基准测试 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 检查点
- [TimeLens（CVPR 2026）](https://github.com/TencentARC/TimeLens) — 规模化时间定位
- [Gemini 视频长上下文](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源备选
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评测基准

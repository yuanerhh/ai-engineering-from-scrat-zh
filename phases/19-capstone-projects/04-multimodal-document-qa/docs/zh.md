# 顶点项目 04 — 多模态文档问答（视觉优先的 PDF、表格、图表）

> 2026 年文档问答的前沿已从"OCR 后转文本"转向"视觉优先后期交互（late interaction）"。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，用多向量后期交互嵌入，让查询直接与图像块（patch）对齐。在金融 10-K 报告、科学论文和手写笔记上，这种模式大幅超越 OCR 优先的方法。在 1 万页文档上端到端构建这条流水线，并发布与 OCR 转文本方式的对比报告。

**类型：** 顶点项目  
**语言：** Python（流水线），TypeScript（阅览器 UI）  
**前置条件：** 第 4 阶段（计算机视觉），第 5 阶段（NLP），第 7 阶段（Transformer），第 11 阶段（LLM 工程），第 12 阶段（多模态），第 17 阶段（基础设施）  
**涵盖阶段：** P4 · P5 · P7 · P11 · P12 · P17  
**预计时间：** 30 小时

## 问题背景

企业积累了大量被 OCR 流水线处理得残缺不全的 PDF：旋转了的含表格扫描版 10-K 报告、密布公式的科学论文、只有作为图像才有意义的图表、手写批注。将这些当作文本优先处理意味着损失一半的信息。2026 年的答案是对原始页面图像进行后期交互多向量检索。ColPali（Illuin Tech）率先引入了这种方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了精度。在 ViDoRe v3 上，视觉优先检索明显超越 OCR 转文本方案——而且在图表、表格和手写上，差距还在扩大。

权衡在于存储和延迟。ColQwen 嵌入每页约有 2048 个图像块向量，而不是一个 1024 维的单一向量。原始存储量急剧膨胀。DocPruner（2026 年）以 50% 的剪枝实现了几乎零精度损失。你将索引 1 万页文档，测量 ViDoRe v3 nDCG@5，在 2 秒内提供答案，并与 OCR 转文本基线进行直接对比。

## 核心概念

后期交互（late interaction）意味着每个查询词元与每个图像块词元分别评分，每个查询词元的最大分数被累加。你可以在不需要单一池化向量的情况下进行细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个图像块的嵌入，并在检索时运行 MaxSim 操作。

答案生成器是一个视觉语言模型（VLM），它接收查询以及检索到的 top-k 页面作为图像，并撰写带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学符号，OCR 回退方案（Nougat、dots.ocr）作为可选文本通道被接入。

评测是一个二维矩阵。一个维度：内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）。另一个维度：检索方式（视觉优先后期交互 vs OCR 转文本 vs 混合）。每个单元格分别计算 nDCG@5 和答案精度。这份报告就是交付物。

## 架构图

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，竖向归一化
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：Vespa 多向量字段，或 Qdrant 多向量，或 AstraDB（MaxSim）
- 剪枝：DocPruner 2026 策略（保留高方差图像块，50% 压缩，精度损失 < 0.5%）
- OCR 回退（公式/密集表格）：dots.ocr 或 Nougat
- VLM 答案生成器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 备选
- 评测：ViDoRe v3 基准，M3DocVQA（多页推理）
- 阅览器 UI：Next.js 15，带证据区域画布叠加

## 构建步骤

1. **摄入。** 遍历包含 10-K 报告、科学论文和扫描文档的 1 万 PDF 页面语料库。将每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每张页面图像运行 ColQwen2.5-v0.2。输出形状约为 dim 128 的 2048 个图像块嵌入。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对于每个传入查询，用查询塔（token 级嵌入）进行嵌入。对索引运行 MaxSim：对每个查询 token，取所有页面图像块嵌入的最大点积，求和。返回 top-k 页面。

4. **合成。** 用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："仅根据提供的页面回答。对每个论断以 (doc_id, page) 引用，并指明区域（图表、表格、段落）。"

5. **证据区域。** 后处理答案以提取引用区域。如果 VLM 输出了边界框（Qwen3-VL 会输出），在阅览器中将其渲染为叠加层。

6. **OCR 回退。** 对被识别为公式密集的页面（基于图像方差的启发式判断），运行 Nougat 或 dots.ocr，将 OCR 文本作为额外通道与图像一起传入。

7. **评测。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答精度）。同时在相同语料库上用相同合成器运行 OCR 转文本流水线。生成内容类型 × 方法矩阵。

8. **UI。** 先做 Streamlit 原型；Next.js 15 生产阅览器，带逐页证据区域叠加。

## 使用示例

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付物

`outputs/skill-doc-qa.md` 描述了交付物：一个视觉优先的多模态文档问答系统，针对特定语料库调优，并在 ViDoRe v3 上与 OCR 转文本基线进行了对比评测。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 精度 | 基准测试数字 vs OCR 文本基线及已发布排行榜 |
| 20 | 证据区域定位 | 引用区域实际包含答案片段的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩率、索引 p95、答案 p95 |
| 20 | 多页推理 | 手工标注 100 个问题多页集上的精度 |
| 15 | 来源查看体验 | 阅览器清晰度、叠加准确度、并排比较工具 |
| **100** | | |

## 练习题

1. 在相同语料库上对比 ColQwen2.5-v0.2 和 ColQwen3-omni。哪些页面一个对了另一个错了？在索引中添加"内容类别"标签，按类型路由。

2. 激进剪枝（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 跌破 OCR 基线的那个点。

3. 构建混合方案：并行运行 OCR 转文本和 ColQwen，用 RRF 融合，再用交叉编码器重排序。混合方案是否超过任一单独方案？在哪里帮助最大？

4. 将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。测量精度-每美元曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索效果。与手写 OCR 流水线进行比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Late interaction（后期交互） | "ColPali 风格检索" | 查询 token 独立对页面图像块评分；MaxSim 聚合 |
| Multi-vector（多向量） | "每图像块嵌入" | 每个文档有多个向量，而不是一个池化向量 |
| MaxSim | "后期交互评分" | 对每个查询 token，取所有文档向量的最大相似度；求和 |
| DocPruner | "图像块压缩" | 2026 年的剪枝方案，保留 50% 的图像块且精度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年视觉文档检索的衡量标准 |
| Evidence region（证据区域） | "引用边界框" | 源页面上定位答案片段的边界框 |
| OCR fallback（OCR 回退） | "公式通道" | 对公式或表格密集页面与视觉同时使用的文本流水线 |

## 延伸阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 参考后期交互文档检索
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 生产就绪检查点
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退

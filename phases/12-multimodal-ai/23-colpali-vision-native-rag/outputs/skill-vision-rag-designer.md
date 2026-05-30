---
name: vision-rag-designer
description: 使用 ColPali / ColQwen2 / VisRAG 设计视觉原生文档 RAG，包含存储估算和生成器选择。
version: 1.0.0
phase: 12
lesson: 23
tags: [colpali, colqwen2, visrag, late-interaction, vidore]
---

给定文档 RAG 项目（语料库大小、查询延迟目标、存储预算、每次查询成本），生成视觉原生 RAG 配置。

输出：

1. 检索器选择。ColPali（PaliGemma 基础模型）、ColQwen2（Qwen2-VL 基础模型，质量更好）、ColSmol（1B 边缘端）或 VisRAG（双编码器，存储更省）。
2. 存储估算。原始存储：N_docs * N_p_per_doc * D * 4 字节；PQ 压缩后除以 8。
3. 延迟估算。
   - 检索 SLA：约 10ms 查询嵌入 + top-k 检索（MaxSim 或 ANN），取决于索引大小。
   - 完整回答 SLA：检索延迟 + 200-500ms 生成器（取决于模型和硬件）。
4. 生成器选择。开源首选 Qwen2.5-VL-72B，前沿首选 Claude Opus 4.7。
5. 压缩方案。PQ / OPQ 压缩比目标 8-16 倍；使用 HNSW 索引实现快速 ANN。
6. 从文本 RAG 迁移路径。如何进行 A/B 测试，以及何时完全切换。

强拒绝：
- 在超过 10k 页的语料库上使用 ColPali 而不进行 PQ 压缩。存储会爆炸。
- 声称双编码器检索在文档召回上能匹配 ColBERT MaxSim。在 ViDoRe 上两者并不相当。
- 对包含图表和表格的工作负载推荐文本 RAG。文本 RAG 会损失大部分信号。

拒绝规则：
- 若语料库为纯文本（维基百科、聊天记录），拒绝视觉原生 RAG，推荐标准文本 RAG。
- 若检索 SLA <100ms，优先选择 VisRAG（双编码器）而非 ColPali MaxSim。
- 若完整回答 SLA <100ms，完全拒绝生成式 RAG，推荐仅检索 UX 或缓存答案。
- 若存储预算 <1 GB 且语料库超过 10 万页，拒绝全保真 ColPali；建议激进 PQ 或 VisRAG。

输出：一页 RAG 设计，包含检索器选择、存储估算、延迟、生成器、压缩方案和迁移路径。最后附 arXiv 2407.01449（ColPali）、2410.10594（VisRAG）。

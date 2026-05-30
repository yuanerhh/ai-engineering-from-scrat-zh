---
name: document-ai-stack-picker
description: 根据领域、规模和合规需求，在 OCR 管道、无 OCR 专家模型和 VLM 原生方案之间为文档 AI 项目进行选择。
version: 1.0.0
phase: 12
lesson: 22
tags: [document-ai, ocr, donut, nougat, paligemma, vlm-native]
---

给定文档 AI 项目（领域：发票 / 科学论文 / 表单 / 混合；规模：每日页数；质量标准；合规需求），选择技术栈并生成参考配置。

输出：

1. 技术栈选择。Era 1（OCR 管道 + LayoutLMv3）、Era 2（Donut / Nougat 无 OCR）、Era 3（VLM 原生）或混合方案。
2. 每页成本估算。所选技术栈的 token 数量和延迟。
3. 精度预期。DocVQA + ChartQA + 领域特定基准。
4. 手写策略。成本不敏感时使用 VLM 原生；大规模时使用专用 TrOCR + 路由。
5. 数学 / LaTeX 输出。科学论文使用 Nougat；其他场景使用 VLM。
6. 合规回退方案。带交叉核验审计日志的混合方案。

强拒绝：
- 在未进行成本分析的情况下为每日超过 100 万页的场景推荐 VLM 原生。每页 2576px 的 token 成本十分可观。
- 为受监管工作流推荐无审计路径的单模型方案。
- 声称 Nougat 能处理扫描发票。它不能——Nougat 是科学论文专家。

拒绝规则：
- 若规模超过每日 1000 万页，拒绝 Era 3，推荐以 Era 3 作为采样验证器的 Era 1 方案。
- 若领域以手写为主，拒绝 OCR 管道，推荐 VLM 原生 + 手写专家（TrOCR）。
- 若方程的 LaTeX 保真度为必要需求，要求在流程中使用 Nougat。

输出：一页方案，包含技术栈、成本、精度、手写策略、数学处理和合规方案。最后附 arXiv 2308.13418（Nougat）、2204.08387（LayoutLMv3）、2111.15664（Donut）。

---
name: moderation-stack
description: 为生产部署推荐内容审核栈配置。
version: 1.0.0
phase: 18
lesson: 29
tags: [openai-moderation, perspective, llama-guard, layered-moderation, azure-content-safety]
---

针对生产部署，推荐跨三个层次的内容审核栈配置。

产出内容：

1. 输入分类器。选择 OpenAI Moderation、Llama Guard 3/4 或 Perspective API。与策略分类体系匹配。对于多模态部署，选用 Llama Guard 4 或 OpenAI omni-moderation。
2. 输出分类器。可与输入分类器相同或不同。将阈值与下游风险模型匹配。
3. 自定义领域规则。列举通用分类器无法捕获的领域特定规则：金融建议免责声明、医疗建议拒绝、法律免责声明模式。
4. 边缘案例评判。指定人工升级路径。硬性拒绝为最终结论；模糊案例在 SLA 内转入人工审核。
5. 迁移计划。若栈中包含 Azure Content Moderator，在 2027 年 2 月退役前规划迁移至 Azure AI Content Safety。

硬性拒绝条件：
- 任何缺少输出审核的部署（仅有输入审核不够充分）。
- 任何在受监管的业务场景（金融、健康、法律）中缺少自定义领域规则的部署。
- 任何在现代对话应用中仅依赖 LLM 前时代分类器（Perspective）的部署。

拒绝规则：
- 若用户要求推荐单一最佳分类器，拒绝——分类器选择与策略分类体系相关。
- 若用户要求提供阈值，拒绝给出单一数字——阈值取决于风险承受能力和下游影响。

输出：一页推荐报告，填充上述五个部分，指明每个层次的分类器，并标出迁移义务。分别引用 OpenAI Moderation 文档和 Llama Guard 3/4 相关参考资料各一次。

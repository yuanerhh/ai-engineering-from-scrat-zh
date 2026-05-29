# 顶点项目 08 — 监管垂直领域生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产形态。用 docling 或 Unstructured 摄入文档，用 ColPali 处理视觉内容。混合搜索，用 bge-reranker-v2-gemma 重排序，用 Claude Sonnet 4.7 配合 60-80% 命中率的提示缓存合成答案。用 Llama Guard 4 和 NeMo Guardrails 做防护，用 Langfuse 和 Phoenix 做监控，用 RAGAS 在 200 个问题的黄金集上评分。在监管领域（法律、临床、保险）构建一个，顶点项目就是通过黄金集评测、红队测试和漂移仪表板这三关。

**类型：** 顶点项目  
**语言：** Python（流水线 + API），TypeScript（聊天 UI）  
**前置条件：** 第 5 阶段（NLP），第 7 阶段（Transformer），第 11 阶段（LLM 工程），第 12 阶段（多模态），第 17 阶段（基础设施），第 18 阶段（安全）  
**涵盖阶段：** P5 · P7 · P11 · P12 · P17 · P18  
**预计时间：** 30 小时

## 问题背景

监管领域 RAG（法律合同、临床试验方案、保险政策）是 2026 年交付量最大的生产形态，因为投资回报显而易见，风险也切实存在。Harvey（Allen & Overy）为法律领域构建了它，Mendable 交付了开发者文档版本，Glean 覆盖了企业搜索。模式是：高保真摄入，混合检索加重排序，带引用强制执行和提示缓存的合成，多层安全防护，持续漂移监控。

难点不在于模型，而在于司法管辖感知的合规（HIPAA、GDPR、SOC2）、引用级别的可审计性、成本控制（提示缓存命中率高时能带来 60-90% 的折扣）、通过 RAGAS 忠实度检测幻觉，以及当源文档被更新而索引没有跟上时的漂移检测。本顶点项目要求你在 200 个问题的黄金集和同步进行的红队测试套件上交付所有这些内容。

## 核心概念

流水线分两侧。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；代码块获得摘要、标签和基于角色的访问标签。向量写入 pgvector + pgvectorscale（低于 5000 万向量时）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮；每次查询运行混合检索，用 bge-reranker-v2-gemma-2b 重排序，用 Claude Sonnet 4.7（提示缓存）合成，输出经过 Llama Guard 4 和 NeMo Guardrails，返回带引用锚点的回答。

评测栈有四层。**黄金集**（200 个带引用的标注 Q/A）用于正确性。**红队**（越狱、PII 提取尝试、超域问题）用于安全性。**RAGAS** 用于每轮自动评分忠实度/答案相关度/上下文精度。**漂移仪表板**（Arize Phoenix）每周监控检索质量和幻觉得分。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索上下文。命中率达到 60-80% 时，每次查询成本下降 3-5 倍。流水线必须针对稳定前缀（系统提示 + 重排序上下文排前）进行设计，以实现高缓存命中率。

## 架构图

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄入：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库：5000 万向量以下用 pgvector + pgvectorscale；否则用 Qdrant Cloud
- 稀疏索引：带字段权重的 Tantivy BM25
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器：自托管 bge-reranker-v2-gemma-2b 或托管 Voyage rerank-2
- LLM：Claude Sonnet 4.7 带提示缓存；备选自托管 Llama 3.3 70B
- 评测：RAGAS 0.2（在线），DeepEval 用于幻觉和越狱套件
- 可观测性：自托管 Langfuse 带标注队列；Arize Phoenix 用于漂移
- 护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清洗
- 合规：代码块上的基于角色的访问标签；GDPR/HIPAA 司法管辖标签

## 构建步骤

1. **摄入。** 用 Unstructured 或 docling 解析语料库（认真构建需要 1000-10000 个文档）。对扫描/视觉丰富的页面路由到 ColPali。生成带摘要、角色标签和司法管辖标签的代码块。

2. **索引。** 稠密嵌入（Voyage-3 或 Nomic-embed-v2）写入 pgvector + pgvectorscale。通过 Tantivy 建立 BM25 侧索引。角色和司法管辖过滤器作为 payload。

3. **混合检索。** 先按角色+司法管辖过滤；然后并行稠密 + BM25；用倒数排名融合合并；前 20 送重排序器；前 5 送合成器。

4. **带提示缓存的合成。** 系统提示 + 静态策略在缓存头中；重排序上下文作为缓存扩展；用户问题作为未缓存后缀。目标稳定状态下 60-80% 的缓存命中率。

5. **护栏。** 输入时用 Llama Guard 4；NeMo Guardrails 拦截超域问题或策略禁止的话题；Presidio 清洗输出中意外的 PII；引用强制执行后置过滤器。

6. **黄金集。** 由领域专家标注的 200 个 Q/A 对，包含（答案、引用）。对智能体按精确引用匹配、答案正确性、忠实度（RAGAS）进行评分。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 提取尝试、超域问题、跨司法管辖泄露。用通过/失败和严重程度评分。

8. **漂移仪表板。** Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。5% 下降时告警。

9. **成本报告。** Langfuse：提示缓存命中率、每次查询 token 数、按阶段划分的每次查询成本。

## 使用示例

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付物

`outputs/skill-production-rag.md` 描述了交付物。一个带合规标签、通过评分标准、配有实时漂移监控的监管领域聊天机器人。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关度 | 黄金集（200 Q/A）上的在线分数 |
| 20 | 引用正确性 | 带可验证来源锚点的答案比例 |
| 20 | 护栏覆盖 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、p95 延迟、每次查询成本 |
| 15 | 漂移监控仪表板 | Phoenix 实时仪表板带每周检索质量趋势 |
| **100** | | |

## 练习题

1. 在不同司法管辖下构建第二个语料库切片（例如，GDPR 旁边加 HIPAA）。在 20 个跨司法管辖探测题上展示角色+司法管辖过滤防止交叉泄露。

2. 测量一周生产流量的提示缓存命中率。找出哪些查询破坏了缓存前缀。重新设计。

3. 添加带 10k token 摘要缓冲区的多轮记忆。测量随对话增长忠实度是否下降。

4. 将 Claude Sonnet 4.7 替换为自托管 Llama 3.3 70B。测量每次查询成本和忠实度差值。

5. 添加"不确定"模式：如果重排序后的最高分低于阈值，智能体说"我没有可信引用"而不是给出答案。测量虚假置信减少情况。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 提示缓存 | "缓存系统 + 上下文" | Claude/OpenAI 特性：缓存的前缀 token 在命中时折扣 60-90% |
| RAGAS | "RAG 评测器" | 忠实度、答案相关度、上下文精度的自动评分 |
| 黄金集 | "标注评测集" | 200+ 个带引用的专家标注 Q/A；真实答案参考 |
| 司法管辖标签 | "合规标签" | 附加到代码块的 GDPR/HIPAA/SOC2 范围；由检索过滤器执行 |
| 引用忠实度 | "有据可查答案率" | 可追溯到可检索来源片段的论断比例 |
| 漂移 | "检索质量衰减" | nDCG 或引用得分的每周变化；告警阈值 5% |
| 红队 | "对抗性评测" | 发布前的越狱、PII 提取、超域探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Glean 企业搜索](https://www.glean.com) — 企业规模 RAG 参考
- [Mendable 文档](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 标准 RAG 评测框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏框架

# 顶点项目 02 — 代码库语义搜索（跨仓库 RAG）

> 2026 年，每家认真的工程团队都在运行一套理解语义而非仅匹配字符串的内部代码搜索系统。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图谱、Aider 的 repomap、Pinterest 内部 MCP——形态如出一辙：摄入多个仓库，用 tree-sitter 解析，在函数和类粒度嵌入，混合搜索，重排序，带引用答题。本顶点项目要求你构建一个能处理 10 个仓库共 200 万行代码、并在每次 git push 后存活增量重索引的系统。

**类型：** 顶点项目  
**语言：** Python（摄入），TypeScript（API + UI）  
**前置条件：** 第 5 阶段（NLP 基础），第 7 阶段（Transformer），第 11 阶段（LLM 工程），第 13 阶段（工具），第 17 阶段（基础设施）  
**涵盖阶段：** P5 · P7 · P11 · P13 · P17  
**预计时间：** 30 小时

## 问题背景

2026 年，每个前沿编程智能体都内置了代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 100 万 token 上下文有所帮助，但并不能消除排序检索的需求。对原始代码块进行朴素的余弦搜索，在生成代码、单体仓库重复内容以及鲜少导入的符号长尾上会毒化搜索结果。生产级答案是：在 AST 感知代码块上进行混合（稠密 + BM25）搜索，配合重排序器，并由符号引用图支撑。

你通过索引一个真实代码仓库集群——而不是一个教程仓库——来学习这一点，并测量 MRR@10、引用忠实度和增量新鲜度。失败模式属于基础设施层面：10 万文件的单体仓库、一次修改了一半文件的 push、一个需要跨四个仓库才能正确回答的查询。

## 核心概念

AST 感知摄入流水线用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界而非固定 token 窗口处分块。每个代码块获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及简短的自然语言摘要。摘要增加了第三种可检索的模态——用户询问"X 是如何授权的"，摘要中提到了"authz"，即使代码中只有 `check_permission`。

检索是混合式的。一次查询同时触发稠密和 BM25 搜索，合并 top-k，然后将并集传给交叉编码器重排序模型（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表送入长上下文合成器（带提示缓存的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），指令要求对每个论断都通过文件和行范围进行引用。没有引用的答案被后置过滤器拒绝。

增量新鲜度是基础设施难题。git push 触发差异计算：哪些文件改变了，哪些符号改变了。只有受影响的代码块重新嵌入。受影响的跨文件符号边（imports、方法调用）被重新计算。索引无需每次提交都重新处理 200 万行代码即可保持一致。

## 架构图

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter 支持 17 种语言文法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 备选
- 稀疏索引：Tantivy（Rust）配合 BM25F，对符号名 vs 正文进行字段加权
- 向量数据库：Qdrant 1.12（混合搜索），或 pgvector + pgvectorscale（向量数低于 5000 万的团队）
- 代码块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，提示缓存
- 重排序器：Cohere rerank-3 或自托管 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex Workflows（摄入），LangGraph（查询智能体）
- 合成器：Claude Sonnet 4.7（100 万上下文）配合提示缓存
- 符号图：Neo4j（托管）或 kuzu（嵌入式），存储导入和调用边
- 可观测性：Langfuse，每步检索 + 合成均有 span

## 构建步骤

1. **摄入遍历器。** 在每次 push 钩子时遍历 git 历史。收集已更改的文件。对每个文件，用 tree-sitter 解析，提取函数和类节点及其完整源码范围。输出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要器。** 将代码块批量送入 Haiku 4.5（系统前言使用提示缓存）。提示词："用一句话总结这个函数，说明其公开契约和副作用。"将摘要与代码块一起存储。

3. **嵌入池。** 两个并行队列：稠密（Voyage-code-3 批量 128）和摘要（同一模型，但在摘要字符串上）。将向量写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，符号正文权重 1，摘要权重 2。支持"查找名为 X 的函数"和"查找做 X 事情的函数"两类查询。

5. **符号图。** 对每个代码块，记录边：imports（此文件使用仓库 Z 中符号 Y）、calls（此函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索。

6. **查询智能体。** LangGraph 包含三个节点。`retrieve` 并行触发稠密 + BM25，以（repo, path, symbol）去重。`rerank` 对前 50 个运行交叉编码器，保留前 10 个。`synth` 用重排序后的代码块作为上下文调用 Claude Sonnet 4.7，缓存系统提示，要求 file:line 引用。

7. **引用强制执行。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的论断都被标记为需要重新询问或删除。仅返回带引用的答案给用户。

8. **增量重索引。** 在每次 webhook 时，计算符号级差异。只重新嵌入文本已更改的代码块。重新计算导入已更改的代码块的符号边。衡量标准：对 200 万行代码库而言，一次 50 个文件的 push 在 60 秒内重新索引完成。

9. **评测。** 标注 100 个跨仓库问题，并给出黄金 file:line 答案。测量 MRR@10、nDCG@10、引用忠实度（带可验证锚点的论断比例）以及 p50/p99 延迟。

## 使用示例

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付物

可交付技能 `outputs/skill-codebase-rag.md`。给定一组仓库，它搭建摄入流水线、混合索引和查询智能体，并为任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 检索质量 | 100 个问题保留集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 带可验证 file:line 锚点的答案论断比例 |
| 20 | 延迟与规模 | 索引语料大小下 10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 50 个文件提交从 git push 到可搜索的时间 |
| 15 | 体验与答案格式 | 引用可点击性、片段预览、追问功能 |
| **100** | | |

## 练习题

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的差值。报告开启重排序后差距是否收窄。

2. 向语料库中注入 20% 的生成代码（LLM 产生的样板代码）并重新评测。观察检索污染现象。在 payload 中添加"generated"标志并对这些结果降权。

3. 在你的语料库大小下对 Qdrant 混合搜索与 pgvector + pgvectorscale 进行基准测试。报告批量大小为 1 时的 p99 结果。

4. 添加基于采样的漂移检查：每周重新运行 100 个问题的评测。当 MRR@10 下降超过 5% 时发出告警。

5. 扩展到跨语言符号解析：一个通过 gRPC 调用 Go 服务的 Python 函数。使用符号图将它们关联起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AST 感知分块 | "函数级切分" | 在 tree-sitter 节点边界而非固定 token 窗口处切割代码 |
| 混合搜索 | "稠密 + 稀疏" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| 交叉编码器重排序 | "第二阶段排序" | 将（查询, 候选）对一起评分的模型，比余弦更精确 |
| 提示缓存 | "缓存系统提示" | 2026 年 Claude/OpenAI 特性，对重复前缀 token 最高折扣 90% |
| 符号图 | "代码图" | 跨文件和仓库的导入、调用、继承边 |
| 引用忠实度 | "有据可查答案率" | 用户通过点击锚点并阅读引用范围可验证的论断比例 |
| 增量重索引 | "push 到可搜索时间" | 从 git push 到已更改符号可查询的墙上时钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本顶点项目的参考深度剖析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序仓库视图
- [Augment Code 企业图谱](https://www.augmentcode.com) — 商业符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考

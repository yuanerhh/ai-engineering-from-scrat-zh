# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接将它们锚定。关系抽取找到它们之间的边。知识图谱是节点、边及其来源溯源的总和。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 06（命名实体识别）、Phase 5 · 25（实体链接）
**时长：** 约 60 分钟

## 问题背景

分析师读到："Tim Cook became CEO of Apple in 2011."（蒂姆·库克于 2011 年成为苹果公司 CEO）四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction，RE）将自由文本转化为结构化三元组 `(主体, 关系, 客体)`。聚合整个语料库，就得到了知识图谱。聚合并查询，就得到了 RAG、分析或合规审计的推理底座。

2026 年的问题：LLM 热情地抽取关系，但热情过头了。它们会幻化出源文本并不支持的三元组。没有溯源，你无法分辨真实三元组和貌似合理的虚构。2026 年的答案是 AEVS 风格的锚定验证流水线。

## 核心概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(主体实体, 关系类型, 客体实体)`。关系来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（开放信息抽取风格，任意关系均可）。

**三种抽取方法。**

1. **规则/模式匹配（Rule / pattern-based）。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工编写的正则表达式。易碎、精确、可解释。
2. **有监督分类器（Supervised classifier）。** 给定句子中两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015-2022 年的标准方案。
3. **生成式 LLM（Generative LLM）。** 提示模型输出三元组。开箱即用，但需要溯源，否则会幻化出貌似合理的垃圾。

**AEVS（锚定-抽取-验证-补全，Anchor-Extraction-Verification-Supplement，2026）。** 当前的幻觉缓解框架：

- **锚定（Anchor）。** 标识每个实体跨度和关系短语跨度的精确位置。
- **抽取（Extract）。** 生成链接到锚定跨度的三元组。
- **验证（Verify）。** 将每个三元组元素与源文本匹配；拒绝任何不受支持的内容。
- **补全（Supplement）。** 覆盖率遍历确保没有已锚定的跨度被丢弃。

幻觉大幅减少，但需要更多计算，且可审计。

**开放与封闭的权衡。**

- **封闭本体（Closed ontology）。** 固定属性列表（如 Wikidata 的 11,000+ 属性）。可预测、可查询、难以凭空创造。
- **开放信息抽取（Open IE）。** 任何动词短语都可以成为关系。高召回，低精确，难以查询。

生产知识图谱通常混合使用：用开放信息抽取进行发现，然后在合并到主图之前将关系规范化到封闭本体上。

## 动手实现

### 步骤一：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

见 `code/main.py` 中的完整教学抽取器。Hearst 模式仍然在领域专属流水线中使用，因为它们可调试。

### 步骤二：有监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：输入文本，输出三元组，已转化为 Wikidata 属性 ID。在远程监督数据上微调。标准开源权重基准线。

### 步骤三：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

对每个返回的跨度验证源文本。拒绝任何 `text[start:end] != triple_entity` 的内容。这是 AEVS"验证"步骤的最简形式。

### 步骤四：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占工程工作量的 60-80%，要为此留出预算。

### 步骤五：构建小型图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 KG-RAG 系统的原子单元。使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储进行扩展。

## 常见坑

- **RE 之前要先做指代消解。** "He founded Apple"——RE 需要知道"he"是谁。先运行指代消解（第 24 课）。
- **实体规范化。** "Apple Inc"和"Apple"必须解析到同一节点。先做实体链接（第 25 课）。
- **幻化的三元组。** LLM 会输出源文本不支持的三元组，必须强制执行跨度验证。
- **关系规范化漂移。** 开放信息抽取的关系不一致（"was born in"、"came from"、"is a native of"）。折叠到规范 ID，否则图谱无法查询。
- **时间错误。** "Tim Cook is CEO of Apple"——现在是真的，2005 年是假的。许多关系是时间有界的，要使用限定符（Wikidata 中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在维基百科上训练。法律、医疗和科学文本通常需要领域微调的 RE 模型。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域专属（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| LLM 提示，可审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补全 |
| 大规模新闻信息抽取 | 模式匹配 + 有监督混合 |
| 从零构建知识图谱 | 开放信息抽取 + 人工规范化 |
| 时序知识图谱 | 带限定符抽取（开始/结束时间、时间点） |

集成模式：NER → 指代消解 → 实体链接 → 关系抽取 → 本体映射 → 图谱加载。每个阶段都是潜在的质量门控点。

## 上手实践

将以下内容保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 5 篇新闻文章句子上运行 `code/main.py` 中的模式抽取器，人工检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或小型 LLM），比较三元组。哪种抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 流水线：用 LLM 抽取并对源文本验证跨度。在 50 个维基百科风格句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 三元组（Triple） | 主谓宾 | `(s, r, o)` 元组，是知识图谱的原子单元。 |
| 开放信息抽取（Open IE） | 抽取任何内容 | 开放词汇的关系短语；高召回，低精确。 |
| 封闭本体（Closed ontology） | 固定模式 | 有界的关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化（Canonicalization） | 标准化一切 | 将表面名称/关系映射到规范 ID。 |
| AEVS | 有根据的抽取 | 锚定-抽取-验证-补全流水线（2026）。 |
| 溯源（Provenance） | 真相来源链接 | 每个三元组携带指向其源文本的文档 ID + 字符跨度。 |
| 远程监督（Distant supervision） | 廉价标注 | 将文本与已有知识图谱对齐以创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力方案
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉缓解设计
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图查询

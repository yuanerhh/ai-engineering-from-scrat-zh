# 实体链接与消歧

> NER 找到了"巴黎"。实体链接要决定：是法国巴黎？是帕丽斯·希尔顿？是德克萨斯州的巴黎？还是巴黎（特洛伊王子）？不做链接，你的知识图谱就会一直模糊不清。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 06（命名实体识别）、Phase 5 · 24（指代消解）
**时长：** 约 60 分钟

## 问题背景

有一个句子写道："Jordan beat the press."（乔丹击败了媒体）你的 NER 将"Jordan"标注为人名（PERSON）。很好。但是哪个 Jordan？

- 迈克尔·乔丹（篮球运动员）？
- 迈克尔·B·乔丹（演员）？
- 迈克尔·I·乔丹（伯克利 ML 教授——是的，这种混淆在 ML 论文中真实存在）？
- 约旦（国家）？
- 乔丹（希伯来语名字）？

实体链接（Entity Linking，EL）将每个提及解析到知识库（KB）中的唯一条目：Wikidata、Wikipedia、DBpedia，或你的领域知识库。两个子任务：

1. **候选生成（Candidate generation）。** 给定"Jordan"，知识库中哪些条目是可能的？
2. **消歧（Disambiguation）。** 根据上下文，哪个候选是正确的？

两个步骤都是可学习的，都有基准测试。整个流水线框架已经稳定了十年——变化的是消歧器的质量。

## 核心概念

![实体链接流水线：提及 → 候选 → 消歧实体](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（"Jordan"），在别名索引中查找候选。维基百科的别名字典覆盖了大多数命名实体："JFK"→ 约翰·肯尼迪、杰奎琳·肯尼迪、肯尼迪机场、电影《JFK》。典型的索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验概率 + 上下文（Prior + context，Milne & Witten 2008）。** `P(实体 | 提及) × 上下文相似度(实体, 文本)`。效果好、速度快、无需训练。
2. **基于嵌入（Embedding-based，ESS / REL / Blink）。** 编码提及 + 上下文。编码每个候选的描述。取最大余弦相似度。2020-2024 年的默认方案。
3. **生成式（Generative，GENRE 2021；基于 LLM，2023+）。** 逐 token 解码实体的规范名称。约束到有效实体名称的 trie（前缀树），确保输出是有效的知识库 ID。

**端到端 vs 流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次推断中完成 NER + 候选生成 + 消歧。流水线系统在生产中仍占主导，因为可以独立替换各组件。

### 两个评估维度

- **提及召回率（Mention recall，候选生成）。** 正确知识库条目出现在候选列表中的黄金提及比例。是整个流水线的下界。
- **消歧准确率 / F1。** 给定正确候选，top-1 预测正确的频率。

始终同时报告两者。一个消歧准确率 99% 但候选召回率只有 80% 的系统，实际是一个 80% 的流水线。

## 动手实现

### 步骤一：从维基百科重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

维基百科别名数据：约 1800 万个（别名, 实体）对。从 Wikidata 转储下载。存储为倒排索引。

### 步骤二：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠是一个教学玩具。用嵌入的余弦相似度替换（见 `code/main.py` 步骤二中的 Transformer 版本）。

### 步骤三：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引时，对每个知识库实体嵌入一次。查询时，对提及 + 上下文嵌入一次，与候选池做点积，取最大值。

### 步骤四：生成式实体链接（概念）

GENRE 逐字符解码实体的维基百科标题。约束解码（见第 20 课）确保只能输出有效标题。与基于知识库的 trie 紧密集成。现代继承者是 REL-GEN 和带结构化输出的 LLM 提示 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年最简单可落地的 EL 流水线。

### 步骤五：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准的 EL 基准：1,393 篇路透社文章，34k 提及，维基百科实体。报告知识库内准确率（`P@1`）和知识库外 NIL 检测率。

## 常见坑

- **NIL 处理。** 有些提及不在知识库中（新兴实体、不知名人物）。系统必须预测 NIL 而不是猜测错误的实体，需单独测量。
- **提及边界错误。** 上游 NER 遗漏了部分跨度（"美国银行"只标注了"银行"）。EL 召回率下降。
- **流行度偏差（Popularity bias）。** 训练的系统过度预测频繁出现的实体。ML 论文中"Michael I. Jordan"的提及往往链接到篮球明星乔丹。
- **跨语言 EL。** 将中文文本中的提及映射到英文维基百科实体。需要多语言编码器或翻译步骤。
- **知识库过时。** 新公司、事件、人物不在去年的维基百科转储中。生产流水线需要刷新循环。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 通用英语 + 维基百科 | BLINK 或 REL |
| 跨语言，知识库 = 维基百科 | mGENRE |
| LLM 友好，每天少量提及 | 提示 Claude/GPT-4 提供候选列表 + 约束 JSON |
| 领域专属知识库（医疗、法律） | 带知识库感知检索的定制 BERT + 在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅精确匹配先验（Milne-Witten 基准线） |
| 研究前沿 | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年落地的生产模式：NER → 指代消解 → 对每个提及做 EL → 将簇折叠为每簇一个规范实体。输出：文档中每个实体一个知识库 ID，而不是每个提及一个。

## 上手实践

将以下内容保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在 `code/main.py` 中对 10 个模糊提及（Paris、Jordan、Apple）实现先验 + 上下文消歧器。人工标注正确实体，测量准确率。
2. **中等。** 用句子 Transformer 编码 50 个模糊提及。嵌入每个候选的描述。比较基于嵌入的消歧与 Jaccard 上下文重叠的效果。
3. **困难。** 构建一个 1000 个实体的领域知识库（如公司内的员工 + 产品）。端到端实现 NER + EL。在 100 个保留句子上测量精确率和召回率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 实体链接（Entity linking，EL） | 链接到维基百科 | 将提及映射到知识库中的唯一条目。 |
| 候选生成（Candidate generation） | 可能是谁？ | 返回一个提及的可能知识库条目候选列表。 |
| 消歧（Disambiguation） | 选出正确的那个 | 使用上下文对候选打分，选出胜者。 |
| 别名索引（Alias index） | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在知识库中 | 明确预测没有知识库条目匹配。 |
| 知识库（KB） | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域知识库。 |
| AIDA-CoNLL | 基准数据集 | 1,393 篇路透社文章，附有黄金实体链接标注。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 先验 + 上下文方法的奠基之作
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力方案
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带约束解码的生成式 EL
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准论文
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产技术栈

# 指代消解

> "她打电话给他。他没有接听。医生正在吃午饭。"三个指称指向两个人，却没有一个人被点名。指代消解（Coreference Resolution）要搞清楚谁是谁。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 06（命名实体识别）、Phase 5 · 07（词性标注与句法分析）
**时长：** 约 60 分钟

## 问题背景

从一篇 300 词的文章中提取所有对 Apple Inc. 的提及。当文章写"Apple"时很容易。但当它写"这家公司"、"他们"、"库比蒂诺的科技巨头"或"乔布斯的公司"时就很难了。不将这些提及解析到同一个实体，你的命名实体识别（NER）流水线会漏掉 60-80% 的提及。

指代消解将所有指向同一真实世界实体的表达链接到一个簇中。它是表层 NLP（NER、句法分析）与下游语义（信息抽取、问答、摘要、知识图谱）之间的黏合剂。

2026 年指代消解的重要性：

- **摘要：** "这位 CEO 宣布……"与"Tim Cook 宣布……"——摘要应点名 CEO。
- **问答：** "她打电话给谁？"需要解析"她"。
- **信息抽取：** 知识图谱中"人物1 创立了 Apple"和"Jobs 创立了 Apple"作为两个独立条目是错的。
- **多文档信息抽取：** 跨文章合并同一事件的提及是跨文档指代消解。

## 核心概念

![指代消解聚类：提及 → 实体](../assets/coref.svg)

**任务定义。** 输入：一段文档。输出：提及（文本跨度）的聚类，每个簇指向一个实体。

**提及类型。**

- **命名实体（Named entity）。** "Tim Cook"
- **名词性指称（Nominal）。** "the CEO"、"the company"（"这位 CEO"、"这家公司"）
- **代词性指称（Pronominal）。** "he"、"she"、"they"、"it"（"他"、"她"、"他们"、"它"）
- **同位语（Appositive）。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Rule-based，Hobbs 1978）。** 使用语法规则的句法树代词消解。基准线不错，在代词上出人意料地难以超越。
2. **提及对分类器（Mention-pair classifier）。** 对每对提及（m_i, m_j）预测它们是否共指。通过传递闭包聚类。2016 年前的标准方案。
3. **提及排序（Mention-ranking）。** 对每个提及，排序候选先行词（包括"无先行词"），选取最高分。
4. **基于跨度的端到端（Span-based end-to-end，Lee et al. 2017）。** Transformer 编码器。枚举所有不超过长度上限的候选跨度。预测提及分数。预测每个跨度的先行词概率。贪心聚类。现代默认方案。
5. **生成式（Generative，2024+）。** 提示 LLM："列出这段文本中的每个代词及其先行词。"在简单案例上表现良好，在长文档和罕见指称上表现不佳。

**评估指标。** 五种标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能完整衡量聚类质量。将前三者的平均值报告为 CoNLL F1。2026 年在 CoNLL-2012 上的最高水平：约 83 F1。

**已知的难点案例。**

- 指向数页之前引入的实体的限定描述。
- 桥接回指（Bridging anaphora）："车轮"→ 之前提到的一辆车。
- 汉语、日语中的零回指（Zero anaphora）。
- 逆行指代（Cataphora，代词在指称对象之前）："当**她**走进来时，Mary 笑了。"

## 动手实现

### 步骤一：预训练神经网络指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档上，会得到类似：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 步骤二：基于规则的代词消解器（教学用）

见 `code/main.py` 中的纯标准库实现：

1. 提取提及：命名实体（大写跨度）、代词（字典查找）、限定描述（"the X"）。
2. 对每个代词，查看前 K 个提及并按以下标准打分：
   - 性别/数量一致性（启发式）
   - 近邻性（越近越优）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

与神经网络模型不具竞争力，但展示了端到端模型必须做出决策的搜索空间。

### 步骤三：使用 LLM 进行指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两种失败模式：第一，LLM 过度合并（"him"和"her"指向两个不同的人）。第二，LLM 在长文档中静默遗漏提及。始终通过跨度偏移检查验证。

### 步骤四：评估

标准 CoNLL-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，从带注释测试集上的跨度级精确率和召回率开始，然后添加提及链接 F1。

## 常见坑

- **单例爆炸（Singleton explosion）。** 有些系统将每个提及报告为其自己的簇。B³ 对此宽松，MUC 会惩罚。始终检查全部三个指标。
- **长上下文中的代词。** 在超过 2,000 token 的文档上性能下降约 15 F1。要谨慎分块。
- **性别假设。** 硬编码的性别规则在非二元指称、组织、动物上失效。使用学习模型或中性评分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地对跨越 50+ 段落的提及进行聚类。使用滑动窗口（sliding-window）加合并策略。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 英语，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经网络共指 |
| 多语言 | 在 OntoNotes 或多语言 CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专用端到端模型（2025-26 SOTA） |
| 快速 LLM 基准线 | GPT-4o / Claude 配合结构化输出共指提示 |
| 生产对话系统 | 基于规则的回退 + 神经网络主系统 + 关键槽位人工审核 |

2026 年落地的集成模式：先运行 NER，再运行指代消解，将共指簇合并到 NER 实体中。下游任务看到每个簇一个实体，而不是每个表面提及一个实体。

## 上手实践

将以下内容保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 5 个手工编写的段落上运行 `code/main.py` 中的基于规则的消解器。对照真值测量提及链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经网络共指模型。将簇与你自己的人工标注进行比较，它在哪里失败了？
3. **困难。** 构建一个共指增强的 NER 流水线：先 NER，然后通过共指簇合并。在 100 篇文章上测量与仅用 NER 相比的实体覆盖率提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 提及（Mention） | 一个引用 | 指向一个实体的文本跨度（名称、代词、名词短语）。 |
| 先行词（Antecedent） | "it"所指的内容 | 后一个提及与之共指的较早提及。 |
| 簇（Cluster） | 实体的提及集 | 所有指向同一真实世界实体的提及的集合。 |
| 回指（Anaphora） | 向后引用 | 后一提及指向前一提及（"he"→"John"）。 |
| 逆行指代（Cataphora） | 向前引用 | 前一提及指向后一提及（"When he arrived, John..."）。 |
| 桥接（Bridging） | 隐式引用 | "我买了一辆车。车轮坏了。"（那辆车的车轮。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于跨度的端到端方法
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改进共指的预训练方法
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准数据集
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典方法

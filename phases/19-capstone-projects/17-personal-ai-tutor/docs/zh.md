# 顶点项目 17 — 个人 AI 辅导员（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM/Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在 2026 年大规模交付了自适应多模态辅导。共同形态是：苏格拉底式策略（绝不直接给答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音 + 文字 + 照片数学输入、课程图谱检索、间隔重复调度，以及适龄内容的严格安全过滤。这个顶点项目要求你交付一个特定科目的辅导员（K-12 代数或入门 Python），进行为期两周有 10 名学习者参与的效果研究，并通过内容安全审计。

**类型：** 顶点项目  
**语言：** Python（后端，学习者模型），TypeScript（Web 应用），SQL（Postgres + Neo4j 的课程图谱）  
**前置条件：** 第 5 阶段（NLP），第 6 阶段（语音），第 11 阶段（LLM 工程），第 12 阶段（多模态），第 14 阶段（智能体），第 17 阶段（基础设施），第 18 阶段（安全）  
**涵盖阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18  
**预计时间：** 30 小时

## 问题背景

自适应辅导曾经是一个教育科技研究的小众领域。到 2026 年，它已经成为消费产品。Khanmigo 部署在美国大多数学区。Duolingo Max 月活用户数以千万计。Google 的 LearnLM/Gemini for Education 为 Google Classroom 提供辅导支持。Quizlet Q-Chat 与闪卡并肩使用。Synthesis Tutor 以"为好奇心强的孩子提供辅导"的定位实现了病毒式传播。共同要素：多模态输入（打字、说话、拍摄公式）、苏格拉底式教学法（先提问后解释）、每次交互后更新的学习者模型，以及严格的适龄安全过滤。

你将为一个特定群体构建其中一个。衡量标准是一次真实的效果研究：10 名学习者在两周内的前测和后测分数。语音循环必须感觉自然（顶点项目 03 的子技术栈）。记忆必须尊重隐私。安全过滤器必须通过面向 K-12 的 COPPA 感知红队测试。

## 核心概念

四个组件。**辅导员策略**是一个苏格拉底式循环：当学习者要求直接给答案时，策略改为提出引导性问题；当他们答对时，进入下一个概念；当他们卡住时，提供脚手架式提示。**学习者模型**是贝叶斯知识追踪（Bayesian Knowledge Tracing，或简单变体），在每次交互后更新每个课程节点的掌握概率。**课程图谱**是一个带前置条件边的 Neo4j 概念图；策略通过遍历图谱来选择下一个概念。**记忆**是一个情节式 + 语义式存储（agentmemory 风格），保存过去的交互记录、错误和偏好。

用户体验是多模态的。文字输入用于打字回答。语音输入通过 LiveKit + Whisper（复用顶点项目 03）。照片输入通过 dots.ocr 或 PaliGemma 2 处理数学题。语音输出通过 Cartesia Sonic-2。安全性使用 Llama Guard 4 加上一个适龄过滤器（屏蔽成人内容、暴力、自我伤害）以及 COPPA 感知的记忆保留策略。

效果研究是交付物。10 名学习者，前测和后测，两周时间。报告学习增益差值和置信区间。与非自适应基线（相同内容线性呈现，不带辅导员策略）进行比较。

## 架构图

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 科目选择：K-12 代数或入门 Python（选一个深入）
- 辅导员策略：基于 Claude Sonnet 4.7（带提示缓存）的 LangGraph
- 学习者模型：经典贝叶斯知识追踪或 FSRS 间隔重复
- 课程图谱：概念节点 + 前置条件边 + OER 内容的 Neo4j
- 记忆：agentmemory 风格的持久化向量 + 情节式 + 语义式存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用顶点项目 03 子技术栈）
- 照片数学：dots.ocr 或 PaliGemma 2 用于公式识别
- 安全性：Llama Guard 4 + 自定义适龄过滤器
- 评测：Bloom 层次问题生成，前/后测框架，效果研究工具

## 构建步骤

1. **课程图谱。** 在 Neo4j 中构建 50-150 个概念节点（例如，K-12 代数从"数轴"到"二次方程"）的图谱，带前置条件边。为每个节点附加 OER 内容（开放教材、OpenStax）。

2. **学习者模型。** 用先验参数初始化贝叶斯知识追踪：猜测率、失误率、学习率。每次交互后更新每个概念的掌握率。按学习者持久化。

3. **辅导员策略。** LangGraph，节点包括：`read_signal`（学习者的答案是正确/部分正确/卡住？）、`select_concept`（遍历课程图谱选择最高优先级概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. **记忆。** 每次交互写入情节式存储。错误和偏好升级到语义记忆。COPPA 感知保留策略：1 年后自动删除，家长可访问。

5. **语音通道。** 连接辅导员策略的 LiveKit Agents worker。ASR 通过 Whisper-v3-turbo。TTS 通过 Cartesia Sonic-2。支持打断（复用顶点项目 03 的机制）。

6. **照片数学通道。** 上传或拍摄图像；运行 dots.ocr 或 PaliGemma 2 识别公式；作为结构化输入馈送给辅导员。

7. **安全性。** 每个模型输出通过 Llama Guard 4 + 适龄过滤器（屏蔽自我伤害、成人内容、暴力）。记忆访问按学习者 ID 限定范围；为家长提供删除访问接口。

8. **效果研究。** 10 名学习者，前测（标准化 30 道题基线），两周辅导交互（每周 3 次课），后测。与在相同内容上的 10 名非自适应基线组学习者进行比较。

9. **每周进度报告。** 为每个学习者自动生成 PDF 摘要，包含探索的主题、掌握度轨迹和推荐的后续步骤。

## 使用示例

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付物

`outputs/skill-ai-tutor.md` 是交付物。一个特定科目的自适应辅导员，带多模态输入、学习者模型、记忆、安全保障和可测量的效果。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 学习增益差值 | 10 名学习者两周研究的前/后测差值 |
| 20 | 苏格拉底式策略忠实度 | 对转录样本的评分标准评分 |
| 20 | 多模态体验 | 语音 + 照片 + 文字端到端连贯性 |
| 20 | 安全 + 隐私立场 | Llama Guard 4 通过率 + COPPA 感知保留 |
| 15 | 课程广度和图谱质量 | 概念覆盖度 + 前置条件图谱一致性 |
| **100** | | |

## 练习题

1. 有无自适应学习者模型（随机概念顺序）地运行效果研究。报告差值。预期自适应会胜出，但胜出幅度才是有趣的数字。

2. 添加多模态探测：同一概念问题以文字、语音和照片三种方式呈现。测量学习者是否用他们偏好的模态收敛得更快。

3. 构建家长仪表板：练习的主题、掌握度轨迹、即将到来的概念、安全事件（任何护栏触发）。COPPA 合规。

4. 添加语言切换模式：辅导员接受西班牙语输入并用西班牙语教学。测量 X-Guard 覆盖情况。

5. 对记忆隐私进行压力测试：验证即使通过语音片段重新摄入攻击，学习者 A 也无法看到学习者 B 的数据。记录尝试的访问并发出告警。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 苏格拉底式策略 | "提问而非直接告知" | 辅导员提出引导性问题而不是直接给出答案 |
| 贝叶斯知识追踪（BKT） | "学习者模型经典算法" | 每个概念掌握概率的经典学习者模型方程 |
| FSRS | "自由间隔重复调度器" | 2024 年间隔重复调度器，优于 SM-2 |
| 课程图谱 | "概念 DAG" | 带前置条件边的 Neo4j 概念图谱 |
| 情节式记忆 | "每次交互日志" | 每次交互都存储以备后续检索 |
| 语义记忆 | "已学习模式存储" | 从情节式记忆中升级的压缩错误和偏好 |
| COPPA | "儿童隐私法" | 限制对 13 岁以下儿童数据收集的美国法律 |

## 延伸阅读

- [Khanmigo（可汗学院）](https://www.khanmigo.ai) — 参考消费级 K-12 辅导员
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习辅导员
- [Google LearnLM/Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 备选参考
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考
- [FSRS 算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈

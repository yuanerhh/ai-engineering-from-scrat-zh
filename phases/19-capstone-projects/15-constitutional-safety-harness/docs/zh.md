# 顶点项目 15 — 宪法安全防护框架 + 红队靶场

> Anthropic 的宪法分类器、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 内容安全，以及用于多语言覆盖的 X-Guard，定义了 2026 年的安全分类器技术栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为了标准对抗性评测工具。NeMo Guardrails v0.12 将它们整合进一个生产流水线。这个顶点项目将所有这些串联在一起：围绕目标应用的分层安全防护框架，一个运行 6 种以上攻击类别的自主红队智能体，以及一个产生可测量无害性提升的宪法自我批评运行。

**类型：** 顶点项目  
**语言：** Python（安全流水线，红队），YAML（策略配置）  
**前置条件：** 第 10 阶段（从零构建 LLM），第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 18 阶段（伦理、安全、对齐）  
**涵盖阶段：** P10 · P11 · P13 · P14 · P18  
**预计时间：** 25 小时

## 问题背景

2026 年 LLM 安全的前沿不在于分类器是否有效（大体上是有效的），而在于如何正确地在生产应用周围组合它们——既不过度拒绝，也不留下明显的漏洞。Llama Guard 4 处理英语策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕捉基于图像的提示词注入。NVIDIA Nemotron 3 内容安全覆盖企业类别。Anthropic 的宪法分类器是一种在训练期间而非服务期间使用的不同方法。

攻击演化同样重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮和代码混用攻击利用智能体记忆。任何已部署的 LLM 都需要一个红队靶场——garak 和 PyRIT 是标准驱动工具——加上记录在案的缓解措施和 CVSS 评分的发现。

你将加固一个目标应用（一个 8B 指令调优模型或其他顶点项目的某个 RAG 聊天机器人），对其运行 6 种以上的攻击类别，并产生前后无害性测量结果。

## 核心概念

安全流水线是五层。**输入清洗**：去除零宽字符，解码 base64/rot13，规范化 Unicode。**策略层**：NeMo Guardrails v0.12 护栏（超域、毒性、PII 提取）。**分类器门控**：输入时用 Llama Guard 4，非英语内容用 X-Guard，图像输入用 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：输出时用 Llama Guard 4，Presidio PII 清洗，适用时的引用强制执行。**HITL 层**：标记为高风险的输出进入 Slack 队列。

红队靶场按计划运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII/base64/rot13 编码攻击。多轮攻击（人物角色扮演、记忆利用）。代码混用攻击（英语混合斯瓦希里语或泰语）。每次运行生成一个带 CVSS 评分和披露时间线的结构化发现文件。

宪法自我批评运行是一种训练期干预。取 1000 个有害尝试提示词，让模型起草一个回复，对照书面宪法（禁止伤害规则）进行批评，然后在批评循环上重新训练。在保留评测集上测量前后无害性差值。

## 架构图

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 内容安全、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动器：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao 等，2023）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪法训练：Anthropic 风格的自我批评循环 + 批评微调
- PII 清洗：Presidio
- 目标：一个 8B 指令调优模型或其他顶点项目的某个 RAG 聊天机器人

## 构建步骤

1. **目标设置。** 在 vLLM 上搭建一个 8B 指令调优模型（或复用其他顶点项目的 RAG 聊天机器人）。这是被测应用。

2. **安全流水线封装。** 将五层流水线围绕目标串联。验证每一层都可以单独观测（Langfuse 中每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在一个小型标注集上运行每个，建立基线。

4. **红队调度器。** 调度 garak、PyRIT、一个 PAIR 智能体、一个 TAP 智能体、一个 GCG 运行器、一个多轮攻击器和一个代码混用攻击器。每个在独立队列上运行。

5. **攻击套件。** 六种攻击类别：(1) PAIR 自动越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII/base64/rot13 编码，(5) 多轮角色扮演，(6) 多语言代码混用。按类别报告成功率。

6. **宪法自我批评。** 整理 1000 个有害尝试提示词。对每个，目标起草一个回复。一个批评 LLM 对照书面宪法（"无伤害"、"引用证据"、"拒绝非法请求"）进行评分。批评者反对的提示词被改写；目标在批评改进的对上进行微调。在保留评测集上测量前后无害性。

7. **过度拒绝测量。** 在良性提示词套件（例如 XSTest）上跟踪假阳性率。目标必须在良性问题上保持有帮助。

8. **CVSS 评分。** 对每次成功越狱，按 CVSS 4.0（攻击向量、复杂度、影响）进行评分。生成披露时间线和缓解计划。

9. **靶场自动化。** 以上所有内容按计划运行；发现写入队列；过度拒绝回归告警发往 Slack。

## 使用示例

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付物

`outputs/skill-safety-harness.md` 是交付物。一个生产级分层安全流水线，加上一个带前后无害性差值的可复现红队靶场。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6 种以上攻击类别，2 种以上语言 |
| 20 | 真阳性/假阳性权衡 | 攻击阻止率 vs XSTest 良性通过率 |
| 20 | 自我批评差值 | 保留评测集上的前后无害性 |
| 20 | 文档与披露 | 带时间线的 CVSS 评分发现 |
| 15 | 自动化与可重复性 | 一切按计划运行并带告警 |
| **100** | | |

## 练习题

1. 在 RAG 聊天机器人上运行 garak 的提示词注入插件，比较有无输出过滤层时的攻击成功率。

2. 添加第七种攻击类别：通过检索文档的间接提示词注入。测量所需的额外防御。

3. 实现"拒绝并给予帮助"模式：当护栏阻止时，目标提供一个更安全的相关答案而不是简单拒绝。测量 XSTest 差值。

4. 多语言覆盖空白：找到 X-Guard 表现欠佳的一种语言。提出一个针对它的微调数据集。

5. 在 30B 模型上运行宪法自我批评，测量差值是否随规模扩展。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 分层安全 | "纵深防御" | 在输入、门控、输出、HITL 处的多重护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考输入/输出内容分类器 |
| PAIR | "越狱智能体" | Chao 等论文中的 LLM 驱动越狱发现 |
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪心坐标梯度" | 基于梯度的对抗性后缀攻击 |
| 宪法自我批评 | "Anthropic 风格训练" | 目标起草 -> 批评者评分 -> 改写 -> 重新训练 |
| XSTest | "良性探测集" | 过度拒绝回归的基准测试 |
| CVSS 4.0 | "严重性评分" | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic 宪法分类器](https://www.anthropic.com/research/constitutional-classifiers) — 训练期参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 内容安全](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard（arXiv:2504.08848）](https://arxiv.org/abs/2504.08848) — 132 语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 越狱智能体论文

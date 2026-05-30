---
name: skill-library
description: 生成一个 Voyager 形态的技能库，支持注册、按相似度检索、组合执行和失败驱动的精炼。
version: 1.0.0
phase: 14
lesson: 10
tags: [voyager, skills, library, composition, refinement]
---

给定目标运行时和领域，构建一个支持 Voyager 三大组件的技能库：课程钩子、可检索技能库、迭代精炼。

输出内容：

1. `Skill` 类型，包含 `name`、`description`、`code`、`version`、`tags`、`depends_on`、`history`。每次写入时记录上一版本代码。
2. `SkillLibrary`，含 `register(skill, dedup=True)`（新增或版本递增）、`search(query, top_k, tag_filter)`、`get(name)`、`topo_order(name)`（依赖解析）、`execute(name, context)`（拓扑顺序执行）。
3. 检索必须使用嵌入相似度或 BM25，不得对整个库使用 LLM 评分。允许在 top-k 候选上进行 LLM 重排。
4. 执行时必须对每个技能捕获异常，并将其作为精炼循环可消费的反馈注入追踪记录。
5. 一个精炼钩子：`execute` 失败后，运行时收集 (task, skill_name, error, env_state)，传给模型，并对重写后的技能调用 `register`。版本递增；历史记录保留旧代码。

硬性拒绝：

- 技能库中技能以散文字符串存储而非代码。技能必须可执行，散文属于 `description`。
- 组合执行时不做拓扑排序。无环检测的深度优先遍历在技能 DAG 上会崩溃。
- 静默版本覆盖。每次精炼必须递增 `version`，并将旧代码推入 `history` 以备审计。

拒绝规则：

- 如果目标运行时对技能执行没有沙箱，拒绝在涉及生产系统的领域上线。上线前必须先有沙箱（参考第 9 课原则）。
- 如果用户要求"每次失败都自动重试但不做精炼"，拒绝。不精炼的重试只会放大 bug，而非修复它。
- 如果库中技能超过约 200 个且使用扁平检索，拒绝称其为"生产可用"。先添加标签过滤和层级命名空间。

输出：`skill.py`、`library.py`、`execute.py`、`refine.py`，以及 `README.md`（说明去重规则、检索后端、精炼提示词和版本策略）。末尾附"下一步阅读"，若需集成 Claude Agent SDK 指向第 17 课，若需 OpenAI Agents SDK 工具转换指向第 16 课，若需评估技能库质量指向第 30 课。

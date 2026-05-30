---
name: computer-use-safety
description: 为计算机使用智能体构建逐步安全分类器 + 确认门控，包含允许列表导航和注入标记过滤。
version: 1.0.0
phase: 14
lesson: 21
tags: [computer-use, safety, claude, openai-cua, gemini]
---

给定一个计算机使用智能体和一组目标应用，构建一个在每个动作执行前进行分类的安全层。

输出内容：

1. `SafetyClassifier.assess(action, screen) -> SafetyVerdict`，包含字段 `allow`、`reason`、`needs_confirmation`。
2. 智能体可点击元素标签的允许列表；不在列表中的元素拒绝操作。
3. 智能体可导航 URL 的允许列表；跳转到列表外的 URL 时拒绝操作。
4. 对 DOM 文本、获取的内容和输入文本进行注入标记过滤。任何匹配项都会阻止该动作。
5. 敏感动作（登录、购买、删除、发布）的确认门控。人在回路回调接口。
6. 追踪发射器：每次决策都记录 (action, verdict, reason)。

硬性拒绝：

- 安全分类器只在第一个动作上运行。每个动作都必须被分类。
- 形如 `*` 的允许列表。允许一切的允许列表根本不是允许列表。
- 因为模型"看起来很确信"就跳过确认。确信度不等于安全。

拒绝规则：

- 如果智能体拥有计算机使用权限但没有逐步安全机制，拒绝上线。
- 如果智能体可以导航到任意 URL，拒绝。要求允许列表或屏蔽列表。
- 如果敏感动作在任何模式下都能绕过确认门控，拒绝。

输出：`classifier.py`、`allowlist.py`、`confirmation.py`、`trace.py`、`README.md`（说明门控策略、注入标记和允许列表维护流程）。末尾附"下一步阅读"，指向第 27 课（提示注入）和第 23 课（安全决策的 OTel span 归因）。

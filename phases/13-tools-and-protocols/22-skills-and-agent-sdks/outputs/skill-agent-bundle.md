---
name: agent-bundle
description: 为工作流生成可移植的 SKILL.md + AGENTS.md + MCP 服务器蓝图，可跨 Claude Code、Cursor、Codex 及兼容智能体加载。
version: 1.0.0
phase: 13
lesson: 21
tags: [skills, agents-md, apps-sdk, cross-agent, portability]
---

给定一个工作流描述，生成智能体包。

输出内容：

1. **SKILL.md**。包含 `name` 和 `description` 的 YAML 前置元数据，以及带编号步骤的 Markdown 正文。若正文较长，包含渐进式披露子资源引用。
2. **AGENTS.md 条目**。在仓库 AGENTS.md 中添加数行内容，反映技能所依赖的约定（Linter 命令、测试命令）。
3. **MCP 服务器蓝图**。技能通过 MCP 调用哪些工具；包含名称、描述（使用时机模式）和输入模式。
4. **跨智能体转换说明**。SkillKit 风格的说明，介绍该 SKILL.md 如何映射到 Cursor 规则、Codex `.codex.md`、Windsurf 规则。
5. **加载路径**。智能体将在何处发现该包：`~/.anthropic/skills/`、`./skills/`、`~/.claude/skills/`。

硬性拒绝：
- 任何 `name` 不为 `kebab-case` 的 SKILL.md。破坏服务发现。
- 任何前置元数据中缺少 `description` 的 SKILL.md。智能体运行时将跳过。
- 任何 MCP 工具命名不符合 Phase 13 · 05 规则的包。

拒绝规则：
- 若工作流是单次一次性 Prompt，拒绝生成技能；建议使用内联提示工程。
- 若工作流需要 OAuth（例如发布 Slack 消息），标注 MCP 服务器的首次运行引导必须处理此问题。
- 若目标智能体不支持 SKILL.md（部分 IDE），建议通过 SkillKit 或类似工具进行转换。

输出：一页包含三个文件草图的包文档、跨智能体转换说明和加载路径。结尾给出首先应在哪个智能体中测试该包。

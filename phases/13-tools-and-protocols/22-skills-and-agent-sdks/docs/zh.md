# 技能与智能体 SDK——Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说的是"工具是什么"。技能说的是"如何完成任务"。2026 年的技术栈将两者叠加。Anthropic 的智能体技能（2025 年 12 月发布的开放标准）以带渐进式披露的 SKILL.md 形式交付。OpenAI 的 Apps SDK 是 MCP 加上小部件元数据。AGENTS.md（现已出现在 60,000+ 个代码库中）位于代码库根目录，作为项目级智能体上下文。本课介绍各自的覆盖范围，并构建一个可在智能体间传递的最小 SKILL.md + AGENTS.md 包。

**类型：** 学习
**编程语言：** Python（标准库，SKILL.md 解析器和加载器）
**前置知识：** Phase 13 · 07（MCP 服务器）
**预计时间：** 约 45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写带 YAML 前置元数据和渐进式披露的 SKILL.md。
- 以文件系统方式将技能加载到智能体运行时。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使一个包在 Claude Code、Cursor 和 Codex 中都能工作。

## 问题背景

一名工程师将发布说明撰写工作流提炼为多步骤提示词："读取最新合并的 PR。按领域分组。分别摘要。按团队风格撰写变更日志条目。发布到 Slack 草稿。"他们将其放在 Notion 文档中供团队使用。

现在他们希望从 Claude Code、Cursor 和 Codex CLI 使用这个工作流。每个智能体都有不同的加载指令方式：Claude Code 的斜杠命令、Cursor 的规则文件、Codex 的 `.codex.md`。工程师将工作流复制了三次，并维护三份副本。

AGENTS.md 和 SKILL.md 共同解决了这个问题：

- **AGENTS.md** 位于代码库根目录。每个兼容的智能体在会话开始时读取它。"这个项目是如何工作的？约定是什么？哪些命令运行测试？"
- **SKILL.md** 是一个可移植的包：YAML 前置元数据（名称、描述）+ Markdown 正文 + 可选资源。支持技能的智能体按名称按需加载它们。
- **MCP**（Phase 13 · 06-14）处理技能需要调用的工具。

三个层次，一个可移植制品。

## 核心概念

### AGENTS.md（agents.md）

2025 年底发布，截至 2026 年 4 月已被 60,000+ 个代码库采用。一个文件位于代码库根目录。格式：

```markdown
# 项目：my-service

## 约定
- 使用严格模式的 TypeScript。
- Python 端使用 Pydantic 定义模型。
- 测试通过 `pnpm test` 运行。

## 构建与运行
- `pnpm dev` 用于本地开发服务器。
- `pnpm build` 用于生产构建。
```

智能体在会话开始时读取这个文件，并用它来校准对该项目的行为。2026 年每个编码智能体都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的智能体技能（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: 按照本项目的风格为最新合并的 PR 编写变更日志条目。
---

# 发布说明撰写器

调用时，执行以下步骤：

1. 列出上次标签以来合并的 PR。使用 `gh pr list --base main --state merged`。
2. 按标签分组：feature、fix、chore、docs。
3. 对每个分组中的每个 PR，写一行：`- <标题> (#<编号>)`。
4. 起草发布说明并暂存到 CHANGELOG.md 中。

如果用户说"发布"，运行 `git tag vX.Y.Z` 和 `gh release create`。

## 注意事项

- 永远不要包含没有 PR 的提交。
- 从公开变更日志中跳过"chore"条目。
```

前置元数据声明技能的身份。正文是技能加载时显示给模型的提示词。

### 渐进式披露

技能可以引用子资源，智能体仅在需要时获取。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说"风格规则请参见 style-guide.md"。智能体只在技能正在运行时才拉取 style-guide.md。这避免了将模型可能不需要的细节塞入提示词。

### 文件系统发现

智能体运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名称和前置元数据中的 `name` 加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨智能体）都遵循这种模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，在运行时中将其作为可调用的"智能体"暴露。当用户调用技能时，智能体循环分发到对应技能。

### OpenAI Apps SDK

2025 年 10 月发布；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到单一开发者接口。Apps SDK 应用包括：

- 一个 MCP 服务器（工具、资源、提示）。
- 加上 ChatGPT UI 的小部件元数据。
- 加上可选的 MCP Apps `ui://` 资源用于交互式界面。

相同协议，更丰富的 UX。

### 通过 SkillKit 实现跨智能体可移植性

SkillKit 等跨智能体分发层将单个 SKILL.md 翻译为 32+ 个 AI 智能体（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。一个事实来源；多个消费者。

### 三层技术栈

| 层次 | 文件 | 加载时机 | 目的 |
|------|------|---------|------|
| AGENTS.md | 代码库根目录 | 会话开始时 | 项目级约定 |
| SKILL.md | 技能目录 | 技能被调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用动作 |

三者组合：智能体在会话开始时读取 AGENTS.md，用户调用技能，技能的指令包含 MCP 工具调用，智能体通过 MCP 客户端分发。

## 动手实践

`code/main.py` 提供一个标准库 SKILL.md 解析器和加载器。它发现 `./skills/` 下的技能，解析 YAML 前置元数据和 Markdown 正文，并生成以技能名称为键的字典。然后模拟一个按名称调用 `release-notes-writer` 的智能体循环。

重点关注：
- YAML 前置元数据使用最小标准库解析器解析（无 `pyyaml` 依赖）。
- 技能正文原样存储；智能体在调用时将其前置到系统提示词。
- 通过 `read_subresource` 函数演示渐进式披露，按需拉取引用的文件。

## 产出技能

本课产出 `outputs/skill-agent-bundle.md`：给定一个工作流，该技能生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可在智能体间移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，确认加载器能找到它。

2. 为本课程代码库编写 AGENTS.md。包括测试命令、风格约定和 Phase 13 思维模型。

3. 将你的团队内部文档中的一个多步骤工作流移植为 SKILL.md。在 Claude Code 中验证它能加载。

4. 手动将该技能翻译为 Cursor 和 Codex 的原生规则格式。统计格式差异——这正是 SkillKit 自动化的翻译面。

5. 阅读 Anthropic 智能体技能博客文章。找出 Claude Agent SDK 中本课加载器未涵盖的一个功能。（提示：智能体子调用。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| SKILL.md | "技能文件" | YAML 前置元数据加 Markdown 正文，由智能体运行时加载 |
| AGENTS.md | "代码库根目录智能体上下文" | 会话开始时读取的项目级约定文件 |
| 渐进式披露 | "懒加载子资源" | 技能正文引用仅在需要时拉取的文件 |
| 前置元数据 | "顶部 YAML 块" | `---` 分隔符中的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | "MCP + 小部件元数据" | OpenAI 基于 MCP 加上 ChatGPT UI 钩子构建的开发者接口 |
| 技能发现 | "文件系统扫描" | 遍历已知目录查找 SKILL.md，按名称索引 |
| 跨智能体可移植性 | "一个技能多个智能体" | 通过 SkillKit 类工具将一个 SKILL.md 翻译为 32+ 个智能体 |
| 智能体技能（Agent Skill） | "可移植知识" | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | Connectors 和 Custom GPTs 在 MCP 上的统一 |

## 延伸阅读

- [Anthropic——智能体技能公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic——智能体技能文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI——Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式与采用情况
- [Anthropic——anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例

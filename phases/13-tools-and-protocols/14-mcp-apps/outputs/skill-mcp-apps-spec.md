---
name: mcp-apps-spec
description: 为需要交互式 UI 资源的工具生成完整的 MCP Apps 合约。
version: 1.0.0
phase: 13
lesson: 14
tags: [mcp, apps, ui-resources, csp, iframe-sandbox]
---

给定一个能从交互式 UI（时间轴、表单、仪表盘、地图、图表）中受益的工具，生成 MCP Apps 合约。

输出内容：

1. **`ui://` URI**。UI 资源的唯一规范名称（例如 `ui://notes/timeline`）。
2. **工具结果结构**。包含 `text` 前言和 `ui_resource` 块的 `content[]`；填充 `_meta.ui`。
3. **CSP**。`default-src`、`script-src`、`connect-src`、`img-src`、`style-src` 的最小允许列表。除非必要，避免使用 `'unsafe-inline'`。
4. **权限列表**。如需要则列出摄像头 / 麦克风 / 地理位置 / 网络；不需要则留空。
5. **postMessage 入口点**。UI 将发起哪些 `host.*` 调用及其返回值。
6. **安全检查清单**。与宿主的区分、防点击劫持、严格的 connect-src、用户内容渲染时的 HTML 净化。

硬性拒绝：
- CSP 为 `default-src *`。开放的安全风险。
- 任何超出 UI 实际使用范围的 `permissions` 请求。最小权限原则。
- 任何加载外部脚本的 `ui://` 资源。请打包或拒绝。
- 任何在未净化的情况下渲染用户可控 HTML 的 UI。XSS 漏洞。

拒绝规则：
- 若 UI 仅展示静态结果，拒绝为其搭建 App；改为返回文本内容。
- 若工具能从原生宿主控件（进度条、确认对话框）中受益，建议优先使用这些控件。
- 若宿主尚不支持 MCP Apps（截至 2026 年 4 月的 VS Code 稳定版、Zed、Windsurf），标注回退到纯文本的路径。

输出：一页合约文档，包含 `ui://` URI、工具结果 JSON、CSP、权限、postMessage 入口点和安全检查清单。结尾用一句话说明能渲染该 UI 的最低宿主要求。

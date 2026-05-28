# MCP Apps——通过 `ui://` 实现的交互式 UI 资源

> 纯文本工具输出限制了智能体所能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）允许工具返回沙箱化的交互式 HTML，直接在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内嵌渲染。仪表盘、表单、地图、3D 场景，全部通过一个扩展实现。本课讲解 `ui://` 资源 scheme、`text/html;profile=mcp-app` MIME 类型、iframe 沙箱 postMessage 协议，以及允许服务器渲染 HTML 所带来的安全面。

**类型：** 构建
**编程语言：** Python（标准库，UI 资源发射器）、HTML（示例应用）
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 10（资源）
**预计时间：** 约 75 分钟

## 学习目标

- 从工具调用中返回 `ui://` 资源，并设置正确的 MIME 和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现 iframe 沙箱 postMessage JSON-RPC 以进行 UI 到宿主的通信。
- 应用防御 UI 发起攻击的默认 CSP 和权限策略。

## 问题背景

2025 年时代的 `visualize_timeline` 工具只能返回"以下是按时间顺序排列的 14 条笔记：……"，这只是一段文字。用户真正想要的是交互式时间线。在 MCP Apps 出现之前，选择只有：客户端特定的小部件 API（Claude Artifacts、OpenAI 自定义 GPT HTML）或根本没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）将这份契约标准化。工具结果包含一个 URI 为 `ui://...`、MIME 为 `text/html;profile=mcp-app` 的资源。宿主在带有有限 CSP 且默认无网络访问权限的沙箱 iframe 中渲染它。iframe 内的 UI 通过一个小型 postMessage JSON-RPC 方言与宿主通信。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）都以相同方式渲染相同的 `ui://` 资源。一个服务器，一个 HTML 包，通用 UI。

## 核心概念

### `ui://` 资源 scheme

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "这是您的笔记时间线："},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

然后宿主在 `ui://notes/timeline` URI 上调用 `resources/read`，得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主在沙箱化的 `<iframe>` 中渲染 HTML，配置如下：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明更严格）
- 通过响应头应用服务器声明的 CSP。
- 无来自宿主来源的 Cookie 和 localStorage。
- 网络访问限制在 CSP 的 `connectSrc` 范围内。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信，使用一个小型 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定到对端的精确来源，并在接收端验证 `event.origin` 是否在白名单中，然后再处理任何载荷。两端都不要使用 `"*"`——消息体携带工具调用和资源读取。

```js
// iframe 到宿主（固定到宿主来源）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主到 iframe（固定到 iframe 来源）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 两端的接收器
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 安全地处理 event.data
});
```

UI 可调用的宿主端方法：

- `host.callTool(name, arguments)`——调用服务器工具。
- `host.readResource(uri)`——读取 MCP 资源。
- `host.getPrompt(name, arguments)`——获取提示模板。
- `host.close()`——关闭 UI。

每次调用仍通过 MCP 协议进行，并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera`——访问用户摄像头（用于文档扫描 UI）。
- `microphone`——语音输入。
- `geolocation`——位置。
- `network:*`——比单独 `connectSrc` 更宽泛的网络访问。

每个权限都会在 UI 渲染前向用户显示提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面包括：

- **通过 UI 进行提示注入。** 恶意服务器 UI 可以显示看起来像系统消息的文本，欺骗用户。宿主渲染应在视觉上将服务器 UI 与宿主 UI 区分开来。
- **通过 `connectSrc` 数据外泄。** 如果 CSP 允许 `connect-src: *`，UI 可以将数据发送到任何地方。默认应严格限制。
- **点击劫持。** UI 叠加宿主界面。宿主必须防止 z-index 操控并强制执行不透明度规则。
- **焦点劫持。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

Phase 13 · 15 作为 MCP 安全的一部分对这些内容进行深入讲解；本课仅作介绍。

### `ui/initialize` 握手

iframe 加载后，通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主响应能力列表和会话令牌。UI 在后续每次宿主调用时使用该会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露了两个便利原语：

- `AppRenderer`（服务器端）——将 React / Vue / Solid 组件包装并发射带正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）——接收资源，挂载 iframe，并调解 postMessage。

你可以使用这些，也可以手动编写 HTML 和 JSON-RPC。

### 生态系统状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持情况：

- **Claude Desktop。** 2026 年 1 月起完全支持。
- **ChatGPT。** 通过 Apps SDK 完全支持（同一底层 MCP Apps 协议）。
- **Cursor。** 测试版；通过设置启用。
- **VS Code。** 仅限 Insider 构建版本。
- **Goose。** 完全支持。
- **Zed、Windsurf。** 已列入路线图。

生产中的服务器用途：仪表盘、地图可视化、数据表格、图表构建器、沙箱 IDE 预览。

## 动手实践

`code/main.py` 在笔记服务器基础上扩展了一个 `visualize_timeline` 工具，该工具返回 `ui://notes/timeline` 资源，以及处理该 URI 的 `resources/read`，后者返回包含 SVG 时间线的小型但完整的 HTML 包。HTML 使用标准库模板——无需构建系统。postMessage 以 JS 注释的形式勾勒（因为标准库无法驱动浏览器）。

重点关注：
- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 无需网络访问即可渲染；所有数据都内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在此标准库演示中有文档但不起作用）。

## 产出技能

本课产出 `outputs/skill-mcp-apps-spec.md`：给定一个可以受益于交互式 UI 的工具，该技能生成完整的 MCP Apps 契约：`ui://` URI、CSP、permissions、postMessage 入口点和安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查生成的 HTML。直接在浏览器中打开 HTML；验证 SVG 是否渲染。然后勾勒 UI 调用 `host.callTool("notes_update", ...)` 的 postMessage 契约。

2. 收紧 CSP：删除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码会有什么变化？

3. 添加第二个 UI 资源 `ui://notes/editor`，包含一个用于就地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可以在哪里注入内容？iframe 沙箱能防御什么，不能防御什么？

5. 阅读 SEP-1724 规范，找出 MCP Apps SDK 中这个玩具实现未使用的一个功能。（提示：组件级状态同步。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MCP Apps | "交互式 UI 资源" | SEP-1724 扩展，2026 年 1 月 26 日发布 |
| `ui://` | "应用 URI scheme" | UI 包的资源 scheme |
| `text/html;profile=mcp-app` | "该 MIME" | MCP App HTML 的内容类型 |
| Iframe 沙箱 | "渲染容器" | 带 CSP 和权限的浏览器 UI 沙箱 |
| postMessage JSON-RPC | "UI 到宿主的线路" | 用于宿主调用的小型 postMessage 之上的 JSON-RPC 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据 |
| CSP | "内容安全策略" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务器 SDK 原语" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端 SDK 原语" | 负责调解 postMessage 的 iframe 挂载助手 |
| `ui/initialize` | "握手" | UI 到宿主的第一条 postMessage |

## 延伸阅读

- [MCP ext-apps——GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现与 SDK
- [MCP Apps 规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP——Apps 扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) — 高层次文档
- [MCP 博客——MCP Apps 发布](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布帖
- [MCP Apps API 参考](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格的 SDK 参考

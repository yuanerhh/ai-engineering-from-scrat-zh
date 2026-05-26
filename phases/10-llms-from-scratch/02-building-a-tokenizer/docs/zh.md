# 从零构建分词器

> 第01课给了你一个玩具。本课给你一把武器。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 10 · 01（分词器：BPE、WordPiece、SentencePiece）
**时长：** 约 90 分钟

## 学习目标

- 构建一个处理 Unicode、空白规范化和特殊 token 的生产级 BPE 分词器
- 实现字节级回退，使分词器能够编码任何输入（包括表情符号、CJK 字符和代码）而不产生未知 token
- 添加预分词（pre-tokenization）正则表达式模式，在应用 BPE 合并前在词边界处分割文本
- 在语料库上训练自定义分词器，并在多语言文本上与 tiktoken 的压缩率进行对比

## 问题背景

你在第01课中构建的 BPE 分词器可以处理英文文本。现在试着输入日语。或者表情符号。或者带有混合制表符和空格的 Python 代码。

它会崩溃。

不是因为 BPE 有问题——而是因为实现不完整。生产级分词器需要处理任意编码的原始字节、在分割前规范化 Unicode、管理永不参与 BPE 合并的特殊 token、将预分词与子词分割链式处理，并且全程足够快速，不会成为处理 15 万亿 token 的训练流水线的瓶颈。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些不是玩具数字。这些词汇表背后的合并表是在数百 GB 的文本上训练的，而周边机制——规范化、预分词、特殊 token 注入、聊天模板格式化——正是将能处理"hello world"的分词器与能处理整个互联网的分词器区分开来的东西。

你将要构建这套机制。

## 核心概念

### 完整流水线

生产级分词器不是一个算法，而是一条五阶段的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[规范化]
    B --> C[预分词]
    C --> D[BPE 合并]
    D --> E[特殊 Token]
    E --> F[Token ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有特定的职责：

| 阶段 | 功能 | 重要性 |
|------|------|--------|
| 规范化（Normalize） | NFKC Unicode，可选小写，可选去除重音符号 | "fi" 连字（U+FB01）变为"fi"（两个字符）。若无此步骤，同一个词会得到不同的 token。 |
| 预分词（Pre-Tokenize） | 在 BPE 之前将文本分割成块 | 防止 BPE 跨词边界合并。"the cat"永远不应该产生"e c"这样的 token。 |
| BPE 合并（BPE Merge） | 对字节序列应用学到的合并规则 | 核心压缩步骤。将原始字节转换为子词 token。 |
| 特殊 Token | 注入 [BOS]、[EOS]、[PAD]、聊天模板标记 | 这些 token 有固定 ID，永不参与 BPE 合并。模型需要它们来获得结构信息。 |
| ID 映射（ID Mapping） | 将 token 字符串转换为整数 ID | 模型看到的是整数，而不是字符串。 |

### 字节级 BPE

第01课的分词器操作的是 UTF-8 字节，这是正确的选择。但我们跳过了一个重要问题：当这些字节不是有效的 UTF-8 时会发生什么？

字节级 BPE（Byte-level BPE）通过将每个可能的字节值（0-255）都视为有效 token 来解决这个问题。基础词汇表恰好是 256 个条目。任何文件——文本、二进制、损坏的文件——都可以被分词而不产生未知 token。

GPT-2 加了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使词汇表保持人类可读。字节 0x20（空格）在其映射中变成字符"G"。这纯粹是美观上的处理，算法本身并不关心。

真正的威力在于：字节级 BPE 能处理地球上的每一种语言。中文字符每个占 3 个 UTF-8 字节，日语可以占 3-4 个字节，阿拉伯语、梵文、表情符号——全都只是字节序列。BPE 算法在这些字节序列中寻找模式，与在英文 ASCII 字节中寻找模式完全相同。

### 预分词

在 BPE 接触文本之前，你需要将其分割成块。这防止了合并算法创建跨词边界的 token。

GPT-2 使用正则表达式模式来分割文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式在缩写（"don't"变成"don"+"'t"）、带可选前导空格的单词、数字、标点和空白处分割。前导空格保持附着在单词上——所以"the cat"变成`[" the", " cat"]`，而不是`["the", " ", "cat"]`。

Llama 使用 SentencePiece，完全跳过正则表达式。它将原始字节流视为一个长序列，让 BPE 算法自己确定边界。这更简单，但给了 BPE 更多创建跨词 token 的自由。

这个选择很重要。GPT-2 的正则表达式防止分词器学习"一个词末尾的'the'"和"下一个词开头的'the'"应该合并。SentencePiece 允许这样做，有时产生更高效的压缩，但 token 的可解释性更差。

### 特殊 Token

每个生产级分词器都为结构标记保留 token ID：

| Token | 用途 | 使用方 |
|-------|------|--------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批次对齐的填充 | BERT、T5 |
| `[UNK]` | 未知 token（字节级 BPE 消除了这个） | BERT、WordPiece |
| `<\|im_start\|>` | 聊天消息边界开始 | ChatGPT、Qwen |
| `<\|im_end\|>` | 聊天消息边界结束 | ChatGPT、Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊 token 永远不会被 BPE 分割。它们在合并算法运行之前被精确匹配，替换为固定 ID，周围的文本正常分词。

### 聊天模板

这是大多数人感到困惑、大多数实现出错的地方。

当你向聊天模型发送消息时，API 接受消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON。它看到的是一个平坦的 token 序列。聊天模板使用特殊 token 将消息转换为这个平坦序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板用错了，模型就会产生垃圾输出。它是在一种确切的格式上训练的，任何偏差——缺少换行、交换了 token、多了一个空格——都会把输入推出训练分布之外。

### 速度

Python 对于生产级分词来说太慢了。

tiktoken（OpenAI）用 Rust 编写，带有 Python 绑定。HuggingFace tokenizers 也是 Rust 实现。SentencePiece 是 C++。这些比纯 Python 快 10-100 倍。

换个角度看：以 100 万 token/秒（快速 Python）的速度为 Llama 3 预训练分词 15 万亿 token 需要 174 天；以 1 亿 token/秒（Rust）的速度则只需要 1.7 天。

你用 Python 构建是为了理解算法。在生产环境中，你会使用编译好的实现，只接触 Python 包装层。

## 动手构建

### 步骤一：字节级编码

基础。将任意字符串转换为字节序列，将每个字节映射到可打印字符以便显示，并反向还原。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试，查看字节数：

```python
texts = [
    ("英文", "hello"),
    ("中文", "你好"),
    ("表情符号", "🔥"),
    ("混合", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} 个字符 -> {len(b)} 个字节 -> {b}")
```

"hello"是 5 个字节。"你好"是 6 个字节（每个字符 3 个）。火焰表情符号是 4 个字节。字节级分词器不关心是什么语言，字节就是字节。

### 步骤二：带正则表达式的预分词器

使用 GPT-2 正则表达式模式将文本分割成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，所以我们回退到 ASCII 字符类。对于生产级多语言分词器，请安装 `regex`。

试试看：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保持附着在单词上。缩写在撇号处分割。标点变成独立的块。BPE 永远不会跨这些边界合并 token。

### 步骤三：在字节序列上运行 BPE

来自第01课的核心算法，但现在独立地操作预分词后的每个块。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 步骤四：特殊 Token 处理

特殊 token 需要精确匹配和固定 ID，完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 步骤五：完整分词器类

将所有步骤链在一起：规范化、在特殊 token 处分割、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 步骤六：多语言测试

真正的考验。输入英文、中文、表情符号和代码。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"输入:   {text}")
    print(f"Token数: {len(ids)} 个 ID")
    print(f"解码:   {decoded}")
    print()
```

中文字符每个产生 3 个字节。表情符号产生 4 个字节。这些都不会导致分词器崩溃，也不会产生未知 token。这就是字节级 BPE 的威力。

## 实际使用

### 比较真实分词器

加载 Llama 3、GPT-4 和 Mistral 的实际分词器，查看每个如何处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} 个 token): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} 个 token): {pieces[:20]}...")
```

你会看到同一段文本得到不同的 token 数。词汇表有 128K 的 Llama 3 在合并常见模式上更激进。词汇表有 100K 的 GPT-4 居中。词汇表只有 32K 的 Mistral 产生更多 token，但嵌入层更小。

权衡始终如一：词汇表越大，序列越短，但参数越多。

## 产出物

本课产出一个用于构建和调试生产级分词器的提示词。参见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单。** 添加一个 `get_token_bytes(id)` 方法，显示任意 token ID 的原始字节。用它检查你最常见的合并 token 实际代表什么。
2. **中等。** 实现 Llama 风格的预分词器，在空白和数字处分割但保留前导空格。在同一语料库上比较其词汇表与 GPT-2 正则表达式方法的差异。
3. **困难。** 添加一个聊天模板方法，接受一个 `{"role": ..., "content": ...}` 消息列表，生成 Llama 3 聊天格式的正确 token 序列。与 HuggingFace 实现进行对比测试。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 字节级 BPE（Byte-level BPE） | "操作字节的分词器" | 以 256 个字节值为基础词汇表的 BPE——能处理任何输入而不产生未知 token |
| 预分词（Pre-tokenization） | "BPE 之前的分割" | 基于正则表达式或规则的分割，防止 BPE 跨词边界合并 |
| NFKC 规范化 | "Unicode 清理" | 规范分解后兼容组合——"fi"连字变为"fi"，全角"A"变为"A" |
| 聊天模板（Chat template） | "消息变成 token 的方式" | 将角色/内容消息列表转换为平坦 token 序列的确切格式——模型特定，且必须与训练格式一致 |
| 特殊 Token（Special tokens） | "控制 token" | 绕过 BPE 的保留 token ID——[BOS]、[EOS]、[PAD]、聊天标记——在合并前精确匹配 |
| 生育率（Fertility） | "每词 token 数" | 输出 token 与输入词的比率——GPT-4 处理英语约为 1.3，韩语为 2-3，越高意味着上下文窗口浪费越多 |
| tiktoken | "OpenAI 分词器" | 带 Python 绑定的 Rust BPE 实现——比纯 Python 快 10-100 倍 |
| 合并表（Merge table） | "词汇表" | 训练期间学到的字节对合并的有序列表——这就是分词器学到的知识 |

## 延伸阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) — GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) — 支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 论文（Meta，2024）](https://arxiv.org/abs/2407.21783) — 128K 词汇表和分词器训练详情
- [SentencePiece（Kudo & Richardson，2018）](https://arxiv.org/abs/1808.06226) — 语言无关分词
- [GPT-2 分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — 原始字节到 Unicode 的映射

# 子词分词——BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到未见词就卡住。字符级分词器让序列长度爆炸。子词分词器折中两者。每一个现代 LLM 都基于其中一种。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 01（文本处理）、Phase 5 · 04（GloVe / FastText / 子词）
**时长：** 约 60 分钟

## 问题背景

你的词汇表有 50,000 个词。用户输入"untokenizable"，分词器返回 `[UNK]`，模型对这个词失去了所有信号。更糟的是：语料库中 90 百分位的文档包含 40 个罕见词，意味着每篇文档丢失 40 比特的信息。

子词分词解决了这个问题。常见词保持单一 token，罕见词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据能覆盖一切，因为任何字符串最终都是字节序列。

2026 年所有前沿 LLM 都基于三种算法之一（BPE、Unigram、WordPiece），包裹在三种库之一（tiktoken、SentencePiece、HF Tokenizers）中。部署语言模型时必须选择一种。

## 核心概念

![BPE vs Unigram vs WordPiece，逐字符对比](../assets/subword-tokenization.svg)

**BPE（字节对编码，Byte-Pair Encoding）。** 从字符级词汇表开始，统计每个相邻对，将最高频的对合并为新 token，重复直到达到目标词汇表大小。主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE（Byte-level BPE）。** 相同算法，但基于原始字节（256 个基础 token）而非 Unicode 字符，保证零 `[UNK]` token——任意字节序列都能编码。GPT-2 使用 50,257 个 token（256 字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram。** 从庞大的词汇表开始，为每个 token 赋予一元概率，迭代剪除那些移除后对语料库对数似然影响最小的 token。推断时具有概率性：可以采样分词方式（对通过子词正则化进行数据增强有用）。T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并能最大化训练语料库似然的对，而非原始频率最高的对。BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是直接在原始 Unicode 文本上**训练**词汇表的库（BPE 或 Unigram），将空格编码为 `▁`。tiktoken 是 OpenAI 用于对预构建词汇表进行**编码**的快速工具，不做训练。

经验法则：

- **训练新词汇表：** SentencePiece（多语言，无需预分词）或 HF Tokenizers。
- **对 GPT 词汇表快速推断：** tiktoken（cl100k_base，o200k_base）。
- **两者兼顾：** HF Tokenizers——一个库，同时支持训练和服务。

## 动手实现

### 步骤一：从零实现 BPE

见 `code/main.py`。核心循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码了三个要点：`</w>` 标记词尾，使"low"（后缀）和"lower"（前缀）保持区分；频率加权使高频对更早获胜；合并列表有序——推断时按训练顺序应用合并。

### 步骤二：用学到的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素的 O(n·|merges|) 实现。生产实现（tiktoken、HF Tokenizers）使用带优先队列的合并排名查找，接近线性时间运行。

### 步骤三：实践中使用 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # CJK 语言用更低值（如日语用 0.9995，英语用 0.9995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为 `▁`，`character_coverage` 控制罕见字符是被保留还是映射到 `<unk>` 的力度。

### 步骤四：OpenAI 兼容词汇表使用 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅用于编码。速度快（Rust 后端）。与 GPT-4/5 分词完全一致，用于字节计数、成本估算、上下文窗口预算控制。

## 2026 年仍会踩的坑

- **分词器漂移（Tokenizer drift）。** 在词汇表 A 上训练，但部署时用词汇表 B。Token ID 不同，模型输出乱码。在 CI 中检查 `tokenizer.json` 的哈希值。
- **空格歧义。** BPE 中"hello"和" hello"产生不同的 token。始终明确指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 以英语为主的语料库产生对非拉丁文字分解为 5-10 倍更多 token 的词汇表。同样的提示在 GPT-3.5 上用日语/阿拉伯语成本高 5-10 倍，o200k_base 部分修复了这个问题。
- **表情符号分割。** 单个表情符号可能占用 5 个 token。预算上下文时要检查表情符号处理。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 从头训练单语模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 服务 OpenAI 兼容 API | tiktoken（GPT-4+ 用 `o200k_base`） |
| 领域专属词汇（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，与基础词汇表合并 |
| 边缘推断，小模型 | Unigram（更小的词汇表效果更好） |

词汇表大小是一个规模决策，不是常量。粗略经验法则：参数量 <1B 用 3.2 万，1-10B 用 5-10 万，多语言/前沿模型用 20 万+。

## 上手实践

将以下内容保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在 `code/main.py` 的微型语料库上训练 500 次合并的 BPE，对三个保留词编码，统计恰好产生 1 个 token 的词与产生 >1 个 token 的词各有多少。
2. **中等。** 比较 100 个英语维基百科句子在 `cl100k_base`、`o200k_base` 和你用 vocab=32k 训练的 SentencePiece BPE 上的 token 数量，报告每种方案的压缩比。
3. **困难。** 用 BPE、Unigram 和 WordPiece 在同一语料库上训练，衡量在小型情感分类器上使用各种分词器时的下游准确率。选择对 F1 的影响是否超过 1 个百分点？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BPE | 字节对编码 | 贪婪合并最高频字符对，直到达到目标词汇表大小。 |
| 字节级 BPE（Byte-level BPE） | 永不产生未知 token | 基于原始 256 字节的 BPE；GPT-2 / Llama 使用。 |
| Unigram | 概率分词器 | 从大候选集出发，用对数似然剪枝；T5、Gemma 使用。 |
| SentencePiece | 带空格标记的那个 | 在原始文本上训练 BPE/Unigram 的库，空格编码为 `▁`。 |
| tiktoken | 快速的那个 | OpenAI 的 Rust 后端 BPE 编码器，用于预构建词汇表，不做训练。 |
| 合并列表（Merge list） | 神奇数字 | 有序的 `(a, b) → ab` 合并列表；推断时按顺序应用。 |
| 字符覆盖率（Character coverage） | 多罕见才算太罕见 | 分词器必须覆盖训练语料库中字符的比例；典型值约 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 库论文
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考文档
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 使用手册和编码列表

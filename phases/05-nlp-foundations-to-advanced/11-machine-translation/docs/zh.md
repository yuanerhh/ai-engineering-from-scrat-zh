# 机器翻译

> 翻译是为 NLP 研究买单三十年并持续买单的任务。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 10（注意力机制）、Phase 5 · 04（GloVe、FastText、子词）
**时长：** 约 75 分钟

## 问题背景

模型读取一种语言的句子，并生成另一种语言的句子。长度不同，词序不同，有些源词对应多个目标词，反之亦然。习语拒绝逐词映射。"I miss you"在法语中是"tu me manques"——字面意思是"你让我感到缺失"。任何词级对齐都无法存活。

机器翻译是迫使 NLP 发明编码器-解码器、注意力机制、Transformer，最终整个 LLM 范式的任务。每一步进展的出现，都是因为翻译质量是可测量的，而人机差距是顽固的。

本课跳过历史课，直接讲授 2026 年的实用管线：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 和 chrF 评估，以及仍然悄悄进入生产的少数失败模式。

## 核心概念

现代机器翻译是在平行文本上训练的 Transformer 编码器-解码器。编码器用该语言的分词读取源文本，解码器通过交叉注意力（第 10 课）使用编码器的输出，每次生成一个子词作为目标。解码使用束搜索以避免贪婪解码的陷阱。输出经过去分词、去真正字化处理后与参考对比评分。

三个操作选择驱动真实世界的 MT 质量：

- **分词器。** 在混合语言语料库上训练的 SentencePiece BPE。跨语言的共享词汇表使 NLLB 中的零样本语言对成为可能。
- **模型大小。** NLLB-200 distilled 600M 可在笔记本电脑上运行，NLLB-200 3.3B 是已发布的生产默认，54.5B 是研究上限。
- **解码。** 通用内容用束宽 4-5。长度惩罚避免输出过短。需要术语一致性时使用约束解码。

## 动手实现

### 步骤一：调用预训练 MT 模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

三点重要事项：`src_lang` 告诉分词器使用哪种脚本和分割方式；`forced_bos_token_id` 告诉解码器生成哪种语言；两者都是 NLLB 特有的技巧，mBART 和 M2M-100 有各自的约定，不可互换。

### 步骤二：BLEU 和 chrF

BLEU 测量输出和参考之间的 n-gram 重叠。四种参考 n-gram 大小（1-4），精确率的几何平均，对过短输出的简洁惩罚。分数范围 [0, 100]。常用但难以解读：30 BLEU 是"可用"；40 是"好"；50 是"出色"；1 BLEU 以下的差距是噪声。

chrF 测量字符级 F 分数。对形态丰富的语言更敏感，因为 BLEU 在这些语言上低估匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它标准化了分词，使分数可以在不同论文间比较。自己实现 BLEU 计算是产生误导性基准的方式。

### 三层评估体系（2026 年）

现代 MT 评估使用三个互补的指标族。至少使用两个。

- **启发式**（BLEU、chrF）。快速、基于参考、可解释、对改写不敏感。用于历史比较和回归检测。
- **学习式**（COMET、BLEURT、BERTScore）。在人工判断上训练的神经模型；比较翻译与源文本和参考的语义相似度。COMET 自 2023 年以来与 MT 研究的关联最高，是 2026 年质量优先场景的生产默认。
- **LLM 作为评审**（无参考）。提示大型模型在流畅性、充分性、语气、文化适当性上对翻译打分。设计良好的评分标准下，GPT-4 作为评审与人工判断约 80% 一致。用于没有参考的开放式内容。

2026 年实用栈：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，提示 LLM 用于最终面向人类的信号。在信任任何指标用于生产数据之前，用 50-100 个人工标注样本校准每个指标。

无参考指标（COMET-QE、BLEURT-QE、LLM 作为评审）让你在没有参考的情况下评估翻译，这对不存在参考翻译的长尾语言对很重要。

### 步骤三：生产中会出什么问题

上述工作管线 80% 的时间会流畅翻译，剩余 20% 会悄悄失败。已命名的失败模式：

- **幻觉（Hallucination）。** 模型编造源文本中没有的内容。在不熟悉的领域词汇中常见。症状：输出流畅但声称了源文本未陈述的事实。缓解措施：对领域术语进行约束解码，对受监管内容进行人工审查，监控比输入长得多的输出。
- **偏离目标生成（Off-target generation）。** 模型翻译成了错误的语言。NLLB 在罕见语言对上对此出人意料地敏感。缓解措施：验证 `forced_bos_token_id`，始终在输出上用语言识别模型检查。
- **术语漂移（Terminology drift）。** "Sign up"在文档 1 中变成"s'inscrire"，在文档 2 中变成"créer un compte"。对于 UI 文本和面向用户的字符串，一致性比原始质量更重要。缓解措施：词汇表约束解码或后编辑字典。
- **语气不匹配（Formality mismatch）。** 法语"tu"vs"vous"，日语礼貌程度。模型选择训练中更常见的形式。对于面向客户的内容这通常是错误的。缓解措施：如果模型支持，在提示前缀中加入语气标记，或在仅正式语料库上微调小模型。
- **短输入的长度爆炸。** 非常短的输入句子经常产生过长的翻译，因为在源 token 数少于约 5 个时长度惩罚急剧下降。缓解措施：与源长度成比例的硬最大长度上限。

### 步骤四：领域微调

预训练模型是通才。法律、医学或游戏对话翻译可以从在领域平行数据上微调中获得可测量的收益。方法并不特别：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量的平行样本胜过几十万个嘈杂的网络抓取样本。训练数据质量是单个最大的生产杠杆。

## 生产使用

2026 年 MT 的生产栈：

| 使用场景 | 推荐起点 |
|---------|---------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短批次运行，廉价推理，英法/德/西班牙语 | Helsinki-NLP / Marian 模型 |
| 延迟敏感的浏览器端 | ONNX 量化 Marian（约 50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini 加翻译提示 |

截至 2026 年，LLM 在某些语言对上已经超越专业 MT 模型，尤其是在习语内容和长上下文方面。权衡是每 token 成本和延迟。当上下文长度、风格一致性或通过提示进行领域适应比吞吐量更重要时，选择 LLM。

## 上手实践

将以下内容保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一个 5 句英语段落翻译成法语，再翻译回英语。测量回译与原文的接近程度。你应该看到语义保留但词汇选择漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 实现对翻译输出的语言识别检查。集成到 MT 调用中，使偏离目标的生成在返回之前被捕获。
3. **困难。** 在你选择的 5000 对领域语料库上微调 `nllb-200-distilled-600M`。在微调前后测量保留集的 BLEU。报告哪类句子改善了，哪类退步了。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BLEU | 翻译分数 | 带简洁惩罚的 n-gram 精确率，范围 [0, 100]。 |
| chrF | 字符 F 分数 | 字符级 F 分数。对形态丰富的语言更敏感。 |
| NMT | 神经 MT | 在平行文本上训练的 Transformer 编码器-解码器。2017 年以来的默认方案。 |
| NLLB | 不遗漏任何语言 | Meta 的 200 语言 MT 模型族。 |
| 约束解码（Constrained decoding） | 可控输出 | 强制特定 token 或 n-gram 出现/不出现在输出中。 |
| 幻觉（Hallucination） | 编造内容 | 模型输出中源文本不支持的内容。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用微调教程

# 文本处理——分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 2 · 14（朴素贝叶斯）
**时长：** 约 45 分钟

## 问题背景

模型无法阅读"The cats were running."，它读的是整数。

每个 NLP 系统都以相同的三个问题开头：词从哪里开始？词的词根是什么？何时将"run"、"running"、"ran"视为同一事物，何时视为不同事物？

分词错误，模型就从垃圾中学习。如果你的分词器将 `don't` 视为一个 token，但将 `do n't` 视为两个，训练分布就会分裂。如果你的词干提取器将 `organization` 和 `organ` 折叠为同一个词干，主题建模就会失效。如果你的词形还原器需要词性上下文但你没有提供，动词就会被当作名词处理。

本课从零构建三种预处理原语，然后展示 NLTK 和 spaCy 如何完成同样的工作，以便你看到权衡所在。

## 核心概念

三种操作，每种都有其职责和失败模式。

**分词（Tokenization）** 将字符串分割为 token。"Token"是故意模糊的，因为合适的粒度取决于任务。经典 NLP 使用词级别。Transformer 使用子词。没有空格的语言使用字符级别。

**词干提取（Stemming）** 用规则砍掉后缀。快速、激进、简单粗暴。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原（Lemmatization）** 使用语法知识将词归约到其词典形式。较慢、准确、需要查找表或形态分析器。`ran -> run`（需要知道"ran"是"run"的过去时）。`better -> good`（需要知道比较级形式）。

经验法则：速度优先且能容忍噪声时使用词干提取（搜索索引、粗分类）。意义优先时使用词形还原（问答、语义搜索、用户可见的任何内容）。

## 动手实现

### 步骤一：正则表达式词级分词器

最简单的实用分词器在非字母数字字符上分割，同时将标点符号保留为独立 token。不完美，不是最终方案，但只需一行即可运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三种模式按优先级排列：带可选内部撇号的词（`don't`、`it's`）；纯数字；任何单个非空白非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式：`3pm` 分割为 `['3', 'pm']`，因为我们在字母序列和数字序列之间交替。对大多数任务来说足够好。URL、电子邮件、话题标签都会出问题。对于生产环境，在通用模式之前添加特定模式。

### 步骤二：Porter 词干提取器（仅步骤 1a）

完整的 Porter 算法有五个阶段的规则。仅步骤 1a 就涵盖了最频繁的英语后缀，并教会了这个模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下阅读规则。`ies -> i` 规则是 `ponies -> poni` 而非 `pony` 的原因。真实的 Porter 步骤 1b 会修正它。规则相互竞争，前面的规则优先。顺序比任何单一规则都重要。

### 步骤三：基于查找的词形还原器

真正的词形还原需要形态学。一个便于教学的版本使用一个小的词形表和后备规则。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键的教学时刻。`watched` 不在我们的表中，我们的后备只处理 `ing`。真正的词形还原涵盖 `ed`、不规则动词、形容词比较级、有音变的复数（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析工具。

### 步骤四：串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是词性标注器。Phase 5 · 07（词性标注）会构建一个。目前，将所有内容默认为 `NOUN` 并承认这一限制。

## 生产使用

NLTK 和 spaCy 提供了生产版本。每个只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩略词、Unicode 以及正则表达式遗漏的边界情况。`PorterStemmer` 运行所有五个阶段。`WordNetLemmatizer` 需要将词性标签从 NLTK 的 Penn Treebank 方案转换为 WordNet 的缩写集。上面的转换连接代码是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个管线隐藏在 `nlp(text)` 后面。分词、词性标注和词形还原都会运行。在规模上比 NLTK 更快。开箱即用更准确。权衡是你无法轻松替换单个组件。

### 如何选择

| 场景 | 选择 |
|------|------|
| 教学、研究、替换组件 | NLTK |
| 生产、多语言、速度优先 | spaCy |
| Transformer 管线（无论如何都会用模型的分词器） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 两种没人警告你的失败模式

大多数教程教完算法就停了。有两件事会咬坏真实的预处理管线，但几乎从未被讲到。

**可重现性漂移。** NLTK 和 spaCy 在版本之间会改变分词和词形还原行为。在 spaCy 2.x 中产生 `['do', "n't"]` 的代码可能在 3.x 中产生 `["don't"]`。你的模型在一种分布上训练。推理现在在不同的分布上运行。准确率悄悄下降，没人知道为什么。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。在每次升级时运行它。

**训练/推理不匹配。** 训练时用激进的预处理（小写、停用词删除、词干提取），部署时在原始用户输入上运行，然后看着性能崩溃。这是生产 NLP 中最常见的单一失败原因。如果你在训练时预处理，你必须在推理时运行完全相同的函数。将预处理作为函数打包在模型包内交付，而不是作为服务团队会重写的 notebook 单元格。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Token（词元） | "一个词" | 模型消费的任何单位。可以是词、子词、字符或字节 |
| Stem（词干） | "词的词根" | 基于规则的后缀去除的结果。不总是真实的词 |
| Lemma（词形） | "词典形式" | 你会查找的形式。需要语法上下文才能正确计算 |
| POS 标签 | "词性" | 如 NOUN、VERB、ADJ 等类别。准确词形还原所需 |
| Morphology（形态学） | "词形规则" | 词如何根据时态、数量、格来改变形式。词形还原依赖于它 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今仍是最清晰的解释
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实管线的连接方式
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边界情况

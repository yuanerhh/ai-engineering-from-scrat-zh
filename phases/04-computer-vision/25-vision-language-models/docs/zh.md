# 视觉语言模型——ViT-MLP-LLM 模式

> 视觉编码器将图像转换为 token。MLP 投影器将这些 token 映射到 LLM 的嵌入空间。语言模型完成剩余工作。这个模式——ViT-MLP-LLM——就是 2026 年每个生产 VLM 的结构。

**类型：** 学习 + 使用
**语言：** Python
**前置条件：** Phase 4 第 14 课（ViT），Phase 4 第 18 课（CLIP），Phase 7 第 02 课（自注意力）
**时长：** 约 75 分钟

## 学习目标

- 陈述 ViT-MLP-LLM 架构，并解释三个组件各自的贡献
- 在参数量、上下文长度和基准性能方面对比 Qwen3-VL、InternVL3.5、LLaVA-Next 和 GLM-4.6V
- 解释 DeepStack：为何多层 ViT 特征比单一最后层特征更好地收紧视觉-语言对齐
- 用跨模态错误率（Cross-Modal Error Rate，CMER）在生产中测量 VLM 幻觉并据此行动

## 问题背景

CLIP（Phase 4 第 18 课）为图像和文本提供了共享嵌入空间，这对零样本分类和检索已经足够。它无法回答"这张图中有几辆红色汽车？"，因为 CLIP 不生成文本——它只对相似度评分。

视觉语言模型（VLM）——Qwen3-VL、InternVL3.5、LLaVA-Next、GLM-4.6V——将 CLIP 系图像编码器与完整语言模型结合。模型看到一张图像加一个问题，然后生成答案。2026 年，开源 VLM 在多模态基准（MMMU、MMBench、DocVQA、ChartQA、MathVista、OSWorld）上与 GPT-5 和 Gemini-2.5-Pro 相当甚至超越。

三个部件的组合（ViT、投影器、LLM）是标准配置。模型之间的差异在于哪个 ViT、哪个投影器、哪个 LLM、训练数据和对齐方案。一旦理解了这个模式，替换任意组件都是机械性的工作。

## 核心概念

### ViT-MLP-LLM 架构

```mermaid
flowchart LR
    IMG["图像<br/>（H × W × 3）"] --> ViT["视觉编码器<br/>（ViT、CLIP-L、<br/>SigLIP、DINOv3）"]
    ViT --> FEATS["图像 token<br/>（N, d_vit）"]
    FEATS --> PROJ["投影器<br/>（2-4 层 MLP<br/>或 Q-former）"]
    PROJ --> VTOK["LLM 空间中的<br/>图像 token<br/>（N, d_llm）"]
    TXT["文本提示词"] --> TOK["LLM 分词器"]
    TOK --> TTOK["文本 token<br/>（M, d_llm）"]
    VTOK --> CONCAT["交错<br/>或拼接"]
    TTOK --> CONCAT
    CONCAT --> LLM["解码器 LLM<br/>（Qwen3、LLaMA 等）"]
    LLM --> OUT["文本答案"]

    style ViT fill:#dbeafe,stroke:#2563eb
    style PROJ fill:#fef3c7,stroke:#d97706
    style LLM fill:#dcfce7,stroke:#16a34a
```

1. **视觉编码器** — 预训练的 ViT（CLIP-L/14、SigLIP、DINOv3 或微调变体）。生成 patch token。
2. **投影器** — 一个小模块（2-4 层 MLP 或 Q-former），将视觉 token 映射到 LLM 的嵌入维度。大多数微调都在这里进行。
3. **LLM** — 仅解码器的语言模型（Qwen3、Llama、Mistral、GLM、InternLM）。按序列读取视觉 + 文本 token，生成文本。

原则上三个部件都可训练。实践中，视觉编码器和 LLM 大多保持冻结，而投影器在训练——用少量参数传递大量信号。

### DeepStack

普通投影仅使用最后一层 ViT 的特征。DeepStack（Qwen3-VL）从多个 ViT 深度采样特征并堆叠。深层携带高级语义；浅层携带细粒度的空间和纹理信息。将两者都送入 LLM 弥合了"图像包含什么"（语义）和"确切在哪里"（空间定位）之间的差距。

### 三阶段训练

现代 VLM 分阶段训练：

1. **对齐（Alignment）** — 冻结 ViT 和 LLM。仅在图像-描述对上训练投影器。教投影器将视觉空间映射到语言空间。
2. **预训练（Pre-training）** — 解冻所有内容。在大规模交错图像-文本数据（5 亿+ 对）上训练。构建模型的视觉知识。
3. **指令调优（Instruction tuning）** — 在精心筛选的（图像、问题、答案）三元组上微调。教会对话行为和任务格式。这是将"视觉感知语言模型"变成可用助手的步骤。

大多数 LoRA 微调针对第三阶段，使用小型标注数据集。

### 模型家族对比（2026 年初）

| 模型 | 参数量 | 视觉编码器 | LLM | 上下文 | 优势 |
|------|--------|-----------|-----|--------|------|
| Qwen3-VL-235B-A22B（MoE） | 2350 亿（220 亿活跃） | 自研 ViT + DeepStack | Qwen3 | 256K | 通用 SOTA，GUI 智能体 |
| Qwen3-VL-30B-A3B（MoE） | 300 亿（30 亿活跃） | 自研 ViT + DeepStack | Qwen3 | 256K | 更小的 MoE 替代方案 |
| Qwen3-VL-8B（dense） | 80 亿 | 自研 ViT | Qwen3 | 128K | 生产密集型默认方案 |
| InternVL3.5-38B | 380 亿 | InternViT-6B | Qwen3 + GPT-OSS | 128K | 强大的 MMBench / MMVet |
| InternVL3.5-241B-A28B | 2410 亿（280 亿活跃） | InternViT-6B | Qwen3 | 128K | 与 GPT-4o 竞争 |
| LLaVA-Next 72B | 720 亿 | SigLIP | Llama-3 | 32K | 开源，易于微调 |
| GLM-4.6V | ~700 亿 | 自研 | GLM | 64K | 开源，强大的 OCR |
| MiniCPM-V-2.6 | 80 亿 | SigLIP | MiniCPM | 32K | 边缘友好 |

### 视觉智能体

Qwen3-VL-235B 在 OSWorld 上达到全球最高性能——这是一个针对操作 GUI（桌面、移动、Web）的**视觉智能体**的基准。模型看到截图，理解 UI，并发出动作（点击、输入、滚动）。结合工具，它能完成常见桌面任务的闭环。这是大多数 2026 年"AI PC"演示的底层技术。

### 智能体能力 + RoPE 变体

VLM 需要知道视频中帧的**时间位置**。Qwen3-VL 从 T-RoPE（时间旋转位置嵌入）演进到**基于文本的时间对齐**——与视频帧交错的显式时间戳文本 token。模型看到"`<timestamp 00:32>` 帧，提示词"并能推理时间关系。

### 对齐问题

爬取数据集中 12% 的图像-文本对包含未完全基于图像的描述。在此数据上训练的 VLM 悄悄地学会了幻觉——捏造对象、误读数字、臆造关系。在生产中这是主要失败模式。

Skywork.ai 引入了**跨模态错误率（Cross-Modal Error Rate，CMER）**来追踪它：

```
CMER = 文本置信度高但图像-文本相似度（通过 CLIP 系检查器）低的输出比例
```

高 CMER 意味着模型自信地说出未基于图像的内容。将 CMER 监控并作为生产 KPI，使其部署中的幻觉率降低了约 35%。关键不是"修复模型"，而是"将高 CMER 输出路由到人工审查"。

### 使用 LoRA / QLoRA 微调

对 700 亿 VLM 的全量微调对大多数团队来说遥不可及。在注意力 + 投影器层上使用 LoRA（rank 16-64），或使用带 4 位基础权重的 QLoRA，可在单张 A100 / H100 上运行。成本：5000-5 万个样本，100-5000 美元计算费，2-10 小时训练。

### 空间推理仍然较弱

当前 VLM 在空间推理基准（上下、左右、计数、距离）上得分 50-60%（高于随机猜测但远低于人类）。如果你的用例依赖"哪个对象在哪个上面"，需要大量验证——通用 VLM 性能低于人类。对于纯空间任务的更好替代方案：专用关键点/姿态估计器、深度模型或带框几何后处理的检测模型。

## 动手实现

### 步骤一：投影器

你最常训练的部分。带 GELU 的 2-4 层 MLP。

```python
import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vit_dim=768, llm_dim=4096, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x):
        return self.net(x)
```

输入是 `(N_patches, d_vit)` token 张量。输出是 `(N_patches, d_llm)`。LLM 将每个输出行视为另一个 token。

### 步骤二：端到端组装 ViT-MLP-LLM

最小 VLM 前向传播的骨架。真实代码使用 `transformers`；这是概念性布局。

```python
class MinimalVLM(nn.Module):
    def __init__(self, vit, projector, llm, image_token_id):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.llm = llm
        self.image_token_id = image_token_id  # placeholder token in text prompt

    def forward(self, image, input_ids, attention_mask):
        # 1. vision features
        vision_tokens = self.vit(image)                     # (B, N_patches, d_vit)
        vision_embeds = self.projector(vision_tokens)       # (B, N_patches, d_llm)

        # 2. text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, M, d_llm)

        # 3. replace image placeholder tokens with vision embeds
        merged = self._merge(text_embeds, vision_embeds, input_ids)

        # 4. run LLM
        return self.llm(inputs_embeds=merged, attention_mask=attention_mask)

    def _merge(self, text_embeds, vision_embeds, input_ids):
        out = text_embeds.clone()
        expected = vision_embeds.size(1)
        for b in range(input_ids.size(0)):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(positions) != expected:
                raise ValueError(
                    f"batch item {b} has {len(positions)} image tokens but vision_embeds has {expected} patches."
                    " Every sample in the batch must be pre-padded to the same number of image placeholder tokens.")
            out[b, positions] = vision_embeds[b]
        return out
```

文本中的 `<image>` 占位符 token 被真实图像嵌入替换——与 LLaVA、Qwen-VL 和 InternVL 使用的模式相同。

### 步骤三：CMER 计算

一个轻量级运行时检查。

```python
import torch.nn.functional as F


def cross_modal_error_rate(image_emb, text_emb, text_confidence, sim_threshold=0.25, conf_threshold=0.8):
    """
    image_emb, text_emb: embeddings of image and generated text (normalised internally)
    text_confidence:     mean per-token probability in [0, 1]
    Returns:             fraction of high-confidence outputs with low image-text alignment
    """
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    sim = (image_emb * text_emb).sum(dim=-1)        # cosine similarity
    high_conf_low_sim = (text_confidence > conf_threshold) & (sim < sim_threshold)
    return high_conf_low_sim.float().mean().item()
```

将 CMER 作为生产 KPI。按端点、提示词类型、客户监控。CMER 上升表明模型开始在某些输入分布上产生幻觉。

### 步骤四：玩具 VLM 分类器（可运行）

演示投影器的训练。伪"ViT 特征"输入；微型 LLM 风格 token 预测类别。

```python
class ToyVLM(nn.Module):
    def __init__(self, vit_dim=32, llm_dim=64, num_classes=5):
        super().__init__()
        self.projector = Projector(vit_dim, llm_dim, hidden=64)
        self.head = nn.Linear(llm_dim, num_classes)

    def forward(self, vision_tokens):
        projected = self.projector(vision_tokens)
        pooled = projected.mean(dim=1)
        return self.head(pooled)
```

可以在不到 200 步内对合成（特征，类别）对拟合——足以展示投影器模式有效。

## 生产使用

2026 年生产团队使用 VLM 的三种方式：

- **托管 API** — OpenAI Vision、Anthropic Claude Vision、Google Gemini Vision。零基础设施，供应商风险。
- **开源自托管** — 通过 `transformers` 和 `vllm` 使用 Qwen3-VL 或 InternVL3.5。完全控制，前期工作量更高。
- **领域微调** — 加载 Qwen2.5-VL-7B 或 LLaVA-1.6-7B，在 5k-50k 自定义样本上 LoRA 微调，用 `vllm` 或 `TGI` 服务。

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open("plot.png")},
        {"type": "text", "text": "What does this chart show?"},
    ],
}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
generated = model.generate(**inputs, max_new_tokens=256)
answer = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

`apply_chat_template` 隐藏了 `<image>` 占位符的分词；模型在内部处理合并。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| ViT-MLP-LLM | "VLM 模式" | 视觉编码器 + 投影器 + 语言模型；2026 年每个 VLM |
| 投影器（Projector） | "桥梁" | 2-4 层 MLP（或 Q-former），将视觉 token 映射到 LLM 嵌入空间 |
| DeepStack | "Qwen3-VL 特征技巧" | 多层 ViT 特征堆叠，而非仅用最后一层 |
| 图像 token（Image token） | "<image> 占位符" | 文本流中被投影视觉嵌入替换的特殊 token |
| CMER | "幻觉 KPI" | 跨模态错误率；文本置信度高但图像-文本相似度低时较高 |
| 视觉智能体（Visual agent） | "会点击的 VLM" | 操作 GUI（OSWorld、移动、Web）并发出工具调用的 VLM |
| Q-former | "固定数量 token 桥梁" | BLIP-2 风格投影器，生成固定数量的视觉查询 token |
| 对齐 / 预训练 / 指令调优 | "三个阶段" | 标准 VLM 训练管线 |

## 延伸阅读

- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [InternVL3.5 Advancing Open-Source Multimodal Models (arXiv 2508.18265)](https://arxiv.org/html/2508.18265v1)
- [LLaVA-Next series](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)
- [BentoML: Best Open-Source VLMs 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [MMMU: Multi-discipline Multimodal Understanding benchmark](https://mmmu-benchmark.github.io/)
- [VLMs in manufacturing (Robotics Tomorrow, March 2026)](https://www.roboticstomorrow.com/story/2026/03/when-machines-learn-to-see-like-experts-the-rise-of-vision-language-models-in-manufacturing/26335/)

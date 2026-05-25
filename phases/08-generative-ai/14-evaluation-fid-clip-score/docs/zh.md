# 评估——FID、CLIP 分数、人类偏好

> 每个生成模型排行榜都引用 FID、CLIP 分数以及来自人类偏好擂台的胜率。每个数字都有坚定的研究者可以操纵的失效模式。如果你不了解失效模式，就无法区分真正的改进和被操纵的结果。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 8 · 01（分类）、Phase 2 · 04（评估指标）
**时长：** 约 45 分钟

## 问题背景

生成模型根据*样本质量*和*条件遵循度*进行评判。两者都没有封闭形式的度量。你的模型必须生成 1 万张图像；某些标准必须给它们打分；你必须在不同模型家族、不同分辨率、不同架构之间信任这些数字。三种指标在 2014-2026 年间存活下来：

- **FID（Fréchet Inception Distance）。** 在 Inception 网络特征空间中真实分布和生成分布之间的距离。越低越好。
- **CLIP 分数。** 生成图像的 CLIP 图像嵌入与提示词的 CLIP 文本嵌入之间的余弦相似度。越高越好。衡量提示词遵循度。
- **人类偏好。** 将两个模型在相同提示词上进行正面比较，让人类（或 GPT-4 级别的模型）选出更好的，汇总为 Elo 分数。

你还会看到：IS（Inception 分数，大多已退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都修正了前一个的一个失效。

## 核心概念

![FID、CLIP 和偏好：三个轴，不同的失效模式](../assets/evaluation.svg)

### FID——样本质量

Heusel 等（2017）。步骤：

1. 提取 N 张真实图像和 N 张生成图像的 Inception-v3 特征（2048 维）。
2. 对每个池拟合高斯：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯之间的 Fréchet 距离。越低 = 分布越相似。

失效模式：
- **小 N 时有偏差。** FID 是特征分布上的均方——小 N 低估协方差，给出虚假的低 FID。始终使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的域（人脸、艺术、文字图像）产生无意义的 FID。使用特定域的特征提取器。
- **可被操纵。** 对 Inception 先验过拟合可以得到低 FID 而没有视觉质量提升。用 CMMD（下文）来检验。

### CLIP 分数——提示词遵循度

Radford 等（2021）。对生成图像 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 3 万张生成图像上取平均 → 在模型间可比较的标量。

失效模式：
- **CLIP 自身的盲点。** CLIP 的组合推理很弱（"红色立方体在蓝色球体上"经常失败）。模型可以在 CLIP 分数上表现良好，而实际上并没有遵循复杂提示词。
- **短提示词偏差。** 短提示词在自然界中有更多 CLIP 图像匹配。长提示词机械地得到更低的 CLIP 分数。
- **提示词操纵。** 在提示词中包含"高质量、4k、杰作"会在不改善图像-文本绑定的情况下提高 CLIP 分数。

CMMD（Jayasumana 等，2024）修正了其中一些问题：使用 CLIP 特征而非 Inception 特征，使用最大均值差异而非 Fréchet。更善于检测细微质量差异。

### 人类偏好——真实基准

选择一个提示词池。用模型 A 和模型 B 生成。将配对展示给人类（或强大的 LLM 评判者）。将胜利汇总为 Elo 或 Bradley-Terry 分数。基准测试：

- **PartiPrompts（谷歌）**：1,600 个多样化提示词，12 个类别。
- **HPSv2**：107k 人类注释，广泛用作自动代理。
- **ImageReward**：137k 提示词-图像偏好对，MIT 许可证。
- **PickScore**：在 Pick-a-Pic 260 万偏好数据上训练。
- **聊天机器人竞技场风格的图像擂台**：https://imagearena.ai/ 等。

失效模式：
- **评判者方差。** 非专家与专家有不同偏好。两者都要用。
- **提示词分布。** 精心挑选的提示词偏向某个家族。始终记录。
- **LLM 评判者奖励黑客。** GPT-4 评判者会被漂亮但错误的输出愚弄。用人类评判进行三角验证。

## 综合使用

生产评估报告应包括：

1. 针对保留真实分布的 10,000-30,000 个样本的 FID（样本质量）。
2. 相同样本相对于其提示词的 CLIP 分数 / CMMD（遵循度）。
3. 与前一个模型在盲测擂台中的胜率（整体偏好）。
4. 失效模式分析：50 个随机抽样输出，标记已知问题（手部解剖、文字渲染、对象数量一致性）。

任何单一指标都是谎言。三个相互印证的指标 + 定性审查才是声明。

## 动手实现

`code/main.py` 在合成"特征向量"上实现 FID、CLIP 分数类似物和 Elo 聚合（我们使用 4 维向量作为 Inception 特征的替代）。你将看到：

- 小 N 和大 N 下的 FID 计算——偏差。
- "CLIP 分数"作为特征池之间的余弦相似度。
- 来自合成偏好流的 Elo 更新规则。

### 步骤一：四行实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤二：CLIP 风格余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤三：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 常见陷阱

- **N=1000 时的 FID。** N<10k 时启发式不可靠。报告低 N FID 的论文在操纵数字。
- **跨分辨率比较 FID。** Inception 的 299×299 缩放改变了特征分布。只在匹配分辨率下比较。
- **只报告一个种子。** 至少运行 3 个种子。报告标准差。
- **通过负向提示词提高 CLIP 分数。** 某些流水线通过对提示词过拟合来提高 CLIP。检查视觉饱和度。
- **来自提示词重叠的 Elo 偏差。** 如果两个模型在训练时都看到了基准提示词，Elo 就没有意义。使用保留的提示词集。
- **人类评估众包偏差。** Prolific、MTurk 注释者倾向于更年轻/技术友好。与招募的艺术/设计专家混合使用。

## 生产使用

2026 年生产评估协议：

| 支柱 | 最低要求 | 推荐 |
|------|---------|------|
| 样本质量 | 10k 样本 vs 保留真实数据的 FID | + 5k 的 CMMD + 每类别子集 FID |
| 提示词遵循度 | 30k 的 CLIP 分数 | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 盲测配对 vs 基线 | + 2000 配对人类 + LLM 评判者 + 聊天机器人擂台 |
| 失效分析 | 50 个手动标记 | 500 个手动标记 + 自动安全分类器 |

报告中包含全部四个支柱 = 有效声明。任何一个单独 = 营销。

## 上手实践

保存 `outputs/skill-eval-report.md`。该技能接受新模型检查点 + 基线，输出完整评估计划：样本量、指标、失效模式探针、签字标准。

## 练习

1. **简单。** 运行 `code/main.py`。在相同合成分布上比较 N=100 vs N=1000 时的 FID。报告偏差幅度。
2. **中等。** 从合成 CLIP 风格特征实现 CMMD（见 Jayasumana 等，2024 的公式）。比较其对质量差异的灵敏度 vs FID。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 的子集中取 1000 个图像-提示词对，在偏好数据上微调一个小型 CLIP 评分器，并测量其与保留集的一致性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| FID | "Fréchet Inception Distance" | 真实与生成 Inception 特征的高斯拟合之间的 Fréchet 距离。 |
| CLIP 分数（CLIP score） | "文本-图像相似度" | CLIP 图像和文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代" | CLIP 特征 MMD；偏差更小，无高斯假设。 |
| IS | "Inception 分数" | Exp KL(p(y|x) || p(y))；与现代模型相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "学习到的偏好代理" | 在人类偏好上训练的小型模型；用作自动评判者。 |
| Elo | "国际象棋评级" | 配对胜利的 Bradley-Terry 聚合。 |
| PartiPrompts | "基准提示词集" | 谷歌精选的 1,600 个提示词，12 个类别。 |
| FD-DINO | "自监督替代" | 使用 DINOv2 特征的 FD；对 ImageNet 域外更好。 |

## 生产注意：评估也是推理工作负载

在 1 万个样本上运行 FID 意味着生成 1 万张图像。对于单块 L4 上 1024² 的 50 步 SDXL base，这大约需要 11 小时的单请求推理。评估预算是真实存在的，框架与离线推理场景完全相同（最大化吞吐量，忽略 TTFT）：

- **大批处理，忘掉延迟。** 离线评估 = 在适合内存的最大批大小下进行静态批处理。在 80GB H100 上使用 `num_images_per_prompt=8` 的 `pipe(...).images` 运行速度比单请求快 4-6 倍。
- **缓存真实特征。** 对真实参考集的 Inception（FID）或 CLIP（CLIP 分数、CMMD）特征提取*运行一次*，存储为 `.npz`。不要在每次评估时重新计算。

对于 CI/回归门控：每个 PR 在 500 样本子集上运行 FID + CLIP 分数（约 30 分钟）；每晚运行完整的 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式综述

---
name: fairness-criterion
description: 识别声明所援引的公平性标准，并审计其相关假设。
version: 1.0.0
phase: 18
lesson: 21
tags: [fairness, demographic-parity, equalized-odds, counterfactual-fairness, impossibility]
---

针对公平性声明或政策，识别所援引的标准、声明所依赖的假设，以及不可能定理对其余标准的影响。

产出内容：

1. 标准识别。将声明标记为以下之一：人口统计均等性（demographic parity）、均等化优势（equalized odds）、条件使用准确率均等性（conditional use accuracy equality）、个体公平性（individual fairness）、反事实公平性（counterfactual fairness）。模糊声明须在继续分析前先予以明确。
2. 基率审计。部署中各群体的基率是多少？在基率不均等的情况下，Chouldechova / KMR 2017 不可能定理适用：没有任何模型能同时满足所有三个群体标准。
3. 因果有向无环图（DAG）依赖性。若声明为反事实公平性，所用因果 DAG 是什么？反事实公平性的可信度取决于 DAG 本身的合理性，缺乏 DAG 则声明无效。
4. 相似性度量。若声明为个体公平性，相似性度量 d 是什么？选择与任务相关，属于政策决定而非统计决定。
5. 干预合法性。若声明使用了反事实推理，是否涉及对受保护属性的干预？若是，考虑使用回溯反事实（arXiv:2401.13935）以规避法律问题。

硬性拒绝条件：
- 任何未明确标准的"公平"声明。
- 任何在基率不均等情况下主张"满足所有公平标准"但未提及 Chouldechova / KMR 2017 的声明。
- 任何缺乏已发布因果 DAG 的反事实公平性声明。

拒绝规则：
- 若用户询问哪个公平性标准"最正确"，拒绝排名，解释这是一个政策选择。
- 若用户询问模型是否"公平"，拒绝二元声明；公平性是相对于标准而言的。

输出：一页审计报告，填充上述五个部分，在适用时标出不可能定理，并指明声明中隐含的政策选择。视情况分别引用 Dwork 等人 2012、Kusner 等人 2017、Chouldechova 2017 各一次。

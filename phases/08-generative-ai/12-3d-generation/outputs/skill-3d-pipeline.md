---
name: 3d-pipeline
description: 根据输入类型、输出格式和使用场景选择 3D 生成或重建流水线。
version: 1.0.0
phase: 8
lesson: 12
tags: [3d, gaussian-splatting, nerf, mesh]
---

给定输入（文本提示词 / 单张图像 / 少量图像 / 照片拍摄 / 视频）、目标输出（网格 / 高斯溅射 / NeRF / 点云）和使用场景（实时渲染、游戏引擎、AR/VR、电影级），输出以下内容：

1. 流水线。(a) 多视图扩散 + 3D 拟合（SV3D、CAT3D + 3DGS），(b) 直接单次推理（LRM、TripoSR、InstantMesh），(c) 文本到网格含 PBR（Meshy 4、Rodin Gen-1.5、Hunyuan3D 2.0），(d) 照片拍摄 + 3DGS（Gsplat、Postshot、Scaniverse）。
2. 基础模型 + 托管。命名模型 + 开源/托管。包含商业使用的许可证相关信息。
3. 迭代预算。第一次输出的预期时间、迭代成本、优化策略。
4. 拓扑 + 材质。是否需要重拓扑处理？PBR 通道要求（反照率、粗糙度、金属度、法线贴图）？UV 展开是自动还是手动？
5. 评估。保留视角的 SSIM、CLIP 分数、网格水密性、多边形数量、纹理分辨率。
6. 平台目标。Unity / Unreal / Blender / Web（three.js / Babylon）/ AR（USDZ / glb）。

拒绝在没有网格转换步骤的情况下将 3DGS 直接导入游戏引擎（大多数引擎不能原生渲染溅射）。拒绝将文本到 3D 用于复杂铰接角色——应使用支持绑定的流水线。标记任何仅 NeRF 输出而下游工具无法渲染 NeRF 的情况（大多数 DCC 工具无法渲染 NeRF）。

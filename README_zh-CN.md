# ViT-SJTU-captcha

<div align="center">

[English](README.md) | 简体中文

</div>

## 介绍

这是一个**专用于 SJTU Jaccount**的**高效**验证码解决器。其主要为**低 RAM** 与 **CPU 推理**场景而设计。

它基于 ViT（Vision Transformer）架构和一系列数据预处理技巧。由于 SJTU Jaccount 验证码较为简单，故其并未基于 OCR 技术，而是通过数据预处理技巧转换为了**字符分类**问题。

## 模型

现在最新版本模型为 [v3](v3)。全部版本模型性能对比如下：

| 模型 | 参数量 | 字符分类准确率 | 验证码准确率 | 字符分类 FPS | 验证码 FPS | 预处理 FPS | 权重 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [v3 in python](train/acc_fps_test.py) | 5.57K | 99.96% (243179/243285) | 99.78% (54063/54180) | 3482.79 | 688.81 | 6190.90 | [权重](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v3/v3.pth) |
| [v3 in cpp](deploy/acc_fps_test.py) | 5.57K | 99.96% (243178/243285) | 99.78% (54062/54180) | 20613.91 | 3655.02 | 17980.78 | [权重](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v3/v3.gguf) |
| [v2](v2) | 211.53K | 99.99% (240296/240301) | 98.82% (53543/54180) | 1174.71 | 98.43 | 157.62 | [权重](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v2) |
| [v1](v1) | 873.25K | 99.99% (240296/240301) | 98.82% (53543/54180) | 1293.20 | 85.02 | 120.53 | [权重](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model) |

测试环境为 PyTorch v2.6.0，Intel(R) Core(TM) i7-13700，16GB RAM。

> \[!IMPORTANT\]
> 当预处理得到的图片数目（即为字符个数）与真实值不同时，本次字符预测结果将**不会**计入到字符分类准确率的计算过程中。
>
> ```python
> def metric(preds, gts):
>     assert len(preds) == len(gts)
>     captcha_correct, captcha_total = 0, 0
>     char_correct, char_total = 0, 0
>
>     for (pred, gt) in zip(preds, gts):
>         captcha_total += 1
>         if pred == gt:
>             captcha_correct += 1
>
>         if len(pred) == len(gt):
>             char_total += len(gt)
>             for (p, g) in zip(pred, gt):
>                 if p == g:
>                     char_correct += 1
> ```

## 更新日志

<details>
<summary>更新日志（点击展开）</summary>

- 2025.03.05：ViT-SJTU-captcha v3 发布。切图算法的准确率进一步提升，模型参数量进一步降低。此外还提供了 cpp 部署方式，降低了对 RAM 的需求，并大幅提高了 FPS。

- 2024.12.25：ViT-SJTU-captcha v2 发布。其参考了 LLaMA 对于 Transformer 的改进（SwiGLU、GQA 等），整体架构与 Qwen2.5-VL 的 Vision Tower 相似，但早于 Qwen2.5-VL 发布。在不改变准确率的情况下，优化了模型参数量与 RAM 需求。

- 2023.02.26：ViT-SJTU-captcha v1 发布。其基于 vit_pytorch v1.0.0。模型架构为标准 ViT。

</details>

## 致谢

- [LightQuantumArchive](https://github.com/LightQuantumArchive) 在 [jaccount-captcha-solver](https://github.com/LightQuantumArchive/jaccount-captcha-solver) 项目中开源了有标注的 Jaccount 验证码数据集。
- [Gennadiyev](https://github.com/Gennadiyev) 提出需求、实现 [v3 版 `segment` 逻辑](v3/train/segment.py)、协助 debug [cpp 部署逻辑](v3/deploy/csrc/vit_captcha.cpp)。

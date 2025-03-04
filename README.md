# ViT-SJTU-captcha

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

This is an **efficient** captcha solver **specifically designed for SJTU Jaccount**. It is mainly tailored for **low RAM** and **CPU inference** scenarios.

It is based on the ViT (Vision Transformer) and a series of data preprocessing tricks. Since the SJTU Jaccount captcha is relatively simple, it does not rely on OCR technology but instead converts the problem into a **character classification** task using data preprocessing tricks.

## Models

The latest version of the model is [v2](v2). The performance comparison of all versions of the models is as follows:

| Model | #Parameters | Character Classification Accuracy | Captcha Accuracy | Character Classification FPS | Captcha FPS FPS | Preprocessing FPS | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [v2](v2) | 211.53K | 99.99% (240296/240301) | 98.82% (53543/54180) | 1174.71 | 98.43 | 157.62 | [weights](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v2) |
| [v1](v1) | 873.25K | 99.99% (240296/240301) | 98.82% (53543/54180) | 1293.20 | 85.02 | 120.53 | [weights](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model) |

Testing environment: PyTorch v2.6.0, Intel(R) Core(TM) i7-13700, 16GB RAM.

> \[!IMPORTANT\]
> When the number of images obtained from preprocessing (i.e., the number of characters) differs from the actual value, the character prediction results for this instance **will not** be included in the calculation of character classification accuracy.
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

## Changelog

<details>
<summary>Changelog (Click to expand)</summary>

- 2024.12.25: ViT-SJTU-captcha v2 is released. It references LLaMA's improvements to the Transformer (SwiGLU, GQA, etc.). Although the overall architecture is similar to Qwen2.5-VL's Vision Tower, it is released earlier than Qwen2.5-VL. Without changing the accuracy, it optimizes the model parameter count and RAM requirements.

- 2023.02.26: ViT-SJTU-captcha v1 is released. It is based on vit_pytorch v1.0.0. The model architecture is standard ViT.

</details>

## Acknowledgments

- [LightQuantumArchive](https://github.com/LightQuantumArchive) open-sourced a labeled Jaccount captcha dataset in the [jaccount-captcha-solver](https://github.com/LightQuantumArchive/jaccount-captcha-solver) project.
- [Gennadiyev](https://github.com/Gennadiyev) proposed the requirement, implemented the [`segment` of v3](v3/train/segment.py) and helped me debug the [cpp deploy](v3/deploy/csrc/vit_captcha.cpp).

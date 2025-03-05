# v3

<div align="center">

[English](README.md) | 简体中文

</div>

v3 主要包括两部分，分别是基于 PyTorch 的训练，和基于 [GGML](https://github.com/ggml-org/ggml)（CPP）的部署。相关指标对比如下：

| 模型 | 参数量 | 字符分类准确率 | 验证码准确率 | 字符分类 FPS | 验证码 FPS | 预处理 FPS | 权重 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [v3 in python](train/acc_fps_test.py) | 5.57K | 99.96% (243179/243285) | 99.78% (54063/54180) | 3482.79 | 688.81 | 6190.90 | [权重](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v3/v3.pth) |
| [v3 in cpp](deploy/acc_fps_test.py) | 5.57K | 99.96% (243178/243285) | 99.78% (54062/54180) | 20613.91 | 3655.02 | 17980.78 | [权重](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v3/v3.gguf) |

测试环境: PyTorch v2.6.0, Intel(R) Core(TM) i7-13700, 16GB RAM.

## 训练

训练方面，相比于 v2，v3 引入了更多降低模型参数量并保证准确率，以及提升 FPS 的操作。具体地：

- `segment`

  作为将验证码任务分解为字符分类任务的核心功能，v3 的 `segment` 的逻辑有以下改动：

  - 增加了对于**斜体字母**的处理
  - **移除**了在处理字母 i 和字母 j 时对于 `label` 的需求。
  - 此外，最重要的是，v3 的 `segment` 的逻辑**使用 SVM** 有效解决了先前逻辑无法解决的情况。（如斜体 w 与非斜体的 z，会出现 w 与 z 的起笔连起来的情况，导致 contour 错判）。

- `model`

  `model` 部分相比于 `v2` 并无过多改进，主要变动如下：

  - 引入了**旋转位置编码（RoPE）**。
  - 将 GQA 进一步优化为了 MQA。
  - 进一步减少了模型层数与嵌入维度。

- `train`

  `train` 部分逻辑无明显变动，具体改动如下：

  - 引入了更复杂的学习率策略。（在训练的前 20%，学习率线性升高作为 warmup；在后 80%，学习率按照余弦降低。）

## 部署

为了进一步降低 RAM 需求（移除 torch 依赖），并且提升 FPS，我们采用了基于 OpenCV+GGML 的部署策略。

其中 OpenCV 负责实现 `segment` 逻辑（包括 SVM）；GGML 负责在 CPU 上完成模型的高效推理。

### 部署方法

首先构建 `libvit_captcha.so`

```bash
cd /path/to/ViT-SJTU-captcha/v3/deploy
mkdir build && cd build
cmake ../csrc && make
cd ..
```

然后，[测试文件](deploy/acc_fps_test.py) 中提供了 `libvit_captcha.so` 的相关调用方法。

具体地，`libvit_captcha.so` 提供了以下接口：

| 接口名 | 功能 | 输入类型（CPP）| 输入类型（Python）| 输出类型（CPP）| 输出类型（Python）|
| --- | --- | --- | --- | --- | --|
| `load_vit_captcha_v3` | 加载模型 | `char*` | `ctypes.c_char_p` | `ViTCaptchaV3Model`\* | `ctypes.c_void_p` |
| `free_vit_captcha_v3` | 释放模型 | `ViTCaptchaV3Model`\* | `ctypes.c_void_p` | `void` | `None` |
| `build_vit_captcha_v3_graph` | 建立计算图 | `ViTCaptchaV3Model`\* | `ctypes.c_void_p` | `ViTCaptchaV3*` | `ctypes.c_void_p` |
| `predict` | 推理过程 | `ViTCaptchaV3*`, `char*` | `ctypes.c_void_p`, `ctypes.c_char_p` | `char*` | `ctypes.c_char_p` |
| `predict_with_timer` | 计时推理 | `ViTCaptchaV3*`, `char*` | `ctypes.c_void_p`, `ctypes.c_char_p` | `double*` | `ctypes.POINTER(ctypes.c_double)` |

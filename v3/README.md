# v3

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

v3 mainly consists of two parts: training based on PyTorch and deployment based on [GGML](https://github.com/ggml-org/ggml) (CPP). The following table compares the relevant metrics:

| Model | #Parameters | Character Classification Accuracy | Captcha Accuracy | Character Classification FPS | Captcha FPS FPS | Preprocessing FPS | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [v3 in python](train/acc_fps_test.py) | 5.57K | 99.96% (243179/243285) | 99.78% (54063/54180) | 3482.79 | 688.81 | 6190.90 | [Weights](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v3/v3.pth) |
| [v3 in cpp](deploy/acc_fps_test.py) | 5.57K | 99.96% (243178/243285) | 99.78% (54062/54180) | 20613.91 | 3655.02 | 17980.78 | [Weights](https://github.com/fanqiNO1/ViT-SJTU-captcha/releases/tag/model-v3/v3.gguf) |

Testing environment: PyTorch v2.6.0, Intel(R) Core(TM) i7-13700, 16GB RAM.

## Training

In terms of training, compared to v2, v3 introduces more methods aimed at reducing model parameters while maintaining accuracy and improving FPS. Specifically:

- `segment`

  As the core functionality that breaks down the captcha task into character classification tasks, the logic of v3's `segment` has been updated as follows:

  - Added handling for **italic letters**.
  - **Removed** the requirement for `label` when processing letters 'i' and 'j'.
  - Most importantly, v3's `segment` logic now **uses SVM**, effectively solving issues that were previously unresolved (e.g., cases where italic 'w' and non-italic 'z' have their strokes connected, leading to incorrect contour judgments).

- `model`

  Compared to `v2`, there are no significant improvements in the `model` section, with the main changes being:

  - Introduced **rotary position positional encoding (RoPE)**.
  - Further optimized GQA into MQA.
  - Reduced the number of model layers and embedding dimensions.

- `train`

  There are no major changes in the `train` logic, with specific updates including:

  - A more complex learning rate strategy has been introduced. (During the first 20% of training, the learning rate increases linearly as a warm-up; during the remaining 80%, it decreases according to a cosine schedule.)

## Deployment

To further reduce RAM requirements (by removing the torch dependency) and improve FPS, we adopt a deployment strategy based on OpenCV + GGML.

OpenCV is responsible for implementing the `segment` logic (including SVM), while GGML handles efficient model inference on the CPU.

### Deployment Method

First, build `libvit_captcha.so`:

```bash
cd /path/to/ViT-SJTU-captcha/v3/deploy
mkdir build && cd build
cmake ../csrc && make
cd ..
```

Then, the [test file](deploy/acc_fps_test.py) provides methods for calling `libvit_captcha.so`.

Specifically, `libvit_captcha.so` offers the following interfaces:

| Interface Name | Functionality | Input Type (CPP) | Input Type (Python) | Output Type (CPP) | Output Type (Python) |
| --- | --- | --- | --- | --- | --- |
| `load_vit_captcha_v3` | Load the model | `char*` | `ctypes.c_char_p` | `ViTCaptchaV3Model*` | `ctypes.c_void_p` |
| `free_vit_captcha_v3` | Release the model | `ViTCaptchaV3Model*` | `ctypes.c_void_p` | `void` | `None` |
| `build_vit_captcha_v3_graph` | Build computation graph | `ViTCaptchaV3Model*` | `ctypes.c_void_p` | `ViTCaptchaV3*` | `ctypes.c_void_p` |
| `predict` | Inference process | `ViTCaptchaV3*`, `char*` | `ctypes.c_void_p`, `ctypes.c_char_p` | `char*` | `ctypes.c_char_p` |
| `predict_with_timer` | Timed inference | `ViTCaptchaV3*`, `char*` | `ctypes.c_void_p`, `ctypes.c_char_p` | `double*` | `ctypes.POINTER(ctypes.c_double)` |

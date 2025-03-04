#pragma once

#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "segment.h"


struct ViTCaptchaV3Hparams {
    const char* name = "ViTCaptchaV3";
    int image_size = 28;
    int num_layers = 2;
    int num_classes = 26;
    int patch_size = 7;
    int in_channels = 1;
    int hidden_size = 16;
    int num_attention_heads = 4;
    int num_key_value_heads = 1;
    int intermediate_size = 32;
    const char* act_fn = "SiLU";
    float eps = 1e-6;
    int num_patches() const { return (image_size / patch_size) * (image_size / patch_size); }
    int head_dim() const { return hidden_size / num_attention_heads; }
    int seqlen() const { return num_patches() + 1; }
    float attn_scaling() const { return 1.0f / sqrtf(head_dim()); }
};


struct ViTCaptchaV3EncoderLayer {
    // attention
    ggml_tensor* attn_q_proj_weight;
    ggml_tensor* attn_q_proj_bias;
    ggml_tensor* attn_k_proj_weight;
    ggml_tensor* attn_k_proj_bias;
    ggml_tensor* attn_v_proj_weight;
    ggml_tensor* attn_v_proj_bias;
    ggml_tensor* attn_o_proj_weight;
    // norm1
    ggml_tensor* norm1_weight;
    // mlp
    ggml_tensor* mlp_gate_proj_weight;
    ggml_tensor* mlp_up_proj_weight;
    ggml_tensor* mlp_down_proj_weight;
    // norm2
    ggml_tensor* norm2_weight;
};


struct ViTCaptchaV3Model {
    ViTCaptchaV3Hparams hparams = ViTCaptchaV3Hparams();

    ggml_tensor* cls_token;
    ggml_tensor* cls_token_index;
    ggml_tensor* input_position;
    ggml_tensor* patch_embed_proj_weight;
    std::vector<ViTCaptchaV3EncoderLayer> layers;
    ggml_tensor* head_weight;
    ggml_tensor* head_bias;

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context* ctx = nullptr;
};


struct ViTCaptchaV3 {
    const ViTCaptchaV3Model* model;
    ggml_cgraph* graph;
    bool is_allocated = false;
};


extern "C" {
    ViTCaptchaV3Model* load_vit_captcha_v3(const char* model_path);
    void free_vit_captcha_v3(ViTCaptchaV3Model* model);
    ViTCaptchaV3* build_vit_captcha_v3_graph(const ViTCaptchaV3Model* model);
    char* predict(ViTCaptchaV3* vit_captcha_v3, const char* image_path);
    double* predict_with_timer(ViTCaptchaV3* vit_captcha_v3, const char* image_path);
}
void print_tensor_shape(const char* name, ggml_tensor* tensor);

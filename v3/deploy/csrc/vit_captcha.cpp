#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "segment.h"
#include "vit_captcha.h"


void print_tensor_shape(const char* name, ggml_tensor* tensor) {
    printf("%s: (%ld, %ld, %ld, %ld) in type %d\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->type);
}


ViTCaptchaV3Model* load_vit_captcha_v3(const char* model_path) {
    ViTCaptchaV3Model* model = new ViTCaptchaV3Model;
    // load gguf
    gguf_init_params params = {true, &model->ctx}; // no_alloc = true, ctx = &model->ctx
    gguf_context* ctx_gguf = gguf_init_from_file(model_path, params);
    if (!ctx_gguf) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        exit(1);
    }
    // alloc
    model->buffer = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);
    if (!model->buffer) {
        fprintf(stderr, "ggml_backend_alloc_ctx_tensors() failed\n");
        gguf_free(ctx_gguf);
        exit(1);
    }
    // load tensors
    try {
        model->cls_token = ggml_get_tensor(model->ctx, "cls_token");
        model->cls_token_index = ggml_get_tensor(model->ctx, "cls_token_index");
        model->input_position = ggml_get_tensor(model->ctx, "input_position");
        model->patch_embed_proj_weight = ggml_get_tensor(model->ctx, "patch_embed.proj.weight");
        model->layers.resize(model->hparams.num_layers);
        for (int i = 0; i < model->hparams.num_layers; i++) {
            // layers.0.attn.q_proj.weight
            std::string prefix = "layers." + std::to_string(i) + ".";
            model->layers[i].attn_q_proj_weight = ggml_get_tensor(model->ctx, (prefix + "attn.q_proj.weight").c_str());
            model->layers[i].attn_q_proj_bias = ggml_get_tensor(model->ctx, (prefix + "attn.q_proj.bias").c_str());
            model->layers[i].attn_k_proj_weight = ggml_get_tensor(model->ctx, (prefix + "attn.k_proj.weight").c_str());
            model->layers[i].attn_k_proj_bias = ggml_get_tensor(model->ctx, (prefix + "attn.k_proj.bias").c_str());
            model->layers[i].attn_v_proj_weight = ggml_get_tensor(model->ctx, (prefix + "attn.v_proj.weight").c_str());
            model->layers[i].attn_v_proj_bias = ggml_get_tensor(model->ctx, (prefix + "attn.v_proj.bias").c_str());
            model->layers[i].attn_o_proj_weight = ggml_get_tensor(model->ctx, (prefix + "attn.o_proj.weight").c_str());
            model->layers[i].norm1_weight = ggml_get_tensor(model->ctx, (prefix + "norm1.weight").c_str());
            model->layers[i].mlp_gate_proj_weight = ggml_get_tensor(model->ctx, (prefix + "mlp.gate_proj.weight").c_str());
            model->layers[i].mlp_up_proj_weight = ggml_get_tensor(model->ctx, (prefix + "mlp.up_proj.weight").c_str());
            model->layers[i].mlp_down_proj_weight = ggml_get_tensor(model->ctx, (prefix + "mlp.down_proj.weight").c_str());
            model->layers[i].norm2_weight = ggml_get_tensor(model->ctx, (prefix + "norm2.weight").c_str());
        }
        model->head_weight = ggml_get_tensor(model->ctx, "head.weight");
        model->head_bias = ggml_get_tensor(model->ctx, "head.bias");
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        gguf_free(ctx_gguf);
        exit(1);
    }
    // prepare backend buffer
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", model_path);
        gguf_free(ctx_gguf);
        exit(1);
    }
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < n_tensors; i++) {
        const char* tensor_name = gguf_get_tensor_name(ctx_gguf, i);
        ggml_tensor* tensor = ggml_get_tensor(model->ctx, tensor_name);
        int offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        std::vector<int> buffer(ggml_nbytes(tensor));
        if (fseek(f, offset, SEEK_SET) != 0) {
            fprintf(stderr, "Failed to fseek %s\n", tensor_name);
            gguf_free(ctx_gguf);
            fclose(f);
            exit(1);
        }
        if (fread(buffer.data(), 1, buffer.size(), f) != buffer.size()) {
            fprintf(stderr, "Failed to fread %s\n", tensor_name);
            gguf_free(ctx_gguf);
            fclose(f);
            exit(1);
        }
        ggml_backend_tensor_set(tensor, buffer.data(), 0, buffer.size());
    }
    fclose(f);
    gguf_free(ctx_gguf);
    return model;
}


void free_vit_captcha_v3(ViTCaptchaV3Model* model) {
    ggml_backend_buffer_free(model->buffer);
    ggml_backend_free(model->backend);
    ggml_free(model->ctx);
    delete model;
}


ViTCaptchaV3* build_vit_captcha_v3_graph(const ViTCaptchaV3Model* model) {
    static unsigned long int buffer_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<int> buffer(buffer_size);

    ggml_init_params params = {buffer_size, buffer.data(), true};
    // mem_size = buffer_size, mem_buffer = buffer.data(), no_alloc = true
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    // set input
    // since the number of images is 5 at maximum, we set the batch size to 5
    int batch_size = 5;
    ggml_backend_cpu_set_n_threads(model->backend, 1);
    // (b, c, h, w)? (w, h, c, b)!
    ggml_tensor* images = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 28, 28, 1, batch_size);
    ggml_set_name(images, "images");
    ggml_set_input(images);
    // forward pass
    ggml_tensor* hidden_states;
    {
        // patch embed; stride = kernel, padding = 0
        // 28, 28, 1, 5 -> 4, 4, 16, 5 -> 16, 4, 4, 5 -> 16, 16, 5, 1
        // (w, h, c, b) -> (n_w, n_h, d, b) -> (d, n_w, n_h, b) -> (d, n, b)
        hidden_states = ggml_conv_2d_sk_p0(ctx, model->patch_embed_proj_weight, images);
        hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 1, 2, 0, 3));
        hidden_states = ggml_reshape_4d(ctx, hidden_states, model->hparams.hidden_size, model->hparams.num_patches(), batch_size, 1);
        // cls token; 16, 1, 1, 1 -> 16, 1, 5, 1
        // 16, 16, 5, 1 -> 16, 17, 5, 1
        ggml_tensor* cls_token = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, model->hparams.hidden_size, 1, batch_size, 1);
        cls_token = ggml_repeat(ctx, model->cls_token, cls_token);
        hidden_states = ggml_concat(ctx, cls_token, hidden_states, 1);
        // encoder layers
        for (auto layer : model->layers) {
            ggml_tensor* identity = hidden_states;
            // norm1
            hidden_states = ggml_rms_norm(ctx, hidden_states, model->hparams.eps);
            hidden_states = ggml_mul(ctx, hidden_states, layer.norm1_weight);
            // attention
            // 16, 17, 5, 1 -> 16, 17, 5, 1 or 4, 17, 5, 1 -> 4, 4, 17, 5
            // (d, n, b) -> (n_h*d_h, n, b) -> (d_h, n_h, n, b)
            ggml_tensor* q = ggml_mul_mat(ctx, layer.attn_q_proj_weight, hidden_states);
            q = ggml_add(ctx, q, layer.attn_q_proj_bias);
            q = ggml_reshape_4d(ctx, q, model->hparams.head_dim(), model->hparams.num_attention_heads, model->hparams.seqlen(), batch_size);
            ggml_tensor* k = ggml_mul_mat(ctx, layer.attn_k_proj_weight, hidden_states);
            k = ggml_add(ctx, k, layer.attn_k_proj_bias);
            k = ggml_reshape_4d(ctx, k, model->hparams.head_dim(), model->hparams.num_key_value_heads, model->hparams.seqlen(), batch_size);
            ggml_tensor* v = ggml_mul_mat(ctx, layer.attn_v_proj_weight, hidden_states);
            v = ggml_add(ctx, v, layer.attn_v_proj_bias);
            v = ggml_reshape_4d(ctx, v, model->hparams.head_dim(), model->hparams.num_key_value_heads, model->hparams.seqlen(), batch_size);
            // apply rotary position encoding
            q = ggml_rope_ext(ctx, q, model->input_position, nullptr, model->hparams.head_dim(), GGML_ROPE_TYPE_NEOX, model->hparams.seqlen(), 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, model->input_position, nullptr, model->hparams.head_dim(), GGML_ROPE_TYPE_NEOX, model->hparams.seqlen(), 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f);
            // continue attention
            // 4, 4, 17, 5 -> 4, 17, 4, 5 -> 17, 17, 4, 5
            // (d_h, n_h, n, b) -> (d_h, n, n_h, b)
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
            // (n, n, n_h, b)
            ggml_tensor* attn = ggml_mul_mat(ctx, k, q);
            attn = ggml_scale(ctx, attn, model->hparams.attn_scaling());
            attn = ggml_soft_max(ctx, attn);
            // (d_h, n_h, n, b) -> (n, d_h, n_h, b)
            v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
            hidden_states = ggml_mul_mat(ctx, v, attn);
            hidden_states = ggml_permute(ctx, hidden_states, 0, 2, 1, 3);
            hidden_states = ggml_cont_4d(ctx, hidden_states, model->hparams.hidden_size, model->hparams.seqlen(), batch_size, 1);
            hidden_states = ggml_mul_mat(ctx, layer.attn_o_proj_weight, hidden_states);
            // shortcut
            hidden_states = ggml_add(ctx, hidden_states, identity);
            ggml_tensor* identity2 = hidden_states;
            // norm2
            hidden_states = ggml_rms_norm(ctx, hidden_states, model->hparams.eps);
            hidden_states = ggml_mul(ctx, hidden_states, layer.norm2_weight);
            // mlp
            ggml_tensor* gate = ggml_mul_mat(ctx, layer.mlp_gate_proj_weight, hidden_states);
            gate = ggml_silu(ctx, gate);
            ggml_tensor* up = ggml_mul_mat(ctx, layer.mlp_up_proj_weight, hidden_states);
            ggml_tensor* down = ggml_mul(ctx, gate, up);
            hidden_states = ggml_mul_mat(ctx, layer.mlp_down_proj_weight, down);
            // shortcut
            hidden_states = ggml_add(ctx, hidden_states, identity2);
        }
        // head
        ggml_tensor* cls_token_index = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, batch_size, 1, 1);
        cls_token_index = ggml_repeat(ctx, model->cls_token_index, cls_token_index);
        hidden_states = ggml_get_rows(ctx, hidden_states, cls_token_index);
        hidden_states = ggml_mul_mat(ctx, model->head_weight, hidden_states);
        hidden_states = ggml_add(ctx, hidden_states, model->head_bias);
        hidden_states = ggml_reshape_4d(ctx, hidden_states, model->hparams.num_classes, batch_size, 1, 1);
        hidden_states = ggml_argmax(ctx, hidden_states);
    }
    // set outout
    ggml_set_name(hidden_states, "hidden_states");
    ggml_set_output(hidden_states);
    // build graph
    ggml_build_forward_expand(graph, hidden_states);
    // combine
    ViTCaptchaV3* vit_captcha_v3 = new ViTCaptchaV3;
    vit_captcha_v3->model = model;
    vit_captcha_v3->graph = graph;
    return vit_captcha_v3;
}


char* predict(ViTCaptchaV3* vit_captcha_v3, const char* image_path) {
    // segment the image
    SegmentedImages segmented_images = segment(image_path);
    // alloc
    if (!vit_captcha_v3->is_allocated) {
        static ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(vit_captcha_v3->model->backend));
        if (!ggml_gallocr_alloc_graph(alloc, vit_captcha_v3->graph)) {
            fprintf(stderr, "ggml_gallocr_alloc_graph() failed\n");
            exit(1);
        }
        vit_captcha_v3->is_allocated = true;
    }
    // build input
    ggml_tensor* images = ggml_graph_get_tensor(vit_captcha_v3->graph, "images");
    for (int i = 0; i < segmented_images.num_images; i++) {
        std::vector<float> image(28 * 28);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                image[y * 28 + x] = segmented_images.images[i].at<uchar>(y, x) / 255.0f;
            }
        }
        ggml_backend_tensor_set(images, image.data(), 28*28*i*sizeof(float), 28*28*sizeof(float));
    }
    if (segmented_images.num_images < 5) {
        std::vector<float> image(28 * 28);
        for (int j = 0; j < 28 * 28; j++) {
            image[j] = 0.0f;
        }
        ggml_backend_tensor_set(images, image.data(), 28*28*segmented_images.num_images*sizeof(float), 28*28*sizeof(float));
    }
    // calculate
    if (ggml_backend_graph_compute(vit_captcha_v3->model->backend, vit_captcha_v3->graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute() failed\n");
        exit(1);
    }
    ggml_tensor* hidden_states = ggml_graph_get_tensor(vit_captcha_v3->graph, "hidden_states");
    std::vector<int> result(5);
    ggml_backend_tensor_get(hidden_states, result.data(), 0, 5*sizeof(int));
    // parse the result
    char* captcha = new char[segmented_images.num_images];
    for (int i = 0; i < segmented_images.num_images; i++) {
        captcha[i] = result[i] + 'a';
    }
    return captcha;
}

double* predict_with_timer(ViTCaptchaV3* vit_captcha_v3, const char* image_path) {
    // alloc
    if (!vit_captcha_v3->is_allocated) {
        static ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(vit_captcha_v3->model->backend));
        if (!ggml_gallocr_alloc_graph(alloc, vit_captcha_v3->graph)) {
            fprintf(stderr, "ggml_gallocr_alloc_graph() failed\n");
            exit(1);
        }
        vit_captcha_v3->is_allocated = true;
    }
    auto preproces_begin_time = std::chrono::high_resolution_clock::now();
    // segment the image
    SegmentedImages segmented_images = segment(image_path);
    // build input
    ggml_tensor* images = ggml_graph_get_tensor(vit_captcha_v3->graph, "images");
    for (int i = 0; i < segmented_images.num_images; i++) {
        std::vector<float> image(28 * 28);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                image[y * 28 + x] = segmented_images.images[i].at<uchar>(y, x) / 255.0f;
            }
        }
        ggml_backend_tensor_set(images, image.data(), 28*28*i*sizeof(float), 28*28*sizeof(float));
    }
    if (segmented_images.num_images < 5) {
        std::vector<float> image(28 * 28);
        for (int j = 0; j < 28 * 28; j++) {
            image[j] = 0.0f;
        }
        ggml_backend_tensor_set(images, image.data(), 28*28*segmented_images.num_images*sizeof(float), 28*28*sizeof(float));
    }
    auto preprocess_end_time = std::chrono::high_resolution_clock::now();
    // calculate
    auto predict_begin_time = std::chrono::high_resolution_clock::now();
    if (ggml_backend_graph_compute(vit_captcha_v3->model->backend, vit_captcha_v3->graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute() failed\n");
        exit(1);
    }
    // get result
    ggml_tensor* hidden_states = ggml_graph_get_tensor(vit_captcha_v3->graph, "hidden_states");
    std::vector<int> result(5);
    ggml_backend_tensor_get(hidden_states, result.data(), 0, 5*sizeof(int));
    // parse the result
    char* captcha = new char[segmented_images.num_images];
    for (int i = 0; i < segmented_images.num_images; i++) {
        captcha[i] = result[i] + 'a';
    }
    auto predict_end_time = std::chrono::high_resolution_clock::now();
    // pack and return
    double* records = new double[4 + segmented_images.num_images];
    // in ms
    // all_time, preprocess_time, predict_time, num_images
    std::chrono::duration<double> all_time = predict_end_time - preproces_begin_time;
    std::chrono::duration<double> preprocess_time = preprocess_end_time - preproces_begin_time;
    std::chrono::duration<double> predict_time = predict_end_time - predict_begin_time;
    records[0] = all_time.count() * 1000;
    records[1] = preprocess_time.count() * 1000;
    records[2] = predict_time.count() * 1000;
    records[3] = segmented_images.num_images;
    // captcha
    for (int i = 0; i < segmented_images.num_images; i++) {
        records[4 + i] = captcha[i];
    }
    return records;
}

#include <filesystem>
#include <vector>

#include "vit_captcha.h"


int main(int argc, char** argv) {
    const char* model_path = "";
    const char* path = "";
    int test_mode = 0;
    // test mode 0 for testing predict
    // test mode 1 for testing predict_with_timer
    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        path = argv[2];
    }
    if (argc > 3) {
        test_mode = std::stoi(argv[3]);
    }
    // load model
    ViTCaptchaV3Model* model = load_vit_captcha_v3(model_path);
    printf("ViTCaptchaV3Hparams name:                %s\n", model->hparams.name);
    printf("ViTCaptchaV3Hparams image_size:          %d\n", model->hparams.image_size);
    printf("ViTCaptchaV3Hparams num_layers:          %d\n", model->hparams.num_layers);
    printf("ViTCaptchaV3Hparams num_classes:         %d\n", model->hparams.num_classes);
    printf("ViTCaptchaV3Hparams patch_size:          %d\n", model->hparams.patch_size);
    printf("ViTCaptchaV3Hparams in_channels:         %d\n", model->hparams.in_channels);
    printf("ViTCaptchaV3Hparams hidden_size:         %d\n", model->hparams.hidden_size);
    printf("ViTCaptchaV3Hparams num_attention_heads: %d\n", model->hparams.num_attention_heads);
    printf("ViTCaptchaV3Hparams num_key_value_heads: %d\n", model->hparams.num_key_value_heads);
    printf("ViTCaptchaV3Hparams intermediate_size:   %d\n", model->hparams.intermediate_size);
    printf("ViTCaptchaV3Hparams act_fn:              %s\n", model->hparams.act_fn);
    // build graph
    ViTCaptchaV3* vit_captcha_v3 = build_vit_captcha_v3_graph(model);
    std::cout << "Graph has been built" << std::endl;
    // predict
    if (test_mode == 0) {
        double* result = predict_with_timer(vit_captcha_v3, path);
        // printf("Predict result: %s\n", result);
        printf("All time: %.2fs\n", result[0]);
    }
    // predict_with_timer
    else if (test_mode == 1) {
        std::vector<std::string> image_paths;
        for (const auto &entry : std::filesystem::directory_iterator(path)) {
            image_paths.push_back(entry.path().string());
        }
        double all_time = 0.0, preprocess_time = 0.0, predict_time = 0.0;
        int captcha_correct = 0, captcha_total = 0;
        int char_correct = 0, char_total = 0;
        int num_chars = 0, num_images = 0;
        for (auto image_path : image_paths) {
            double* records = predict_with_timer(vit_captcha_v3, image_path.c_str());
            all_time += records[0];
            preprocess_time += records[1];
            predict_time += records[2];
            num_chars += (int)records[3];
            num_images++;
            printf("Processing %s (%d)\n", image_path.c_str(), num_images);
            // check accuracy
            std::string image_name = image_path.substr(image_path.find_last_of("/\\") + 1);
            image_name = image_name.substr(0, image_name.find_first_of("_"));
            const char* gt = image_name.c_str();
            // metric
            captcha_total++;
            if ((int)records[3] != image_name.size()) {
                continue;
            } else {
                int this_correct = 0;
                for (int i = 0; i < (int)records[3]; i++) {
                    if ((char)records[4 + i] == gt[i]) {
                        this_correct++;
                    }
                }
                if (this_correct == (int)records[3]) {
                    captcha_correct++;
                }
                char_correct += this_correct;
                char_total += (int)records[3];
            }
        }
        double char_acc = (double)char_correct / char_total;
        double captcha_acc = (double)captcha_correct / captcha_total;
        all_time /= 1000.0;
        preprocess_time /= 1000.0;
        predict_time /= 1000.0;
        printf("The accuracy of char: %.2f%% (%d/%d)\n", 100 * char_acc, char_correct, char_total);
        printf("The accuracy of captcha: %.2f%% (%d/%d)\n", 100 * captcha_acc, captcha_correct, captcha_total);
        printf("The fps of char is %.2f (%d chars in %.2fs)\n", num_chars / predict_time, num_chars, predict_time);
        printf("The fps of captcha is %.2f (%d captchas in %.2fs)\n", num_images / all_time, num_images, all_time);
        printf("The fps of preprocess is %.2f (%d captchas in %.2fs)\n", num_images / preprocess_time, num_images, preprocess_time);
    }
    // free
    free_vit_captcha_v3(model);
    return 0;
}

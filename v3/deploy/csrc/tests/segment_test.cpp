#include <filesystem>
#include <vector>

#include "segment.h"


int main(int argc, char **argv) {
    const char* path = "";
    int test_mode = 0;
    // test mode 0 for single image test to check whether error occurs
    // test mode 1 for image directory accuracy test
    if (argc > 1) {
        path = argv[1];
    }
    if (argc > 2) {
        test_mode = std::stoi(argv[2]);
    }
    if (test_mode == 0) {
        SegmentedImages segmented_images = segment(path);
        std::cout << "Number of segmented images: " << segmented_images.num_images << std::endl;
        return 0;
    } else if (test_mode == 1) {
        std::vector<std::string> image_paths;
        for (const auto &entry : std::filesystem::directory_iterator(path)) {
            image_paths.push_back(entry.path().string());
        }
        int num_correct = 0, num_total = 0;
        for (auto image_path : image_paths) {
            std::cout << "Processing " << image_path << std::endl;
            SegmentedImages segmented_images = segment(image_path.c_str());
            std::string image_name = image_path.substr(image_path.find_last_of("/\\") + 1);
            std::string ground_truth = image_name.substr(0, image_name.find_first_of("_"));
            if (segmented_images.num_images == ground_truth.size()) {
                num_correct++;
            }
            num_total++;
        }
        std::cout << "Accuracy: " << num_correct << "/" << num_total << std::endl;
        return 0;
    }
}

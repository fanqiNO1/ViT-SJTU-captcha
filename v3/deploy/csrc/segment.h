#pragma once

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>


struct SplitPoints {
    cv::Mat points;
    cv::Mat labels;
    int num_points = 0;
};


struct SegmentedImages {
    std::vector<cv::Mat> images;
    int num_images = 0;
};


SplitPoints get_split_points(cv::Mat image, cv::Rect bbox);
std::tuple<float, float> get_mean_x_cluster(cv::Mat points, cv::Mat labels);
SegmentedImages segment(const char* image_path);

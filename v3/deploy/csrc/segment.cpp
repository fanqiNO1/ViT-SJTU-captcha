#include <algorithm>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "segment.h"


SplitPoints get_split_points(cv::Mat image, cv::Rect bbox) {
    SplitPoints split_points;
    std::vector<cv::Point> points;
    std::vector<int> x_coords, labels;
    for (int j = bbox.y; j < bbox.y + bbox.height; j++) {
        for (int i = bbox.x; i < bbox.x + bbox.width; i++) {
            if (image.at<uchar>(j, i) > 0) {
                points.push_back(cv::Point(i - bbox.x, j - bbox.y));
                split_points.num_points++;
                x_coords.push_back(i - bbox.x);
            }
        }
    }
    // get the median of y_coords
    // sort is inplace
    std::sort(x_coords.begin(), x_coords.end());
    int x_median = x_coords[x_coords.size() / 2];
    // get the labels
    for (auto point : points) {
        // x > x_median -> 1, x < x_median -> 0
        labels.push_back(point.x > x_median ? 1 : 0);
    }
    // convert points and labels to cv::Mat
    split_points.points = cv::Mat(split_points.num_points, 2, CV_32F);
    split_points.labels = cv::Mat(split_points.num_points, 0, CV_32F);
    for (int i = 0; i < split_points.num_points; i++) {
        split_points.points.at<float>(i, 0) = points[i].y;
        split_points.points.at<float>(i, 1) = points[i].x;
        split_points.labels.push_back(labels[i]);
    }
    return split_points;
}


std::tuple<float, float> get_mean_x_cluster(cv::Mat points, cv::Mat labels) {
    // mean_x_cluster_0 = np.mean(split_points[labels == 0, 1])
    // mean_x_cluster_1 = np.mean(split_points[labels == 1, 1])
    float mean_x_cluster_0 = 0, mean_x_cluster_1 = 0;
    int num_points_cluster_0 = 0, num_points_cluster_1 = 0;
    // labels is a column vector
    for (int i = 0; i < labels.rows; i++) {
        if (labels.at<float>(i, 0) == 0) {
            mean_x_cluster_0 += points.at<float>(i, 1);
            num_points_cluster_0++;
        } else {
            mean_x_cluster_1 += points.at<float>(i, 1);
            num_points_cluster_1++;
        }
    }
    mean_x_cluster_0 /= num_points_cluster_0;
    mean_x_cluster_1 /= num_points_cluster_1;
    return std::make_tuple(mean_x_cluster_0, mean_x_cluster_1);
}


SegmentedImages segment(const char* image_path) {
    const static float kun_ratio = 0.268;  // Don't ask
    const static float kun_aspect_ratio = 1.77;
    // convert image to grayscale and apply threshold
    cv::Mat image = cv::imread(image_path);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // revert italic by affine transformation
    int height = gray.rows, width = gray.cols;
    cv::Mat affine = cv::getRotationMatrix2D(cv::Point2f(width / 2, height / 2), -0.22, 1);
    cv::warpAffine(gray, gray, affine, cv::Size(width, height));
    // apply threshold
    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    // find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    // find bounding box for the whole cropped region
    int min_x = 0x7fffffff, min_y = 0x7fffffff, max_x = 0, max_y = 0;
    for (auto contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x + w);
        max_y = std::max(max_y, y + h);
    }
    // crop the image
    int cropped_to_w = max_x - min_x;
    // re-process the contours
    std::vector<cv::Rect> bboxes;
    for (auto contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        int this_x = rect.x, this_y = rect.y, this_w = rect.width, this_h = rect.height;

        if ((cropped_to_w * kun_ratio < this_w) || (this_w >= kun_aspect_ratio * this_h)) {
            // apply SVM classification to split the wide bounding box
            SplitPoints split_points = get_split_points(thresh, rect);
            if (split_points.num_points > 1) {
                cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
                model->setType(cv::ml::SVM::C_SVC);
                model->setKernel(cv::ml::SVM::LINEAR);
                model->setC(1);
                model->train(split_points.points, cv::ml::ROW_SAMPLE, split_points.labels);
                // predict the labels
                cv::Mat predicted_labels(split_points.num_points, 0, CV_32F);
                model->predict(split_points.points, predicted_labels);
                // find the mean x-values of thr both clusters
                std::tuple<float, float> mean_x_clusters = get_mean_x_cluster(split_points.points, predicted_labels);
                float mean_x_cluster_0 = std::get<0>(mean_x_clusters);
                float mean_x_cluster_1 = std::get<1>(mean_x_clusters);
                int split_x = (mean_x_cluster_0 + mean_x_cluster_1) / 2;
                // ensure a significant separation before splitting
                if (std::abs(mean_x_cluster_0 - mean_x_cluster_1) > this_w * 0.4) {
                    bboxes.push_back(cv::Rect(this_x, this_y, split_x, this_h));
                    bboxes.push_back(cv::Rect(this_x + split_x, this_y, this_w - split_x, this_h));
                } else {
                    bboxes.push_back(rect);
                }
            } else {
                bboxes.push_back(rect);
            }
        } else {
            bboxes.push_back(rect);
        }
    }
    // sort the bounding boxes by x-coordinate
    std::sort(bboxes.begin(), bboxes.end(), [](const cv::Rect &a, const cv::Rect &b) {return a.x < b.x;});
    // patch i and j
    // re-iterate the bboxes, find width < 10 and height < 10
    // and merge it with the closest bbox other than itself
    for (int i = 0; i < bboxes.size(); i++) {
        int x = bboxes[i].x, y = bboxes[i].y, w = bboxes[i].width, h = bboxes[i].height;
        if ((w < 9) && (h < 9)) {
            // find the closest bbox other than itself
            int closest_index = -1;
            int closest_distance = 0x7fffffff;
            for (int j = 0; j < bboxes.size(); j++) {
                if ((i == j) || (bboxes[j].width <= 0) || (bboxes[j].height <= 0)) {
                    continue;
                }
                int x2 = bboxes[j].x, y2 = bboxes[j].y, w2 = bboxes[j].width, h2 = bboxes[j].height;
                int distance = std::abs(x - x2);
                if (distance < closest_distance) {
                    closest_index = j;
                    closest_distance = distance;
                }
            }
            // merge the two bboxes
            int x2 = bboxes[closest_index].x, y2 = bboxes[closest_index].y, w2 = bboxes[closest_index].width, h2 = bboxes[closest_index].height;
            bboxes[closest_index] = cv::Rect(
                std::min(x, x2),
                std::min(y, y2),
                std::max(x + w, x2 + w2) - std::min(x, x2),
                std::max(y + h, y2 + h2) - std::min(y, y2)
            );
            bboxes[i] = cv::Rect();
        }
    }
    // post-process the bboxes
    std::vector<cv::Rect> valid_bboxes;
    for (auto bbox : bboxes) {
        if ((bbox.width > 0) && (bbox.height > 0)) {
            valid_bboxes.push_back(bbox);
        }
    }
    bboxes = valid_bboxes;
    // sort the bboxes by x-coordinate again
    // if 6 bboxes are found, merge two closest bboxes
    std::sort(bboxes.begin(), bboxes.end(), [](const cv::Rect &a, const cv::Rect &b) {return a.x < b.x;});

    while (bboxes.size() >= 6) {
        int min_distance = 0x7fffffff;
        int merge_index = -1;
        for (int i = 0; i < 5; i++) {
            int x = bboxes[i].x, y = bboxes[i].y, w = bboxes[i].width, h = bboxes[i].height;
            int x2 = bboxes[i + 1].x, y2 = bboxes[i + 1].y, w2 = bboxes[i + 1].width, h2 = bboxes[i + 1].height;
            int distance = std::abs(x + w - x2);
            if (distance < min_distance) {
                min_distance = distance;
                merge_index = i;
            }
        }
        if (min_distance < 5) {
            int x = bboxes[merge_index].x, y = bboxes[merge_index].y, w = bboxes[merge_index].width, h = bboxes[merge_index].height;
            int x2 = bboxes[merge_index + 1].x, y2 = bboxes[merge_index + 1].y, w2 = bboxes[merge_index + 1].width, h2 = bboxes[merge_index + 1].height;
            bboxes[merge_index] = cv::Rect(x, y, w + w2, h);
            bboxes[merge_index + 1] = cv::Rect();
        }
        // remove the empty bboxes
        std::vector<cv::Rect> valid_bboxes;
        for (auto bbox : bboxes) {
            if ((bbox.width > 0) && (bbox.height > 0)) {
                valid_bboxes.push_back(bbox);
            }
        }
        bboxes = valid_bboxes;
    }
    // final assertion
    std::sort(bboxes.begin(), bboxes.end(), [](const cv::Rect &a, const cv::Rect &b) {return a.y < b.y;});
    // extract segmented character images
    SegmentedImages segmented_images;
    for (auto bbox : bboxes) {
        cv::Mat this_char = thresh(bbox);
        cv::resize(this_char, this_char, cv::Size(28, 28));
        segmented_images.images.push_back(this_char);
        segmented_images.num_images++;
    }
    return segmented_images;
}

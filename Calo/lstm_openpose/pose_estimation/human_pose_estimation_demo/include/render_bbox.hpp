#pragma once

#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "bbox.hpp"

namespace human_pose_estimation {
    void renderBbox(cv::Mat& image, struct BBox& bbox, int class_id,
                    const std::string& label);

    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 255),    // Purple
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(235,206,135),    // Skyblue
        cv::Scalar(128, 0, 0),      // Navyblue
        cv::Scalar(255, 255, 250),   // Azure
        cv::Scalar(255, 0, 127),    // Slate
        cv::Scalar(30, 105, 210),   // Chocolate
        cv::Scalar(112, 255, 202),  // Olive
        cv::Scalar(0, 140, 255),    // Orange
        cv::Scalar(255, 102, 224)   // Orchi
    };

    cv::Scalar generate_random_color();

}  // namespace human_pose_estimation

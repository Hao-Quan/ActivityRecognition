#include <vector>

#include "render_bbox.hpp"

namespace human_pose_estimation {

cv::Scalar generate_random_color() {
    return cv::Scalar(rand() % (255+1), rand() % (255+1), rand() % (255+1));
}

void renderBbox(cv::Mat& image, struct BBox& bbox, int class_id,
                const std::string& label) {
    CV_Assert(image.type() == CV_8UC3);

    cv::Scalar color;
    if (class_id == -1) {
        color = generate_random_color();
    }
    else {
        color = colors[class_id % colors.size()];
    }

    // Render BBox
    cv::Point2d TL(bbox.xtl, bbox.ytl);
    cv::Point2d BR(bbox.xbr, bbox.ybr);
    cv::rectangle(image, TL, BR, color);

    // Render label
    cv::String cv_label(label);
    cv::putText(image, cv_label, TL, cv::FONT_HERSHEY_COMPLEX_SMALL,
                2, color);

}
}  // namespace human_pose_estimation

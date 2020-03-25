#pragma once

namespace human_pose_estimation {
struct BBox {
    BBox(const int& xtl, const int& ytl, const int& xbr, const int& ybr);

    int xtl;
    int ytl;
    int xbr;
    int ybr;
};
}  // namespace human_pose_estimation

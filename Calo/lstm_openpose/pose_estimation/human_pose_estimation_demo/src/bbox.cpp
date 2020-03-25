#include "bbox.hpp"

namespace human_pose_estimation {

BBox::BBox(const int& xtl, const int& ytl, const int& xbr, const int& ybr)
    : xtl(xtl), ytl(ytl), xbr(xbr), ybr(ybr) {}

}  // namespace human_pose_estimation

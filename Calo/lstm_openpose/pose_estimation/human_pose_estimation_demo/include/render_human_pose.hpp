// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <iomanip>

#include <opencv2/core/core.hpp>

#include "human_pose.hpp"

namespace human_pose_estimation {
    void renderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image,
                         const bool& show_keypoints_index,
                         const bool& show_x=false);
}  // namespace human_pose_estimation

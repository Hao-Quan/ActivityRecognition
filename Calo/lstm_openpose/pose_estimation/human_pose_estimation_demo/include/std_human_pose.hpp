// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

namespace human_pose_estimation {

struct Point2D {
  Point2D(float x, float y, float score) : x(x), y(y), score(score) {

  }
  Point2D() : Point2D(-1.0f, -1.0f, 0.0f) {

  }

  float x;
  float y;
  float score;
};

struct StdHumanPose {
    StdHumanPose() : keypoints(std::vector<Point2D>()), score(0.0f) {
      keypoints.reserve(18);
    }

    std::vector<Point2D> keypoints;
    float score;
};
}  // namespace human_pose_estimation

#pragma once

#include "render_human_pose.hpp"
#include "render_bbox.hpp"

#include <vector>

#include <opencv2/opencv.hpp>

#include "human_pose_estimator.hpp"
#include "human_pose.hpp"

#include "std_human_pose.hpp"
#include "bbox.hpp"

namespace human_pose_estimation {
class CameraHPE {
public:
  CameraHPE(const std::string& modelPath, const std::string& targetDeviceName,
            const std::string& videoPath, bool enablePerformanceReport,
            bool show_keypoints_index, bool show_x);
  ~CameraHPE();

  std::vector<StdHumanPose> estimate_poses();
  StdHumanPose estimate_pose(struct BBox bbox);
  bool read();
  bool render_poses();
  bool render_bbox(struct BBox bbox, int class_id, const std::string& label);
  double get_inference_time();
  double get_instant_inference_time();
  void set_id(int index, int id);
  int get_image_cols();
  int get_image_rows();
private:
  std::vector<StdHumanPose> estimate_poses(bool limit_bbox, struct BBox bbox);
  HumanPoseEstimator estimator;
  cv::VideoCapture cap;
  cv::Mat image;
  std::vector<HumanPose> poses;
  bool show_keypoints_index;
  bool show_x;
  double inferenceTime;
  double instantInferenceTime;
};
}  // namespace human_pose_estimation

#pragma once

#include <vector>
#include "camera_hpe.hpp"
#include "bbox.hpp"

namespace human_pose_estimation {
class CameraHPEProxy {
public:
  CameraHPEProxy(const std::string& modelPath,
                 const std::string& targetDeviceName,
                 const std::string& videoPath,
                 bool enablePerformanceReport = false,
                 bool show_keypoints_index = false,
                 bool show_x = false);
  ~CameraHPEProxy();

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
  CameraHPE ce;
};
}  // namespace human_pose_estimation

#include "camera_hpe_proxy.hpp"

namespace human_pose_estimation {
  CameraHPEProxy::CameraHPEProxy(const std::string& modelPath,
                                 const std::string& targetDeviceName,
                                 const std::string& videoPath,
                                 bool enablePerformanceReport,
                                 bool show_keypoints_index,
                                 bool show_x)
    : ce(modelPath, targetDeviceName, videoPath, enablePerformanceReport,
         show_keypoints_index, show_x) {}

  CameraHPEProxy::~CameraHPEProxy() {}

  std::vector<StdHumanPose> CameraHPEProxy::estimate_poses() {
    return this->ce.estimate_poses();
  }

  StdHumanPose CameraHPEProxy::estimate_pose(struct BBox bbox) {
    return this->ce.estimate_pose(bbox);
  }

  bool CameraHPEProxy::read() {
    return this->ce.read();
  }

  bool CameraHPEProxy::render_poses() {
    return this->ce.render_poses();
  }

  bool CameraHPEProxy::render_bbox(struct BBox bbox, int class_id,
                                   const std::string& label) {
    return this->ce.render_bbox(bbox, class_id, label);
  }

  double CameraHPEProxy::get_inference_time() {
    return this->ce.get_inference_time();
  }

  double CameraHPEProxy::get_instant_inference_time() {
    return this->ce.get_inference_time();
  }

  void CameraHPEProxy::set_id(int index, int id) {
    this->ce.set_id(index, id);
  }

  int CameraHPEProxy::get_image_cols() {
    return this->ce.get_image_cols();
  }

  int CameraHPEProxy::get_image_rows() {
    return this->ce.get_image_rows();
  }

}  // namespace human_pose_estimation

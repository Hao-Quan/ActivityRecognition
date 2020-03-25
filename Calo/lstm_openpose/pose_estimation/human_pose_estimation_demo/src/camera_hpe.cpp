#include "camera_hpe.hpp"

using namespace human_pose_estimation;

namespace human_pose_estimation {

    CameraHPE::CameraHPE(const std::string& modelPath,
                         const std::string& targetDeviceName,
                         const std::string& videoPath,
                         bool enablePerformanceReport,
                         bool show_keypoints_index, bool show_x)
      : estimator(modelPath, targetDeviceName, enablePerformanceReport),
        show_keypoints_index(show_keypoints_index), show_x(show_x),
        inferenceTime(0.0), instantInferenceTime(0.0) {
          //std::cout << "CameraHPE constructor..." << std::endl;
          if (!(videoPath == "cam" ? cap.open(0) : cap.open(videoPath))) {
              throw std::logic_error("Cannot open camera");
          }
          if (!cap.read(image)) {
              throw std::logic_error("Failed to get frame from cv::VideoCapture");
          }
          estimator.estimate(image);  // Do not measure network reshape, if it happened
          //std::cout << "Leaving CameraHPE constructor..." << std::endl;
    }

    std::vector<StdHumanPose> CameraHPE::estimate_poses(bool limit_bbox, struct BBox bbox) {
      //std::cout << "Entering CameraHPE::estimate_poses()..." << std::endl;
      cv::Mat sub_image = this->image;
      cv::Point2d relative_origin(0,0);
      if (limit_bbox) {
        // TODO: here limit the image
        cv::Point2d TL(bbox.xtl, bbox.ytl);
        relative_origin = TL;
        cv::Point2d BR(bbox.xbr, bbox.ybr);
        sub_image = this->image( cv::Rect(TL, BR) );
      }
      double t1 = static_cast<double>(cv::getTickCount());
      std::vector<HumanPose> rel_poses = estimator.estimate(sub_image);
      double t2 = static_cast<double>(cv::getTickCount());
      instantInferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
      if (inferenceTime == 0) {
          inferenceTime = instantInferenceTime;
      } else {
          inferenceTime = inferenceTime * 0.95 + 0.05 * instantInferenceTime;
      }

      std::vector<HumanPose> local_poses;
      std::vector<StdHumanPose> std_poses;
      for (size_t humanPoseId=0; humanPoseId < rel_poses.size(); humanPoseId++) {
        HumanPose const& rel_pose = rel_poses[humanPoseId];

        StdHumanPose stdpose;
        std::vector<cv::Point2f> abs_kpts(rel_pose.keypoints.size(),
                                          cv::Point2f(-1.0f, -1.0f));

        HumanPose human_pose(std::vector<cv::Point2f>(rel_pose.keypoints.size(), cv::Point2f(-1.0f, -1.0f)),
                             0);

        for (size_t kptId=0; kptId < rel_pose.keypoints.size(); kptId++) {
          cv::Point2f const& cvpoint = rel_pose.keypoints[kptId];
          // Reference to absolute origin
          if (cvpoint.x == -1.0f || cvpoint.y == -1.0f) {
            human_pose.keypoints[kptId] = cv::Point2f(-1.0f, -1.0f);
          }
          else {
            human_pose.keypoints[kptId] = cv::Point2f(cvpoint.x + relative_origin.x,
                                                      cvpoint.y + relative_origin.y);
          }

          human_pose.keypoints_scores[kptId] = rel_pose.keypoints_scores[kptId];
          // Convert cv::Point to StdPoint
          float score = rel_pose.keypoints_scores[kptId];
          Point2D stdpoint(human_pose.keypoints[kptId].x,
                           human_pose.keypoints[kptId].y, score);
          stdpose.keypoints.push_back(stdpoint);
        }
        stdpose.score = rel_pose.score;
        std_poses.push_back(stdpose);

        local_poses.push_back(human_pose);
      }

      if (limit_bbox && !local_poses.empty()) {
        this->poses.push_back(local_poses[0]);
      }
      else {
        this->poses = local_poses;
      }

      return std_poses;
    }

    std::vector<StdHumanPose> CameraHPE::estimate_poses() {
      struct BBox inv_bbox(-1,-1,-1,-1);
      return estimate_poses(false, inv_bbox);
    }

    StdHumanPose CameraHPE::estimate_pose(struct BBox bbox) {
      std::vector<StdHumanPose> poses = estimate_poses(true, bbox);
      if (poses.size() > 0) {
        return poses[0];
      }

      struct StdHumanPose null_pose;
      return null_pose;
    }

    bool CameraHPE::read() {
      this->poses.clear();
      return cap.read(image);
    }

    static int delay=33;

    bool CameraHPE::render_poses() {
      renderHumanPose(this->poses, this->image, this->show_keypoints_index,
                      this->show_x);

      cv::Mat fpsPane(35, 155, CV_8UC3);
      fpsPane.setTo(cv::Scalar(153, 119, 76));
      cv::Mat srcRegion = this->image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
      cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
      std::stringstream fpsSs;
      fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
      cv::putText(this->image, fpsSs.str(), cv::Point(16, 32),
                  cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
      cv::imshow("ICV Human Pose Estimation", this->image);

      int key = cv::waitKey(delay) & 255;
      if (key == 'p') {
          delay = (delay == 0) ? 33 : 0;
      }

      return (key == 27);
    }

    bool CameraHPE::render_bbox(struct BBox bbox, int class_id,
                                const std::string& label) {
      renderBbox(this->image, bbox, class_id, label);

      int key = cv::waitKey(delay) & 255;
      if (key == 'p') {
          delay = (delay == 0) ? 33 : 0;
      }

      return (key == 27);
    }

    double CameraHPE::get_inference_time() {
      return this->inferenceTime;
    }

    double CameraHPE::get_instant_inference_time() {
      return this->instantInferenceTime;
    }

    void CameraHPE::set_id(int index, int id) {
      this->poses[index].id = id;
    }

    int CameraHPE::get_image_cols() {
      return this->image.cols;
    }

    int CameraHPE::get_image_rows() {
      return this->image.rows;
    }

    CameraHPE::~CameraHPE() {

    }
}  // namespace human_pose_estimation

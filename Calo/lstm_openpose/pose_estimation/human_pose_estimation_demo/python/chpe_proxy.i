%module chpe
 %{
 #define SWIG_FILE_WITH_INIT
 #define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS
 #include "std_human_pose.hpp"
 #include "bbox.hpp"
 #include "camera_hpe_proxy.hpp"
 %}

 %include "pyabc.i"
 %include "std_string.i"
 %include "std_vector.i"
 %include "std_human_pose.hpp"
 %include "bbox.hpp"
 %include "camera_hpe.hpp"
 %include "camera_hpe_proxy.hpp"


 %template (Point2DVector) std::vector<human_pose_estimation::Point2D>;
 %template (StdHumanPoseVector) std::vector<human_pose_estimation::StdHumanPose>;

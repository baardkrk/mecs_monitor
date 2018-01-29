#ifndef INFO_EXTRACTOR_H
#define INFO_EXTRACTOR_H

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Point.h>

#include <mecs_monitor/ExtInfo.h>

///////// OpenPose dependencies //////////
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

/**
 * Input:  The 2d images we want to create a pose from and the camera
 *         information for these images. 
 * Output: A 3d matrix with the 2d and estimated 3d coordinates
 *         of each person detected in the input.
 *                  +------------+-----+------------+
 *         person 0 | keypoint 0 | ... | keypoint N |
 *                : |     :      |  :  |     :      |
 *         person M | keypoint 0 | ... | keypoint N |
 *                  +------------+-----+------------+
 *
 *         person 0:
 *                    +-----+-----+-----+-----+-----+-------+
 *         keypoint 0 | row | col |  X  |  Y  |  Z  | score |
 *                  : |  :  |  :  |  :  |  :  |  :  |   :   |
 *         keypoint N | row | col |  X  |  Y  |  Z  | score |
 *                    +-----+-----+-----+-----+-----+-------+
 *
 * The 2D keypoints are obtained using OpenPose. 
 * We'll create a few options for HOW we calculate the different 
 * 3D keypoints:
 *  - Manually using depth image
 *  - Machine Learning using the depth image
 *  - Machine Learning using only 2d data
 * 
 */

class InfoExtractor {

 private:
  // We choose to pass in the whole cv Matrices here, since
  // OpenPose wants the input on that format. And, since we're also going
  // to do manipulations/references to the images it's annoying to
  // reference them with a pointer all the time. Although implementing
  // this could speed up the code if neccesary at a later time.
  cv::Mat rgb, depth, ir, blurred_depth;
  sensor_msgs::CameraInfo::ConstPtr camera_info;

  // OpenPose parameters
  op::ScaleAndSizeExtractor *scaleAndSizeExtractor;
  op::CvMatToOpInput cvMatToOpInput;
  op::CvMatToOpOutput cvMatToOpOutput;
  op::PoseExtractorCaffe *poseExtractorCaffe;
  op::PoseCpuRenderer *poseRenderer;
  op::OpOutputToCvMat opOutputToCvMat;
  op::FrameDisplayer *frameDisplayer;

  // Methods
  op::Array<float> run_openpose(cv::Mat inputImage, std::string pw_name);
  std::tuple<double, double, double> project_3d(int row, int col);
  std_msgs::Float64MultiArray get_3d_keypoints(op::Array<float> keypoints);
  
 public:
  InfoExtractor();
  /* void update(sensor_msgs::CameraInfo::ConstPtr& _camera_info, */
  /* 	      cv::Mat _rgb, cv::Mat _depth, cv::Mat _ir); */
  mecs_monitor::ExtInfo extract(sensor_msgs::CameraInfo::ConstPtr& _camera_info,
				cv::Mat _rgb, cv::Mat _depth, cv::Mat _ir);
  /* std::vector<int, int, double, */
  /*   double, double, double> InfoExtractor::get_locations(); */
  void test();
};



#endif // INFO_EXTRACTOR_H

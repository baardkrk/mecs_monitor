#ifndef INFO_EXTRACTOR_H
#define INFO_EXTRACTOR_H

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Point.h>
#include <Eigen/Dense>

#include <mecs_monitor/ExtInfo.h>

#include <math.h>

///////// OpenPose dependencies //////////
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

/**
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

  // ====================================================================================================
  /**
   * Parameters for the human Model created form the COCO Pose model and the 
   * Drillis average human proportions.
   */
  // ====================================================================================================
  std::string constrained_limbs_names[15] = {"  hWidth", "  lCheek", "   lSide", "  rCheek", "   rSide",
					     "shoulder", "    lArm", "lForearm", "    rArm", "rForearm",
					     "     hip", "  lThigh", "    lLeg", "  rThigh", "    rLeg"};
  // the edges
  int constrained_limb_pairs[15][2] = {{16,17}, {0,15},  {15,17}, {0,14}, {14,16},
				       {5,2},   {5,6},   {6,7},   {2,3},  {3,4},
				       {8,11},  {11,12}, {12,13}, {8,9},  {9,10}};
  // the connections from each keypoint in the constrained graphs. -1 means no second connection (leaf)
  // also, the connections are only the ones shown in each graph.
  int constrained_keypoint_connections[18][2] = {{14,15}, {2,5},   {1,3},   {2,4}, {3,-1},
						 {1,6},   {5,7},   {6,-1},
						 {11,9},  {8,10},  {9,-1},
						 {8,12},  {11,13}, {12,-1},
						 {0,16},  {0,17},  {14,17}, {15,16}};
  double kp_Zd[18] = {0, 0, 0.05, 0.03, 0.02, 0.05, 0.03, 0.02, 0, 0.05, 0.03, 0, 0.05, 0.03, 0, 0, 0, 0};
  Eigen::VectorXd norm_constr;
  // ====================================================================================================
  
  // images for this frame
  cv::Mat rgb, depth, ir, hd;
  sensor_msgs::CameraInfo::ConstPtr camera_info;

  // OpenPose parameters
  op::ScaleAndSizeExtractor *scaleAndSizeExtractor;
  op::CvMatToOpInput cvMatToOpInput;
  op::CvMatToOpOutput cvMatToOpOutput;
  op::PoseExtractorCaffe *poseExtractorCaffe;
  op::PoseCpuRenderer *poseRenderer;
  op::OpOutputToCvMat opOutputToCvMat;
  op::FrameDisplayer *frameDisplayer;

  // Image and 3D mapping
  Eigen::Vector3d project_to_3d(cv::Point point, double dZ);
  cv::Point project_to_img(Eigen::Vector3d);
  cv::Point img_map(cv::Mat src, cv::Mat dst, cv::Point point);

  // Info extraction
  op::Array<float> run_openpose(cv::Mat inputImage, std::string pw_name); 
  std_msgs::Float64MultiArray get_3d_keypoints(op::Array<float> keypoints);
  std::tuple<cv::Point, cv::Point> get_roi(Eigen::Vector3d point, double width, double height,
					   double dX, double dY);
  /* cv::Mat get_roi(cv::Point* points); */ // TODO


 public:
  InfoExtractor();
  mecs_monitor::ExtInfo extract(sensor_msgs::CameraInfo::ConstPtr& _camera_info,
				cv::Mat _rgb, cv::Mat _depth, cv::Mat _ir);
  void test();
};


#endif // INFO_EXTRACTOR_H

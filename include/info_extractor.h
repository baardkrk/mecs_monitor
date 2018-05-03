#ifndef INFO_EXTRACTOR_H
#define INFO_EXTRACTOR_H

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Point.h>
#include <Eigen/Dense>

#include <mecs_monitor/ExtInfo.h>

#include <math.h>
#include <list>

///////// OpenPose dependencies //////////
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

class InfoExtractor {

 private:
  double kp_Zd[18] = {0, 0, 0.05, 0.03, 0.02, 0.05, 0.03, 0.02, 0, 0.05, 0.03, 0, 0.05, 0.03, 0, 0, 0, 0};
  
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
  /* Eigen::Vector3d project_to_3d(cv::Point point, double dZ, bool override); */
  cv::Point project_to_img(Eigen::Vector3d);
  cv::Point img_map(cv::Mat src, cv::Mat dst, cv::Point point);

  op::Array<float> run_openpose(cv::Mat inputImage, std::string pw_name);
  std::tuple<cv::Point, cv::Point> get_roi(Eigen::Vector3d point,
					   double width, double height,
					   double dX, double dY);
  std_msgs::Float64MultiArray get_3d_keypoints(op::Array<float> keypoints);
 public:
  /* std::vector< std::vector<int> > traversal_keypoint_connections; */
  
  InfoExtractor();
  mecs_monitor::ExtInfo extract(sensor_msgs::CameraInfo::ConstPtr& _camera_info,
				cv::Mat _rgb, cv::Mat _depth, cv::Mat _ir);
  void test();

  Eigen::Vector3d project_to_3d(cv::Point point, double dZ, bool override);
};

class Skeleton;

class Subgraph {
  
 private:
  
  int seed;
  std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > keypoints;
  Skeleton *parent_skeleton;
  double scale;
  const int *subgraph_keypoints;

  void place_keypoint(int p_id, int c_id);
  Eigen::Vector3d unobserved_child(int p_id, int c_id);
  Eigen::Vector3d keypoint_interpolation(int p_id, int c_id, int n_id);
  Eigen::Vector3d push_vector(Eigen::Vector3d fixed, Eigen::Vector3d pushed, double length);
  void recursive_constrain(int p_id);
  Eigen::MatrixXd get_limb_scores();

  double get_limb_length(std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > graph, int parent, int child);
  
 public:
  
  Subgraph(std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > _kp, int _seed, Skeleton *_parent);
  double get_scale();
  
  std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > get_keypoints();
};

class Skeleton {

 private:
  
  std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > keypoints;
  std::vector<Subgraph> subgraphs;
  
 public:

   // ====================================================================================================
  /**
   * Parameters for the human Model created form the COCO Pose model and the 
   * Drillis average human proportions.
   */
  // ----------------------------------------------------------------------------------------------------

  static constexpr const double edge_lengths[18][18] = {
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,.035,.035,   0,   0},
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,.186,   0,.259,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,.186,   0,.146,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,.146,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,.259,   0,   0,   0,.186,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,.186,   0,.146,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,   0,.146,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,.245,   0,.191,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,   0,   0,   0,.245,   0,.246,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,.246,   0,   0,   0,   0,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,   0,   0,   0,.191,   0,   0,   0,.245,   0,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,.245,   0,.246,   0,   0,   0,   0},
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,.246,   0,   0,   0,   0,   0},
    {.035,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,.058,   0},
    {.035,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,.058},
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,.058,   0,   0,.105},
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,.058,.105,   0}
  };

  // the connections from each keypoint in the constrained graphs. -1 means no second connection (leaf)
  // also, the connections are only the ones shown in each graph.
  static constexpr const int constrained_keypoint_connections[18][2] = {
    {14,15}, {-1,-1},   {5,3},   {2,4}, {3,-1},
    {2,6},   {5,7},   {6,-1},
    {11,9},  {8,10},  {9,-1},
    {8,12},  {11,13}, {12,-1},
    {0,16},  {0,17},  {14,-1}, {15,-1}
  };

  // The three different subgraphs. terminated in -1 to escape travesal since they are of unequal length
  // Used to sort the best keypoints.
  static constexpr const int keypoints_head[6] = {0,16,14,15,17,-1};
  static constexpr const int keypoints_body[7] = {2, 3, 4, 5, 6, 7,-1};
  static constexpr const int keypoints_legs[7] = {8, 9,10,11,12,13,-1};

  // ====================================================================================================
  
  Skeleton(std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector3d, double>> original, InfoExtractor *_extractor);
  std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector3d, double>> constrain_skeleton();
};



#endif // INFO_EXTRACTOR_H

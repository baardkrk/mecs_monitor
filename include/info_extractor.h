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
  // ----------------------------------------------------------------------------------------------------
  std::string constrained_limbs_names[15] = {"  hWidth", "  lCheek", "   lSide", "  rCheek", "   rSide",
					     "shoulder", "    lArm", "lForearm", "    rArm", "rForearm",
					     "     hip", "  lThigh", "    lLeg", "  rThigh", "    rLeg"};
  // the edges
  int constrained_limb_pairs[15][2] = {{16,17}, {0,15},  {15,17}, {0,14}, {14,16},
				       {5,2},   {5,6},   {6,7},   {2,3},  {3,4},
				       {8,11},  {11,12}, {12,13}, {8,9},  {9,10}};

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

  // ending each subgraph in -1 to escaping traversal
  int keypoints_head = {0,16,14,15,17,-1};
  int keypoints_body = {2, 3, 4, 5, 6, 7,-1};
  int keypoints_legs = {8, 9,10,11,12,13,-1};
  
  double kp_Zd[18] = {0, 0, 0.05, 0.03, 0.02, 0.05, 0.03, 0.02, 0, 0.05, 0.03, 0, 0.05, 0.03, 0, 0, 0, 0};
  double norm_constr[15] = {.105, .035, .058, .035, .058,
			    .259, .186, .146, .186, .146,
			    .191, .245, .246, .245, .246};
  // ====================================================================================================
  

  class Skeleton
  {
  private:   
    std::vector< std::tuple<Eigen::Vector2d,
      Eigen::Vector3d, double> > keypoints;
    double scale;
    int seeds[3];
    InfoExtractor *p_ext;
    int visited_keypoints[18];
      
    void place_keypoint(int p_id, int c_id);
    void unobserved_child(int p_id, int c_id);
    void keypoint_interpolation(int p_id, int c_id, int n_id);
    Eigen::Vector3d push_vector(Eigen::Vector3d fixed, Eigen::Vector3d pushed, double length);
    void recursive_constrain(int p_id);
    std::list<int> sort_keypoints(int *subgraph);
    
  public:
    Skeleton(std::vector< std::tuple<Eigen::Vector2d,
	     Eigen::Vector3d, double> > _keypoints, int _seed, InfoExtractor *_p_ext);
    std::vector< std::tuple<Eigen::Vector2d,
      Eigen::Vector3d, double> > get_skeleton();
  };
  
  // images for this frame
  cv::Mat rgb, depth, ir, hd;
  sensor_msgs::CameraInfo::ConstPtr camera_info;

  // Skeletons in this frame
  /* std::vector<Skeleton> skeletons; */
  
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

  // Info extraction
  op::Array<float> run_openpose(cv::Mat inputImage, std::string pw_name); 
  std_msgs::Float64MultiArray get_3d_keypoints(op::Array<float> keypoints);
  std::tuple<cv::Point, cv::Point> get_roi(Eigen::Vector3d point, double width, double height,
					   double dX, double dY);
  
  void constrain_skeleton(std::vector< std::tuple<Eigen::Vector2d,
			  Eigen::Vector3d, double> >* keypoints);
  /* cv::Mat get_roi(cv::Point* points); */ // TODO


 public:
  std::vector< std::vector<int> > traversal_keypoint_connections;
  
  InfoExtractor();
  mecs_monitor::ExtInfo extract(sensor_msgs::CameraInfo::ConstPtr& _camera_info,
				cv::Mat _rgb, cv::Mat _depth, cv::Mat _ir);
  void test();

  Eigen::Vector3d project_to_3d(cv::Point point, double dZ, bool override);
};


#endif // INFO_EXTRACTOR_H

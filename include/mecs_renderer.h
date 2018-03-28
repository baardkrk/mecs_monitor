#ifndef MECS_RENDERER_H
#define MECS_RENDERER_H

#include <ros/ros.h>
#include <mecs_monitor/ExtInfo.h>
#include <visualization_msgs/MarkerArray.h>

class MecsRenderer {
 private:
  // for the COCO model
  std::string keypoint_names[18] = {"nose", "neck", "rShlder", "rElbow" , "rWrist", 
				    "lShlder", "lElbow", "lWrist", "rHip", "rKnee", 
				    "rAnkle", "lHip", "lKnee", "lAnkle", "rEye", "lEye",
				    "rEar", "lEar"};
  int keypoint_pairs[17][2] = {{0,1},{1,2},{2,3},{3,4},{1,5},{5,6},{6,7},{1,8},{8,9},{9,10},
			       {1,11},{11,12},{12,13},{0,14},{14,16},{0,15},{15,17}};
  
  ros::NodeHandle nh;
  ros::Subscriber subscriber;
  ros::Publisher publisher;
  
  std::string keypoint_topic;
  visualization_msgs::MarkerArray markers;
  
  // METHODS
  std::tuple<double, double, double> hsv_to_rgb(int h, int s, int v);

 public:
  MecsRenderer(std::string _keypoint_topic, ros::NodeHandle& _nh);
  void render_keypoints(const mecs_monitor::ExtInfo::ConstPtr& msg);
};

#endif // MECS_RENDERER_H

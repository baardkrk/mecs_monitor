#include <ros/ros.h>
#include <mecs_monitor/ExtInfo.h>

#include "mecs_renderer.h"

std::string keypoint_names[] = {"nose", "neck", "rShlder", "rElbow" , "rWrist",\
				"lShlder", "lElbow", "lWrist", "rHip", "rKnee",\
				"rAnkle", "lHip", "lKnee", "lAnkle", "rEye", "lEye",\
				"rEar", "lEar"};

void callback(const mecs_monitor::ExtInfo::ConstPtr& msg) {
  double x,y,z;
  int num_ppl, num_body_parts, person_stride, body_part_stride;
  
  num_ppl = msg->keypoints.layout.dim[0].size;
  num_body_parts = msg->keypoints.layout.dim[1].size;
  person_stride = msg->keypoints.layout.dim[0].stride;
  body_part_stride = msg->keypoints.layout.dim[1].stride;

  for (int person = 0; person < num_ppl; person++) {
    std::cout << "Peson: " << person << std::endl;
    for (int body_part = 0; body_part < num_body_parts; body_part++) {
      x = msg->keypoints.data[person*person_stride + body_part*body_part_stride + 2];
      y = msg->keypoints.data[person*person_stride + body_part*body_part_stride + 3];
      z = msg->keypoints.data[person*person_stride + body_part*body_part_stride + 4];

      std::cout << keypoint_names[body_part] << "\t : \t(" << x << ", " << y << ", " << z << ")\n";
    }
    std::cout << std::endl;
  }
  // std::cout << msg->header.frame_id << std::endl;
}


int main(int argc, char *argv[])
{

  ros::init(argc, argv, "mecs_info_tracker");
  ros::NodeHandle node;
  ros::Subscriber subscriber;

  // id, Center of Mass
  std::vector< std::tuple<int, geometry_msgs::Point> > tracked_people;
  
  MecsRenderer renderer("/extracted_info", node);

  // subscriber = node.subscribe("/extracted_info", 1, callback);
  ros::spin();
};

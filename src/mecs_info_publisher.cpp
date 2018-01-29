#include <ros/ros.h>
#include <mecs_monitor/ExtInfo.h>

#include "kinect_subscriber.h"
#include "info_extractor.h"


int main(int argc, char *argv[])
{

  ros::init(argc, argv, "mecs_info_extractor");
  ros::NodeHandle node;
  ros::Publisher publisher;

  publisher = node.advertise<mecs_monitor::ExtInfo>("/extracted_info", 1);

  KinectSubscriber kinect("/kinect2","/sd");
  InfoExtractor info;

  int frame_count = 0;
  double rate_sum = .0;    
  cv::Mat depth, rgb, ir;

  while (ros::ok()) {
    ros::Time begin = ros::Time::now();
    
    auto rgb_ptr = kinect.get_cv_rgb_ptr();
    auto depth_ptr = kinect.get_cv_depth_ptr();
    auto ir_ptr = kinect.get_cv_ir_ptr();
    auto info_ptr = kinect.get_camera_info();

    if (rgb_ptr != nullptr && depth_ptr != nullptr
	&& ir_ptr != nullptr && info_ptr != nullptr) {

      // 0.001 gets the value in meters, default is millimiters
      // changed this since it looks a little different in sd. in qhd 0.001 was good
      depth_ptr->image.convertTo(depth, CV_32F, 0.001);
      rgb = rgb_ptr->image;
      ir_ptr->image.convertTo(ir, CV_8UC1, 0.05); // the scale factor should actually be 1/256
      
      publisher.publish(info.extract(info_ptr, rgb, depth, ir));
    }

    ros::spinOnce();

    ros::Time end = ros::Time::now();
    double secs = (end - begin).toSec();
      
    rate_sum += 1/secs;
    if (++frame_count == 100) {
      std::cout << "frame rate: " << (rate_sum/100) << " fps"<< std::endl;
      frame_count=0;
      rate_sum = .0;
    }
  }
  
};

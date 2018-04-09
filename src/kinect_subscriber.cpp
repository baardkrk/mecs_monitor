#ifndef KINECT_SUBSCRIBER_CPP
#define KINECT_SUBSCRIBER_CPP

#include "kinect_subscriber.h"

/***************
 *
 * We are using the sd topic here, since we want to try something with the ir
 * camera. We are also using it to account for missing or bad calibration.
 *
 ***************/

KinectSubscriber::KinectSubscriber(std::string base_topic, std::string quality):
  it(nh),
  rgb_sub(it, base_topic + quality + "/image_color_rect", 1), //"/kinect2/sd/"
  depth_sub(it, base_topic + quality + "/image_depth_rect", 1),
  ir_sub(it, base_topic + "/sd/image_ir_rect", 1),
  // hd_sub(it, base_topic + "/hd/image_color_rect", 1),
  info_sub(nh, base_topic + quality + "/camera_info", 1),
  sync(KinectSyncPolicy(100), rgb_sub, depth_sub, ir_sub, info_sub)
{
  sync.registerCallback(boost::bind(&KinectSubscriber::callback, this, _1, _2, _3, _4));
  rgb_ptr = depth_ptr = ir_ptr = nullptr;
  info_ptr = nullptr;
};

void KinectSubscriber::callback(const sensor_msgs::ImageConstPtr& rgb_msg,
				const sensor_msgs::ImageConstPtr& depth_msg,
				const sensor_msgs::ImageConstPtr& ir_msg,
				// const sensor_msgs::ImageConstPtr& hd_msg,
				const sensor_msgs::CameraInfoConstPtr& info_msg)
{

  try {

    rgb_ptr = cv_bridge::toCvCopy(rgb_msg);
    depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    ir_ptr = cv_bridge::toCvCopy(ir_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    // hd_ptr = cv_bridge::toCvCopy(hd_msg);
    info_ptr = info_msg;
    
  } catch (cv_bridge::Exception& e) { ROS_ERROR("cv_bridge exception: %s\n", e.what()); }

  return;
};

cv_bridge::CvImagePtr& KinectSubscriber::get_cv_rgb_ptr() { return rgb_ptr; };
cv_bridge::CvImagePtr& KinectSubscriber::get_cv_depth_ptr() { return depth_ptr; };
cv_bridge::CvImagePtr& KinectSubscriber::get_cv_ir_ptr() { return ir_ptr; };
// cv_bridge::CvImagePtr& KinectSubscriber::get_cv_hd_ptr() { return hd_ptr; };
sensor_msgs::CameraInfo::ConstPtr& KinectSubscriber::get_camera_info() { return info_ptr; };


#endif // KINECT_SUBSCRIBER_CPP

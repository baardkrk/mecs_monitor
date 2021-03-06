#ifndef KINECT_SUBSCRIBER_H
#define KINECT_SUBSCRIBER_H

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

class KinectSubscriber
{
 private:
  ros::NodeHandle nh;
  image_transport::ImageTransport it;

  image_transport::SubscriberFilter rgb_sub;
  image_transport::SubscriberFilter depth_sub;
  image_transport::SubscriberFilter ir_sub;
  // image_transport::SubscriberFilter hd_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub;

  cv_bridge::CvImagePtr rgb_ptr;
  cv_bridge::CvImagePtr depth_ptr;
  cv_bridge::CvImagePtr ir_ptr;
  // cv_bridge::CvImagePtr hd_ptr;
  sensor_msgs::CameraInfo::ConstPtr info_ptr;


  // TODO: subscribe to HD image as well. However, I don't think that is possible in Kinetic,
  // since it would be >5 messages in the filter.
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image,
    // sensor_msgs::Image,
    sensor_msgs::CameraInfo> KinectSyncPolicy;

  message_filters::Synchronizer< KinectSyncPolicy > sync;

 public:
  KinectSubscriber(std::string base_topic, std::string quality);
  void callback(const sensor_msgs::ImageConstPtr& rgb_msg,
		const sensor_msgs::ImageConstPtr& depth_msg,
		const sensor_msgs::ImageConstPtr& ir_msg,
		// const sensor_msgs::ImageConstPtr& hd_msg,
		const sensor_msgs::CameraInfoConstPtr& info_msg);
  cv_bridge::CvImagePtr& get_cv_rgb_ptr();
  cv_bridge::CvImagePtr& get_cv_depth_ptr();
  cv_bridge::CvImagePtr& get_cv_ir_ptr();
  // cv_bridge::CvImagePtr& get_cv_hd_ptr();
  sensor_msgs::CameraInfo::ConstPtr& get_camera_info();
};

#endif // KINECT_SUBSCRIBER_H

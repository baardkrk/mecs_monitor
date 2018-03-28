#ifndef MECS_RENDERER_CPP
#define MECS_RENDERER_CPP

#include "mecs_renderer.h"

MecsRenderer::MecsRenderer(std::string _keypoint_topic, ros::NodeHandle& _nh)
{

  nh = _nh;
  keypoint_topic = _keypoint_topic;

  subscriber = nh.subscribe(keypoint_topic, 1, &MecsRenderer::render_keypoints, this);
  publisher = nh.advertise<visualization_msgs::MarkerArray>("skeleton_markers", 1);
  
};

std::tuple<double, double, double> MecsRenderer::hsv_to_rgb(int h, int s, int v)
{
  double r, g, b;
  // ensure valid values for hsv
  h = abs(h); s = abs(s); v = abs(v);

  if (h > 359) h = h % 359;
  if (s > 1) s = abs(s / (floor(std::log10(s)) * 10));
  if (v > 1) v = abs(v / (floor(std::log10(v)) * 10));

  float C = (float)(s * v); // should be 1 anyway
  float X = (float)(C * (1.0 - std::abs(fmod((h/60.0),2) -1.0)));
  float m = (float)(v - C);
  
  if      (h <  60) { r = C; g = X; b = 0; }
  else if (h < 120) { r = X; g = C; b = 0; }
  else if (h < 180) { r = 0; g = C; b = X; }
  else if (h < 240) { r = 0; g = X; b = C; }
  else if (h < 300) { r = X; g = 0; b = C; }
  else              { r = C; g = 0; b = X; }

  return std::make_tuple((r+m), (g+m), (b+m));
};

void MecsRenderer::render_keypoints(const mecs_monitor::ExtInfo::ConstPtr& msg)
{
  double x,y,z;
  int num_ppl, num_body_parts, person_stride, body_part_stride;
  
  num_ppl = msg->keypoints.layout.dim[0].size;
  num_body_parts = msg->keypoints.layout.dim[1].size;
  person_stride = msg->keypoints.layout.dim[0].stride;
  body_part_stride = msg->keypoints.layout.dim[1].stride;

  for (int person = 0; person < num_ppl; person++) {
    // std::cout << "Peson: " << person << " =============================================" << std::endl;
    // for (int body_part = 0; body_part < num_body_parts; body_part++) {
    //   x = msg->keypoints.data[person*person_stride + body_part*body_part_stride + 2];
    //   y = msg->keypoints.data[person*person_stride + body_part*body_part_stride + 3];
    //   z = msg->keypoints.data[person*person_stride + body_part*body_part_stride + 4];
    geometry_msgs::Point a, b;
    for (int i = 0; i < 17; i++) {
      a.x = msg->keypoints.data[ person*person_stride + keypoint_pairs[i][0]*body_part_stride + 2 ];
      a.y = msg->keypoints.data[ person*person_stride + keypoint_pairs[i][0]*body_part_stride + 3 ];
      a.z = msg->keypoints.data[ person*person_stride + keypoint_pairs[i][0]*body_part_stride + 4 ];

      b.x = msg->keypoints.data[ person*person_stride + keypoint_pairs[i][1]*body_part_stride + 2 ];
      b.y = msg->keypoints.data[ person*person_stride + keypoint_pairs[i][1]*body_part_stride + 3 ];
      b.z = msg->keypoints.data[ person*person_stride + keypoint_pairs[i][1]*body_part_stride + 4 ];

      visualization_msgs::Marker marker;
      int action = visualization_msgs::Marker::ADD;
      if (a.z < 0.2 || a.z > 6.0 ||
	  b.z < 0.2 || b.z > 6.0) action = visualization_msgs::Marker::DELETE;

      marker.header.frame_id = msg->header.frame_id;
      marker.header.stamp = ros::Time::now();

      std::stringstream namesp;
      namesp << "person_" << person << "_" << i;

      marker.ns = namesp.str();
      marker.action = action;
      marker.pose.orientation.w = 1.0;

      marker.id = i;
      marker.type = visualization_msgs::Marker::LINE_STRIP;

      marker.lifetime = ros::Duration(1.0);

      marker.scale.x = 0.05;
      std::tie(marker.color.r, marker.color.g, marker.color.b) = hsv_to_rgb(i*21, 1, 1);
      marker.color.a = .6;

      marker.points.push_back(a);
      marker.points.push_back(b);
      
      markers.markers.push_back(marker);
      marker.points.clear();
    }
    //   // std::cout << keypoint_names[body_part] << "\t : \t(" << x << ", " << y << ", " << z << ")\n";
    // }
    // std::cout << std::endl;
    
  }
  // std::cout << "published something!" << std::endl;
  publisher.publish(markers);
  markers.markers.clear();
};



#endif // MECS_RENDERER_CPP

#ifndef INFO_EXTRACTOR_CPP
#define INFO_EXTRACTOR_CPP

#include "openpose_flags.cpp"
#include "info_extractor.h"

constexpr const double Skeleton::edge_lengths[18][18];
constexpr const int Skeleton::constrained_keypoint_connections[18][2];

constexpr const int Skeleton::keypoints_head[6];
constexpr const int Skeleton::keypoints_body[7];
constexpr const int Skeleton::keypoints_legs[7];

/**
 * return a tuple of -1 if imaginary solutions
 * could speed this up by storing more values but..
 */ 
std::tuple<double, double> abc_formula(double a, double b, double c)
{
  double sq = std::pow(b,2) - 4*a*c;
  if (sq < 0) return std::make_tuple(-1.0,-1.0);

  double x_1 = (-b + std::sqrt(sq))/(2*a);
  double x_2 = (-b - std::sqrt(sq))/(2*a);

  // if (std::isnan(x_1) || std::isnan(x_2)) std::cout << "a" << a << " b" << b << " c" << c << std::endl;
  return std::make_tuple(x_1,x_2);
};

double simple_gaussian(double x, double y, double sigma, double ux=0, double uy=0)
{
  /* NOTE:
     since the gaussian will be thresholded anyway, to get better separation between the 
     classes, one should rather threshold the incoming x/y values, and change the
     gaussian so the curve flattens around the area where the threshold is located.
   */
  
  // ux, uy = 1, sigma = .41, suggested threshold .43
  return (1/(sigma * std::sqrt(2*M_PI)) * exp(- ((std::pow(x-ux,2) + std::pow(y-uy, 2)) / sigma) ));
};


InfoExtractor::InfoExtractor()
{

  // Initializing code for OpenPose =============================================================
  // Most of this is provided by the OpenPose Tutorial code.
  
  /// READING GOOGLE FLAGS
  const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
  const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
  const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

  // checking contradictory flags
  if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
    op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
  if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
    op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1", __LINE__, __FUNCTION__, __FILE__);

  // enable Google logging
  const bool enableGoogleLogging = true;

  /// INITIALIZE REQUIRED CLASSES
  scaleAndSizeExtractor = new op::ScaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
  poseExtractorCaffe = new op::PoseExtractorCaffe(poseModel, FLAGS_model_folder, FLAGS_num_gpu_start, {}, op::ScaleMode::ZeroToOne, enableGoogleLogging);
  poseRenderer = new op::PoseCpuRenderer(poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending, (float)FLAGS_alpha_pose);
  frameDisplayer = new op::FrameDisplayer("OpenPose tutorial", outputSize);
  
  /// INITIALIZE RESOURCES ON DESIRED THREAD
  poseExtractorCaffe->initializationOnThread();
  poseRenderer->initializationOnThread();

  // OpenPose Initialization Done ===============================================================

};
  
/**
 * Runs openpose on the souce image, and returns the 2D locations and score for each keypoint.
 * The inputImage parameter was kept to make it easier to test differen image streams for 
 * pose detection (ie. depth, ir, rgb).
 */
op::Array<float> InfoExtractor::run_openpose(cv::Mat inputImage, std::string pw_name="openpose_preview")
{

  ////////////// POSE ESTIMATION AND RENDERING //////////////
  const op::Point<int> imageSize{inputImage.cols, inputImage.rows};

  /// GET DESIRED SCALE VALUES
  std::vector<double> scaleInputToNetInputs;
  std::vector<op::Point<int>> netInputSizes;
  double scaleInputToOutput;
  op::Point<int> outputResolution;

  std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution) = scaleAndSizeExtractor->extract(imageSize);

  /// FORMAT INPUT IMAGE TO OPENPOSE I/O FORMATS
  const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
  auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);

  /// ESTIMATE POSEKEYPOINTS
  poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
  const auto poseKeypoints = poseExtractorCaffe->getPoseKeypoints();
  // *pose_keypoints = poseKeypoints;
  
  /// RENDER KEYPOINTS
  poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);

  /// OPENPOSE OUTPUT TO CV::MAT
  auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

  ////////////// SHOWING RESULTS AND CLOSING //////////////
  // frameDisplayer.displayFrame(outputImage, 0);
  // Alternative: cv::imshow(outputImage) + cv::waitKey(0)

  cv::imshow(pw_name, outputImage);
  cv::waitKey(1);
  
  return poseKeypoints;
};
  
/**
 * Finds all relevant features and stores them in a ROS topic. Each detected person gets it's own
 * index which is the same across all arrays.
 * Features : Float64MultiArray - 3D skeleton keypoints
 *            Image[] - HD rgb image of face RoI 
 *            Image[] - Depth image of body RoI
 * return : ROS topic with selected features
 */
mecs_monitor::ExtInfo InfoExtractor::extract(sensor_msgs::CameraInfo::ConstPtr& _camera_info,
					     cv::Mat _rgb, cv::Mat _depth, cv::Mat _ir)
{
  mecs_monitor::ExtInfo e_info;

  // Updating frames and camera info ======================================
  
  camera_info = _camera_info;

  rgb = _rgb; depth = _depth; ir = _ir;
  cv::Mat color_ir;
  cv::cvtColor(ir, color_ir, cv::COLOR_GRAY2BGR);

  // Generating information ===============================================

  // op::Array<float> keypoints = run_openpose(color_ir, "ir_preview");
  op::Array<float> keypoints = run_openpose(rgb, "rgb_preview");
  std_msgs::Float64MultiArray keypoint_array = get_3d_keypoints(keypoints);


  // Testing ==============================================================
  test();

  // Packaging information to message =====================================
  
  e_info.header.stamp = ros::Time::now();
  e_info.header.frame_id = camera_info->header.frame_id;
  e_info.keypoints = keypoint_array;

  return e_info;
};

void InfoExtractor::test()
{
  // std::cout << Eigen::Vector3d(1, 2, 3).transpose() << std::endl;
  
  // cv::imshow("ir_image", ir);
  // cv::imshow("rgb_image", rgb);
  // cv::imshow("depth_image", depth);

  // cv::Mat histeqlized;
  // depth.convertTo(histeqlized, CV_8UC1);
  
  // cv::equalizeHist(histeqlized, histeqlized);

  // cv::imshow("equalized", histeqlized);
  return;
};

/**
 * maps a point between a sec image and a dst image.
 */
cv::Point InfoExtractor::img_map(cv::Mat src, cv::Mat dst, cv::Point point)
{
  double dX, dY;
  dX = dst.cols/src.cols;
  dY = dst.rows/src.rows;

  return cv::Point(point.x*dX, point.y*dY);
};

/**
 * Takes a point (mapped to the DEPTH image!) and returns the 3D location
 * of that point based on the camera parameters.
 * 
 * Parameters:
 * point    - 2D image location of the 3D point we want to get.
 * dZ       - adjust the Z value of the point, to account for surface detection
 * override - used to get the 3D point even if it was detected too close to the camera
 */
Eigen::Vector3d InfoExtractor::project_to_3d(cv::Point point, double dZ=0, bool override=false)
{
  double Z = depth.at<double>(point.y, point.x)+dZ, X, Y;
  double t_zero = 0.05; // too close to the camera. (5 cm)
  if (Z < t_zero+dZ) Z = 0.0;
  if (override) Z = 1.0;
    
  double fx = camera_info->K[0],
    fy = camera_info->K[4],
    Cx = camera_info->K[2],
    Cy = camera_info->K[5];
 
  Y = ((double)point.y - Cy) * Z / fy;
  X = ((double)point.x - Cx) * Z / fx;
    
  
  return Eigen::Vector3d(X, Y, Z);
};

/**
 * Takes a 3D point and projects it into the corresponding pixel values of the
 * depth image.
 */
cv::Point InfoExtractor::project_to_img(Eigen::Vector3d point)
{
  int row, col;
  double X = point(0), Y = point(1), Z = point(2);
  double fx = camera_info->K[0],
    fy = camera_info->K[4],
    Cx = camera_info->K[2],
    Cy = camera_info->K[5];
 
  row = int(Cy + Y * fy / Z);
  col = int(Cx + X * fx / Z);

  return cv::Point(col, row);
};

/**
 * Gets a with and a height in meters of a rectangular region of interest and a seed point.
 * Optional parameters are displacement (in meters agian) of the seed point in relation
 * to the extracted region. 
 * The function returns a tuple with cv::Points that describes the upper left and lower right
 * corner of the region of interest. These points are given with the pixel positions of the
 * depth image, and needs to be mapped if another input stream is desired.
 */
std::tuple<cv::Point, cv::Point> InfoExtractor::get_roi(Eigen::Vector3d point,
							double width, double height,
							double dX=0, double dY=0)
{

  cv::Point upper_left = project_to_img(Eigen::Translation<double, 3>(-width+dX, height+dY, 0) * point);
  cv::Point lower_right = project_to_img(Eigen::Translation<double, 3>(width+dX, -height+dY, 0) * point);

  
  return std::make_tuple(upper_left, lower_right);
};

/**
 * Takes the 2D keypoints from OpenPose, and places them into 3D space
 * using the depth image.
 * It then packages the 3D keypoints into an array containing the 2D image locations,
 * 3D coordinates for each keypoint and the score given from OpenPose.
 */
std_msgs::Float64MultiArray InfoExtractor::get_3d_keypoints(op::Array<float> keypoints)
{
  std_msgs::Float64MultiArray kp_arr;
  double row, col, X, Y, Z, score;


  kp_arr.layout.data_offset = 0;

  kp_arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
  kp_arr.layout.dim[0].label = "person";
  kp_arr.layout.dim[0].size = keypoints.getSize(0);
  kp_arr.layout.dim[0].stride = keypoints.getSize(1) * (keypoints.getSize(2) + 3);
  
  kp_arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
  kp_arr.layout.dim[1].label = "body part";
  kp_arr.layout.dim[1].size = keypoints.getSize(1);
  kp_arr.layout.dim[1].stride = keypoints.getSize(2) + 3;
  
  kp_arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
  kp_arr.layout.dim[2].label = "location";
  kp_arr.layout.dim[2].size = keypoints.getSize(2) + 3; // row,col,score is getSize(2). +3 for xyz
  kp_arr.layout.dim[2].stride = 1;

  kp_arr.data.clear();

  for (auto person = 0; person < keypoints.getSize(0); person++) {

    // collecting all data for this person so we can refine it in a separate step.
    std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > tmp_loc;
    
    for (auto body_part = 0; body_part < keypoints.getSize(1); body_part++) {

      col = (double)keypoints[{person, body_part, 0}];
      row = (double)keypoints[{person, body_part, 1}];
      score = (double)keypoints[{person, body_part, 2}];

      Eigen::Vector3d kp3d = project_to_3d(cv::Point((int)row, (int)col), kp_Zd[body_part]);
      Eigen::Vector2d kp2d(row, col);
      
      tmp_loc.push_back(std::make_tuple(kp2d, kp3d, score));
    }

    Skeleton s(tmp_loc, this);
    // constrain_skeleton(&tmp_loc);
    tmp_loc = s.constrain_skeleton();
    
    // pushing back each refined keypoint ===================================================
    for (std::vector< std::tuple<Eigen::Vector2d,
	   Eigen::Vector3d, double> >::iterator it = tmp_loc.begin();
	 it != tmp_loc.end(); ++it) {

      Eigen::Vector2d kp2d; Eigen::Vector3d kp3d; double score;
      std::tie(kp2d, kp3d, score) = *it;

      kp_arr.data.push_back(kp2d(0)); // row
      kp_arr.data.push_back(kp2d(1)); // col
      kp_arr.data.push_back(kp3d(0)); // X
      kp_arr.data.push_back(kp3d(1)); // Y
      kp_arr.data.push_back(kp3d(2)); // Z
      kp_arr.data.push_back(score);

    }
  }
  return kp_arr;
};

// ========================= END INFO EXTRACTOR ==================================

Skeleton::Skeleton(std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector3d, double>> original, InfoExtractor *_extractor)
{
  keypoints = original;
  
};

std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > Skeleton::constrain_skeleton() {
  return keypoints;
};

// ============================ END SKELETON =====================================

Subgraph::Subgraph(std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > _kp, int _seed, Skeleton *_parent)
{
  keypoints = _kp;
  seed = _seed;
  parent_skeleton = _parent;

  if (_seed == 0 || _seed >= 14 && _seed <= 17) {
    subgraph_keypoints = parent_skeleton->keypoints_head;
  } else if (_seed >= 2 && _seed <= 7) {
    subgraph_keypoints = parent_skeleton->keypoints_body;
  } else {
    subgraph_keypoints = parent_skeleton->keypoints_legs;
  }

  // determine initial scale based on original keypoints
  // Eigen::MatrixXd limb_scores(18,18), limb_lengths(18,18);
  // limb_lengths = Eigen::MatrixXd::Zero(18,18);
  // limb_scores = get_limb_scores();

  // double scale = 0;
  // for (int i = 0; i < subgraph_keypoints[i] != -1; i++) {
    
    
  // }
  scale = 1.75;
  
  recursive_constrain(seed);

  // centering the subgraph
  // calculate move vector
  Eigen::Vector3d move_vec; move_vec << 0,0,0;
  double score_sum = 0, scale_factor = 1.0;
  for (int i = 0; subgraph_keypoints[i] != -1; i++) {
    move_vec += (std::get<1>(keypoints.at(i)) - std::get<1>(_kp.at(i))) * std::get<2>(keypoints.at(i));
    score_sum += std::get<2>(keypoints.at(i));
    
  }
  move_vec = move_vec / score_sum;

  for (int i = 0; subgraph_keypoints[i] != -1; i++) {
    std::get<1>(keypoints.at(subgraph_keypoints[i])) += move_vec;
  }
  
  // adjust scale factor 
};

std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > Subgraph::get_keypoints()
{return keypoints;};

double Subgraph::get_scale()
{
  return scale;
};

Eigen::MatrixXd Subgraph::get_limb_scores()
{

  Eigen::MatrixXd limb_scores(18,18);
  limb_scores = Eigen::MatrixXd::Zero(18,18);
  
  for (int i = 0; subgraph_keypoints[i] != -1; i++) {
    int parent = subgraph_keypoints[i];

    for (int j = 0; j < 2; j++) {
      int child = parent_skeleton->constrained_keypoint_connections[parent][j];

      double p_score = std::get<2>(keypoints.at(parent));
      double c_score = std::get<2>(keypoints.at(child));

      double score = simple_gaussian(p_score, c_score, 0.41, 1,1);
      double threshold = .43;
      score = (score > threshold) ? score : 0;
      
      limb_scores(parent, child) = score;
      limb_scores(child, parent) = score;
    }
  }

  return limb_scores;
};

/**
 * Places the keypoint with ID c_id in relation to the parent keypoint with ID p_id.
 * This means that keypoint P_ID will remain FIXED after the method has run.
 *
 * If either the parent or the child ID is -1 it means we have reached the end of
 * a graph. There is nothing more to be done, so the method returns.
 */
void Subgraph::place_keypoint(int p_id, int c_id)
{
  if (p_id == -1 || c_id == -1) return;
  
  if (std::get<1>(keypoints.at(c_id)).norm() == .0) {   
    std::get<1>(keypoints.at(c_id)) = unobserved_child(p_id, c_id);
    return;
  }
  double limb_length = parent_skeleton->edge_lengths[p_id][c_id] * scale;

  // TODO clean this, so we just pass the id's to the push_vector function.
  // check that shortest distance is less than required distance.
  Eigen::Vector3d child = std::get<1>(keypoints.at(c_id)),
    parent = std::get<1>(keypoints.at(p_id));

  if (std::isnan(parent(2))) std::cout << "parent: " << p_id << "\t";
  if (std::isnan(child(2))) std::cout << "child: " << c_id;
  if (std::isnan(parent(2)) || std::isnan(child(2)))  std::cout << std::endl;
  
  std::get<1>(keypoints.at(c_id)) = push_vector(parent,child,limb_length);
  return;
};

/**
 * Places the keypoint in case it is no way to infer it. The keypoint is placed directly
 * down in case of an appendage such as a limb (note that this will create some strange 
 * head shapes.) or on the line created by the parent point, projected into the xy-plane.
 */
Eigen::Vector3d Subgraph::unobserved_child(int p_id, int c_id)
{
  // std::cout << "unobserved " << c_id << "(" << p_id << ")" << std::endl;
  // could be unobserved in 2d or too close to the camera
  // check if next is observed. (if it is, we do keypoint interpolation)
  int next = (p_id == parent_skeleton->constrained_keypoint_connections[c_id][0]) ?
    parent_skeleton->constrained_keypoint_connections[c_id][1] : parent_skeleton->constrained_keypoint_connections[c_id][0];

  if (next != -1)
    if (std::get<1>(keypoints.at(next)).norm() == .0)
      return keypoint_interpolation(p_id, c_id, next);

  double limb_length = parent_skeleton->edge_lengths[p_id][c_id] * scale;
  
  // check 2d observation
  int exception_list[] = {2,5,8,11};
  if (std::get<0>(keypoints.at(c_id)).norm() == .0) {
    for (int i = 0; i < 4; i++) {
      if (c_id == exception_list[i]) {
	Eigen::Matrix3d tr;
	tr << limb_length, 0, 0, 0, limb_length, 0, 0, 0, 0;
	return (std::get<1>(keypoints.at(p_id)) + tr * std::get<1>(keypoints.at(p_id)).normalized());
      }
    }
    // placing keypoint straight "down" from parent.
    // A better method based on constraining the limb within the visual hull could be implemented
    // std::get<1>(keypoints.at(c_id)) = std::get<1>(keypoints.at(p_id)) + Eigen::Vector3d(limb_length,0,0);
    return (std::get<1>(keypoints.at(p_id)) + Eigen::Vector3d(limb_length,0,0));
  };
};

/**
 * Moves the point c_id. This is used if the next point in a constrained subgraph (n_id) is
 * well observed, but the child point (c_id) is not. 
 * This works by setting c_id to the point on the circle of intersection between the spheres
 * with origin in p_id and n_id and the appropriate limb lengths. 
 *
 * TODO: if there is NO INTERSECTION between the spheres, this just runs the unobserved child
 * algorithm.
 */
Eigen::Vector3d Subgraph::keypoint_interpolation(int p_id, int c_id, int n_id)
{
  Eigen::Vector3d a = std::get<1>(keypoints.at(p_id)), b = std::get<1>(keypoints.at(n_id));
  double r_a = parent_skeleton->edge_lengths[p_id][c_id] * scale, r_b = parent_skeleton->edge_lengths[c_id][n_id] * scale,
    d = (a-b).norm();

  double x = (std::pow(d,2) - pow(r_b,2) + pow(r_a,2))/(2*d);
  double h = (1/d) * std::sqrt((-d+r_b-r_a)*(-d-r_b-r_a)*(-d+r_b+r_a)*(d+r_b+r_a));

  double dir_x=1, dir_y=1,
    dir_z = (-a(0)-a(1)) / a(2);
  
  Eigen::Vector3d direction(dir_x, dir_y, dir_z);
  direction.normalize();
  
  // std::get<1>(keypoints.at(c_id)) = a + x*(b-a).normalized() + (h/2)*direction;
  return (a + x*(b-a).normalized() + (h/2)*direction);
};

/**
 * Creates a vector between the "pushed" point and origo. We push or pull the "pushed" point
 * so the distance between it and the fixed point is equal to the length parameter.
 * If the shoretest distance between the fixed point and the line created by the pushed point
 * and origo is greater than the length parameter, the "pushed" point is moved to the 
 * nearest point of the line while still "length" away from the fixed point. 
 * (I.e. "length" units away from the fixed point on the line through the fixed point, 
 * perpendicular to the line created by the pushed point and origo.)
 */
Eigen::Vector3d Subgraph::push_vector(Eigen::Vector3d fixed, Eigen::Vector3d pushed, double length)
{
  // checking projected distance
  Eigen::Vector3d p_dist = (fixed.dot(pushed) / pushed.dot(pushed)) * pushed;
  if ((fixed - p_dist).norm() > length)
    return fixed + length * (fixed-p_dist).normalized(); // we go toward the line as far as possible
  
  pushed.normalize();
  double a = std::pow(pushed.norm(), 2),
    b = -2 * (fixed.dot(pushed)),
    c = std::pow(fixed.norm(), 2) - std::pow(length, 2);

  // if (std::isnan(a) || std::isnan(b) || std::isnan(c))
  //   std::cout << "pushed " << pushed.transpose() << ", \tfixed " << fixed.transpose() << std::endl;
  double x_1, x_2;
  std::tie(x_1, x_2) = abc_formula(a, b, c);
  
  return std::max(x_1, x_2) * pushed;
};

/**
 * Starts constraining each child of the current keypoint
 */
void Subgraph::recursive_constrain(int p_id)
{

  for (int i = 0; i < 2; i++) {
    int child = parent_skeleton->constrained_keypoint_connections[p_id][i];

    if (child != -1) {
      // first we move the child keypoint with the current keypoint fixed.
      place_keypoint(p_id, child);
      
      // then we move the next keypoint.
      recursive_constrain(child);
    }
  }

  return;
};

double Subgraph::get_limb_length(std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > graph, int parent, int child)
{
  return (std::get<1>(graph.at(parent)) - std::get<1>(graph.at(child))).norm();
};



// ============================ END SUBGRAPH =====================================



#endif // INFO_EXTRACTOR_CPP

#ifndef INFO_EXTRACTOR_CPP
#define INFO_EXTRACTOR_CPP

#include "openpose_flags.cpp"
#include "info_extractor.h"

// linker needs references to the constant expressions in the header file
constexpr const double InfoExtractor::edge_lengths[18][18];
constexpr const int InfoExtractor::constrained_keypoint_connections[18][2];

InfoExtractor::InfoExtractor() :
  traversal_keypoint_connections({
    {1,14,15}, {0,2,5,8,11}, {1,3}, {2,4}, {3}, {1,6}, {5,7}, {6}, {1,9}, {8,10}, {9}, {1,12}, {11,13}, {12}, {0,16}, {0,17}, {14}, {15}
  })
{

  // norm_constr << .105, .035, .058, .035, .058, .259, .186, .146, .186, .146, .191, .245, .246, .245, .246;

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
 */
Eigen::Vector3d InfoExtractor::project_to_3d(cv::Point point, double dZ=0, bool override=false)
{
  double Z = depth.at<double>(point.y, point.x)+dZ, X, Y;
  double t_zero = 0.05; // too close to the camera. (5 cm)
  if (Z < t_zero+dZ && !override) Z = 0.0;
  
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

std::list<int> InfoExtractor::Skeleton::sort_keypoints(int *subgraph)
{
  std::list<int> idx; // first index has highest score
  std::list<int>::iterator idx_it;

  // std::vector< std::tuple<Eigen::Vector2d,
  // 			  Eigen::Vector3d, double> >::iterator it = keypoints->begin();

  
  // initial value
  idx.push_back(*subgraph);

  while (*subgraph != -1) {
    bool ins_back = true;
    int i = 0;
    for (idx_it = idx.begin(); idx_it != idx.end(); idx_it++) {
      if (std::get<2>(keypoints->at(*subgraph)) > std::get<2>(keypoints->at(*idx_it))) {
    	// insert this index
    	idx.insert(idx_it, i);
    	ins_back = false;

    	break;
      }
      i++;
    }
    
    if (ins_back)
      idx.push_back(*subgraph);
    
    
    subgraph++;
  }
  
  // // excluding keypoints at head and neck.
  // for (int i = 3; i < keypoints->size()-4; i++) {
  //   bool ins_back = true;
    
  //   for (idx_it = idx.begin(); idx_it != idx.end(); idx_it++) {
  //     if (std::get<2>(keypoints->at(i)) > std::get<2>(keypoints->at(*idx_it))) {
  // 	// insert this index
  // 	idx.insert(idx_it, i);
  // 	ins_back = false;

  // 	break;
  //     }
  //   }
    
  //   if (ins_back)
  //     idx.push_back(i);
  // }

  return idx;
};

void InfoExtractor::constrain_skeleton(std::vector< std::tuple<Eigen::Vector2d,
				       Eigen::Vector3d, double> >* keypoints)
{
  // start by finding the best of the innermost keypoints (hips, shoulder)
  // std::list<int> idxs = sort_keypoints(keypoints);
  // int N = 8;

  // for (std::list<int>::const_iterator ci = idxs.begin(); ci != idxs.end(); ++ci)
  //   std::cout << *ci << "\t" << std::get<2>(keypoints->at(*ci)) << std::endl;
  // std::cout << std::endl;

  // if (N <= idxs.size()) {
    
  //   std::vector<Skeleton> skeletons;
  //   std::vector<Skeleton>::iterator skelit;
  //   // create skeletons for N best
  //   for (int i = 0; i < N; i++) {
  //     std::list<int>::iterator it = std::next(idxs.begin(), i);

  //     // create 3 different seeds
      
  //     // Only create skeleton if seed is t units away from the camera, and also, that it is
  //     // indeed an observed keypoint
  //     double t = 0.1;
  //     if (std::get<1>(keypoints->at(*it)).norm() > t &&
  // 	  !std::isnan(std::get<1>(keypoints->at(*it))(0))) {
  // 	// recursive_skeleton
  skeletons.push_back(Skeleton(*keypoints, this));
    //   }
    // }
  
    // for (skelit = skeletons.begin(); skelit < skeletons.end(); skelit++) {
    //   std::cout << "Skeleton:" << std::endl;
    //   for (int i = 0; i < 18; i++) {
    // 	//std::cout << std::get<1>(keypoints->at(i)).transpose() << "  --  " << std::get<1>((*skelit).get_skeleton().at(i)).transpose() << std::endl;
    //   }
    // }

	//} // N <= idxs.size

  // check error between the different trees
  
  // Scoring method 1: distance to well observed keypoints

  // Scoring method 2: how well limbs fit with cloud point data in cylinders around limbs
  return;
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

    // Skeleton s(tmp_loc, *this);
    constrain_skeleton(&tmp_loc);
    
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


InfoExtractor::Skeleton::Skeleton(std::vector< std::tuple<Eigen::Vector2d,
				  Eigen::Vector3d, double> > _keypoints, int _seed,
  				  InfoExtractor *_p_ext) :
  p_ext(_p_ext),
  seed(_seed),
  visited_keypoints{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
{
  scale = 1.75; // TODO
  keypoints = _keypoints;
  
  // generate seeds from the info extractor
  std::list<int> head_kp = sort_keypoints(*keypoints_head[0]);
  std::list<int> body_kp = sort_keypoints(*keypoints_body[0]);
  std::list<int> legs_kp = sort_keypoints(*keypoints_legs[0]);

  int N = 3; // nubmber of seeds per graph

  std::vector< std::tuple<Eigen::Vector2d,
			  Eigen::Vector3d, double> > head_array[N];
  std::vector< std::tuple<Eigen::Vector2d,
			  Eigen::Vector3d, double> > body_array[N];
  std::vector< std::tuple<Eigen::Vector2d,
			  Eigen::Vector3d, double> > legs_array[N];

  std::list<int>::iterator list_it;
  
  for (int i = 0; i < N; i++) {
    list_it = head_kp.begin();
    head_array[i] = recursive_constrain(std::next(list_it, i));

    list_it = body_kp.begin();
    body_array[i] = recursive_constrain(std::next(list_it, i));

    list_it = list_kp.begin();
    legs_array[i] = recursive_constrain(std::next(list_it, i));

    
  }
  
  // constrain the skeletons
  // recursive_constrain(seed);
  

  
  // center and update skeleton scales
  
  // score skeletons (distance from well-observed points, over threshold)

  // update scale and recenter

  // combine
  
  
  // the neck is not set by the constrain method, so we set it here.
  // std::get<1>(keypoints.at(1)) = std::get<1>(keypoints.at(2)) + (std::get<1>(keypoints.at(2))-std::get<1>(keypoints.at(5)))/2;
  
};

std::vector< std::vector< std::tuple<Eigen::Vector2d,
				     Eigen::Vector3d, double> > > InfoExtractor::Skeleton::get_skeletons(int num_seeds, int seeds[num_seeds])
{
  std::vector< std::vector< std::tuple<Eigen::Vector2d,
			  Eigen::Vector3d, double> > > skeleton_vector;
  double t = .4;
  for (int i = 0; i < num_seeds; i++) {
    if (std::get<2>(keypoints.at(seeds[i])) > t)
      skeleton_vector.push_back(recursive_constrain(seeds[i]))
  }
  
  return skeleton_vector;
};

std::vector< std::tuple<Eigen::Vector2d,
			Eigen::Vector3d, double> > InfoExtractor::Skeleton::recursive_constrain(int p_id)
{

  
};


void InfoExtractor::Skeleton::recursive_constrain(int p_id)
{
  if (std::isnan(std::get<1>(keypoints.at(p_id))(0))) std::cout << "GOT A NAN " << p_id  << "seed: " << seed << std::endl;
  
  if (visited_keypoints[p_id] == 1) return;
  visited_keypoints[p_id] = 1;

  // constraining edges
  place_keypoint(p_id, constrained_keypoint_connections[p_id][0]);
  place_keypoint(p_id, constrained_keypoint_connections[p_id][1]);

  // now, the two children SHOULD be OK. (check to make sure)
  if (constrained_keypoint_connections[p_id][0] != -1)
    recursive_constrain(constrained_keypoint_connections[p_id][0]);
      
  if (constrained_keypoint_connections[p_id][1] != -1)
    recursive_constrain(constrained_keypoint_connections[p_id][1]);
  
  // traversing OK!
  // std::vector<int>::iterator trav_it;
  // std::vector<int> children = p_ext->traversal_keypoint_connections.at(p_id);
  // for (trav_it = children.begin(); trav_it != children.end(); trav_it++)
  //   recursive_constrain(*trav_it);

  return;
};

std::vector< std::tuple<Eigen::Vector2d,
			Eigen::Vector3d, double> > InfoExtractor::Skeleton::get_skeleton()
{
  return keypoints;
};

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

/**
 * Decides the next keypoint in the graph, constrained by the parent.
 */
void InfoExtractor::Skeleton::place_keypoint(int p_id, int c_id)
{
  if (p_id == -1 || c_id == -1) return;
  
  if (std::get<1>(keypoints.at(c_id)).norm() == .0)      
    return unobserved_child(p_id, c_id);

  double limb_length = edge_lengths[p_id][c_id] * scale;
  
  // check that shortest distance is less than required distance.
  Eigen::Vector3d child = std::get<1>(keypoints.at(c_id)),
    parent = std::get<1>(keypoints.at(p_id));

  if (std::isnan(parent(0))) std::cout << "parent: " << p_id << "\t";
  if (std::isnan(child(0))) std::cout << "child: " << c_id;
  if (std::isnan(parent(0)) || std::isnan(child(0)))  std::cout << std::endl;
  
  std::get<1>(keypoints.at(c_id)) = push_vector(parent,child,limb_length);
  return;
};

Eigen::Vector3d InfoExtractor::Skeleton::push_vector(Eigen::Vector3d fixed, Eigen::Vector3d pushed,
						     double length)
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

void InfoExtractor::Skeleton::unobserved_child(int p_id, int c_id)
{
  // std::cout << "unobserved " << c_id << "(" << p_id << ")" << std::endl;
  // could be unobserved in 2d or too close to the camera
  // check if next is observed. (if it is, we do keypoint interpolation)
  int next = (p_id == constrained_keypoint_connections[c_id][0]) ?
    constrained_keypoint_connections[c_id][1] : constrained_keypoint_connections[c_id][0];

  if (next != -1)
    if (std::get<1>(keypoints.at(next)).norm() == .0)
      return keypoint_interpolation(p_id, c_id, next);

  double limb_length = edge_lengths[p_id][c_id] * scale;
  
  // check 2d observation
  if (std::get<0>(keypoints.at(c_id)).norm() == .0) {
    // placing keypoint straight "down" from parent.
    // A better method based on constraining the limb within the visual hull could be implemented
    std::get<1>(keypoints.at(c_id)) = std::get<1>(keypoints.at(p_id)) + Eigen::Vector3d(limb_length,0,0);
    return;
  }

  // keypoint was too close to camera, so we now set it out
  Eigen::Vector3d pt = p_ext->project_to_3d(cv::Point(std::get<0>(keypoints.at(c_id))(0),
  						      std::get<0>(keypoints.at(c_id))(1)), 1.0, true);

  std::get<1>(keypoints.at(c_id)) = push_vector(std::get<1>(keypoints.at(p_id)), pt, limb_length);
  return;  
};

void InfoExtractor::Skeleton::keypoint_interpolation(int p_id, int c_id, int n_id)
{
  Eigen::Vector3d a = std::get<1>(keypoints.at(p_id)), b = std::get<1>(keypoints.at(n_id));
  double r_a = edge_lengths[p_id][c_id] * scale, r_b = edge_lengths[c_id][n_id] * scale,
    d = (a-b).norm();

  double x = (std::pow(d,2) - pow(r_b,2) + pow(r_a,2))/(2*d);
  double h = (1/d) * std::sqrt((-d+r_b-r_a)*(-d-r_b-r_a)*(-d+r_b+r_a)*(d+r_b+r_a));

  double dir_x=1, dir_y=1,
    dir_z = (-a(0)-a(1)) / a(2);
  
  Eigen::Vector3d direction(dir_x, dir_y, dir_z);
  direction.normalize();
  
  std::get<1>(keypoints.at(c_id)) = a + x*(b-a).normalized() + (h/2)*direction;
  return;
};




#endif // INFO_EXTRACTOR_CPP

#ifndef INFO_EXTRACTOR_CPP
#define INFO_EXTRACTOR_CPP

#include "openpose_flags.cpp"
#include "info_extractor.h"

InfoExtractor::InfoExtractor() :
  norm_constr(15)
{

  norm_constr << .105, .035, .058, .035, .058, .259, .186, .146, .186, .146, .191, .245, .246, .245, .246;

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

  op::Array<float> keypoints = run_openpose(color_ir, "ir_preview");
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
Eigen::Vector3d InfoExtractor::project_to_3d(cv::Point point, double dZ=0)
{
  double Z = depth.at<double>(point.y, point.x)+dZ, X, Y;
  
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

      Eigen::Vector3d kp3d = project_to_3d(cv::Point((int)row, (int)col));
      Eigen::Vector2d kp2d(row, col);
      
      tmp_loc.push_back(std::make_tuple(kp2d, kp3d, score));
    }

    // constraining step ===================================================================

    Eigen::VectorXd limb_scores(15), limb_lengths(15);
    double scale = .0;
    
    // preparing vector of actual lengths and calculating scale
    for (int i = 0; i < 15; i++) {
      
      Eigen::Vector2d a_kp2d, b_kp2d;
      Eigen::Vector3d a_kp3d, b_kp3d;
      double a_score, b_score;
      
      std::tie(a_kp2d, a_kp3d, a_score) = tmp_loc.at(constrained_limb_pairs[i][0]);
      std::tie(b_kp2d, b_kp3d, b_score) = tmp_loc.at(constrained_limb_pairs[i][1]);

      limb_scores(i) = pow(std::min(a_score,b_score),2)*(a_score + b_score)/2.0;
      limb_lengths(i) = (a_kp3d - b_kp3d).norm();

      //std::cout << constrained_limbs_names[i] << ":\t" << limb_lengths(i) << "\t\t" << limb_scores(i) << std::endl;

      scale += (limb_lengths(i)/norm_constr(i)) * limb_scores(i);
      
    }
    
    scale = scale/limb_scores.sum();
    // until scale actually works, TODO
    scale = 1.72;
    // std::cout << "height, p" << person << ": " << scale << "=================================" << std::endl;

    // modifying limb lengths based on observations and set tolerances
    // skipping tolerances for now
    
    // placing out the constrained skeleton

    // starting with the "innermost" keypoints
    // checking which of the innermost has the highest score, and fixing that in place first.
    
    // check  keypoint 8/11 and 5/2
    // Eigen::Vector2d a_kp2d;
    // Eigen::Vector3d a_kp3d;
    // double a_score;

    if (std::get<2>(tmp_loc.at(8)) > std::get<2>(tmp_loc.at(11)))
      std::cout << "RIGHT" << std::endl;
    else
      std::cout << "LEFT" << std::endl;
    
    // std::tie(a_kp2d, a_kp3d, a_score) = tmp_loc.at(8);
    // if (a_score == tmp_loc.at(8)[4])
    // largest probability gets placed first.
    // then, normal placement algorithm places the rest of the points. 
    

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


#endif // INFO_EXTRACTOR_CPP

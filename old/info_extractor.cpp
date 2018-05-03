#ifndef INFO_EXTRACTOR_CPP
#define INFO_EXTRACTOR_CPP

#include "openpose_flags.cpp"
#include "info_extractor.h"

// TODO: fix the obvious security holes by initializing the camera_info pointer and the matrices.

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

mecs_monitor::ExtInfo InfoExtractor::extract(sensor_msgs::CameraInfo::ConstPtr& _camera_info,
					     cv::Mat _rgb, cv::Mat _depth, cv::Mat _ir)
{
  mecs_monitor::ExtInfo e_info;
  // Updating frames and camera info ======================================
  
  camera_info = _camera_info;

  rgb = _rgb; depth = _depth; ir = _ir;
  cv::Mat color_ir;
  cv::cvtColor(ir, color_ir, cv::COLOR_GRAY2BGR);

  // other blurring methods could be utilized, such as average
  cv::GaussianBlur(depth, blurred_depth, cv::Size(5,5), 0);

  // Generating information ===============================================

  op::Array<float> keypoints = run_openpose(color_ir, "ir_preview");
  std_msgs::Float64MultiArray keypoint_array = get_3d_keypoints(keypoints);
  // limbs described in the coco model

  // TBD: get RoIs remember to use the same keypoint indices!


  // testing

  cv::Mat depth_edges, struct_element;
  struct_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4,4));
    
  depth.convertTo(depth_edges, CV_8UC1, 70); // 70 was chosen arbitrarily to increase contrast
 
  // double X,Y,Z;
  // std::tie(X,Y,Z) = project_3d(depth_edges.rows/2, depth_edges.cols/2);

  Eigen::Vector3d pt;
  // pt(0) = X; pt(1) = Y; pt(2) = Z; // a lot of trouble for keeping the 'tie' function...
  //std::cout << "middle screen point (" << X << "," << Y << "," << Z << ")" << std::endl;

  // accessing the keypoint array
  if (keypoint_array.layout.dim[0].size > 0) {
    
    pt(0) = keypoint_array.data[0*keypoint_array.layout.dim[1].stride + 2];
    pt(1) = keypoint_array.data[0*keypoint_array.layout.dim[1].stride + 3];
    pt(2) = keypoint_array.data[0*keypoint_array.layout.dim[1].stride + 4];

    double width = .2, height = .2;
    
    cv::Mat roi, tmp;
    roi = get_roi(pt, width, height, color_ir);

    if (pt(2) > .0) {
      
      Eigen::Matrix<int, 2, 4> corners;
      corners << get_window_corners(pt, height, width);
      cv::Point pt1(corners(1,0), corners(0,0)), pt2(corners(1,3), corners(0,3));
    
      // removing noise by opening and closing the image
      // cv::morphologyEx(roi, roi, cv::MORPH_OPEN, struct_element,  cv::Point(-1,-1), 1);
      // cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, struct_element,  cv::Point(-1,-1), 1);

    
      // // threshold using otsu's method
      // double otsu_thresh = cv::threshold(roi, tmp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
      // cv::Canny(roi, roi, otsu_thresh*.5, otsu_thresh);
      // cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, struct_element);

      // cv::floodFill(roi, cv::Point(roi.cols/2, roi.rows/2), cv::Scalar(255));
      
      // cv::cvtColor(depth_edges, depth_edges, CV_GRAY2BGR);
 
      cv::rectangle(depth_edges, pt1, pt2, cv::Scalar(0,0,255), 3);
      cv::circle(depth_edges, pt1, 5, cv::Scalar(255,0,0), 2);
      cv::circle(depth_edges, pt2, 5, cv::Scalar(0,255,0), 2);
    }
    cv::imshow("Region of Interest", roi);
  }

  // std::cout << "top corner: (" << rect(0,0) << "," << rect(1,0) << ")" << std::endl;
 
  cv::imshow("depth", depth_edges);
  cv::waitKey(1);
  // Packaging information to message =====================================
  
  e_info.header.stamp = ros::Time::now();
  e_info.header.frame_id = camera_info->header.frame_id;
  e_info.keypoints = keypoint_array;

  return e_info;
};

// Any depth that is nearer to the camera than 0.2m is invalid
std::tuple<double, double, double> InfoExtractor::project_3d(int row, int col)
{
  double Z = depth.at<double>(row, col), X, Y;
  double fx = camera_info->K[0],
    fy = camera_info->K[4],
    Cx = camera_info->K[2],
    Cy = camera_info->K[5];

  Y = ((double)row - Cy) * Z / fy;
  X = ((double)col - Cx) * Z / fx;

  return std::make_tuple(X, Y, Z);
};

// Projects a 3d point into the pixel image. (basically 
Eigen::Matrix<int, 2, 1> InfoExtractor::project_to_img(Eigen::Vector3d point)
{
  double X, Y, Z;
  Eigen::Matrix<int, 2, 1> point2d;
  point2d << Eigen::Matrix<int, 2, 1>::Zero();
  
  X = point(0);   Y = point(1);  Z = point(2);

  point2d(0) = int(camera_info->K[5] + Y * camera_info->K[4] / Z); // row
  point2d(1) = int(camera_info->K[2] + X * camera_info->K[0] / Z); // col
  
  return point2d;
};

// Returns the pixel coordinates for the specified window around the specified point
// Height and width are given in meters
Eigen::Matrix<int, 2, 4> InfoExtractor::get_window_corners(Eigen::Vector3d point, double height, double width)
{
  Eigen::Matrix<int, 2, 4> points;
  points << Eigen::Matrix<int, 4, 2>::Zero();
  
  // top left corner  
  points.block(0,0,2,1) = project_to_img(Eigen::Translation<double, 3>(-width, -height, 0) * point);
  // top right corner
  points.block(0,1,2,1) = project_to_img(Eigen::Translation<double, 3>(width, -height, 0) * point);
  // lower right corner
  points.block(0,2,2,1) = project_to_img(Eigen::Translation<double, 3>(-width, height, 0) * point);
  // lower left corner
  points.block(0,3,2,1) = project_to_img(Eigen::Translation<double, 3>(width, height, 0) * point);

  return points;
};

cv::Point InfoExtractor::project_to_cv_img(Eigen::Vector3d point)
{
  cv::Point res;
  double X,Y,Z;

  X = point(0);   Y = point(1);  Z = point(2);

  res.y = int(camera_info->K[5] + Y * camera_info->K[4] / Z); // row
  res.x = int(camera_info->K[2] + X * camera_info->K[0] / Z); // col

  return res;
};

cv::Mat InfoExtractor::get_roi(Eigen::Vector3d pt, double width, double height, cv::Mat src)
{
  cv::Mat roi;
  roi = cv::Mat(1,1, CV_8UC1, cv::Scalar(0));
  
  if (pt(2) > .0) {
    
    Eigen::Matrix<int, 2, 4> corners;
    corners << get_window_corners(pt, height, width);
    cv::Point pt1(corners(1,0), corners(0,0)), pt2(corners(1,3), corners(0,3));

    
    Eigen::Matrix<int, 2, 1> tpcorner = project_to_img()
    
    if (pt1.x < 0) pt1.x = 0; if (pt1.y < 0) pt1.y = 0;
    if (pt1.x > depth_edges.cols) pt1.x = depth_edges.cols;
    if (pt1.y > depth_edges.rows) pt1.y = depth_edges.rows;

    if (pt2.x < 0) pt2.x = 0; if (pt2.y < 0) pt2.y = 0;
    if (pt2.x > depth_edges.cols) pt2.x = depth_edges.cols;
    if (pt2.y > depth_edges.rows) pt2.y = depth_edges.rows;

    roi = depth_edges(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x)).clone();

    // // removing noise by opening and closing the image
    // cv::morphologyEx(roi, roi, cv::MORPH_OPEN, struct_element,  cv::Point(-1,-1), 3);
    // cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, struct_element,  cv::Point(-1,-1), 3);

    
    // // threshold using otsu's method
    // double otsu_thresh = cv::threshold(roi, tmp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // cv::Canny(roi, roi, otsu_thresh*.5, otsu_thresh);
    // cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, struct_element);

  }

  return roi;
};


/**
 * moves the point (mvX,mvY,mvZ) so the distance between the returned point and the fixed point,
 * (fxX,fxY,fxZ) is equal to l.
 */
// std::tuple<double, double, double> InfoExtractor::move_keypoint(double mvX, double mvY, double mvZ,
// 								double fxX, double fxY, double fxZ,
// 								double l)
// {};


// [Person][Body Part][Point]->[row, col, X,Y,Z, score]
std_msgs::Float64MultiArray InfoExtractor::get_3d_keypoints(op::Array<float> keypoints)
{
  std_msgs::Float64MultiArray kp_arr;
  double row, col, X, Y, Z, score;

  // Getting message data ready =====================================================================
  
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
  kp_arr.layout.dim[2].size = keypoints.getSize(2) + 3; // row,col,score is already there. +3 for xyz
  kp_arr.layout.dim[2].stride = 1;

  kp_arr.data.clear();

  for (auto person = 0; person < keypoints.getSize(0); person++) {

    // collecting all data for this person so we can refine it in a separate step.
    // std::vector< std::tuple<double, double, double, double, double, double> > tmp_loc;
    std::vector< std::tuple<Eigen::Vector2d, Eigen::Vector3d, double> > tmp_loc;
    for (auto body_part = 0; body_part < keypoints.getSize(1); body_part++) {

      col = (double)keypoints[{person, body_part, 0}];
      row = (double)keypoints[{person, body_part, 1}];
      score = (double)keypoints[{person, body_part, 2}];

      std::tie(X, Y, Z) = project_3d((int)row, (int)col);

      // std::cout << "(" << X << ", " << Y << ", " << Z << ")\n";
      Eigen::Vector2d kp2d(row, col);
      Eigen::Vector3d kp3d(X, Y, Z);
      
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
    Eigen::Vector2d a_kp2d;
    Eigen::Vector3d a_kp3d;
    double a_score;
    
    std::tie(a_kp2d, a_kp3d, a_score) = tmp_loc.at(8);
    if (a_score == tmp_loc.at(8)[4])
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
}




    
// // choose which point to push back
//     // calculating limb scores to get the weighted sum for scale later

//     // Taking the weighted average scale 
//     double scale = .0, score_sum = .0;
//     std::vector<double> scores;
//     for (int i = 0; i < 10; i++) {
//       std::tuple<double, double, double, double, double, double> a, b;
//       a = tmp_loc.at(constrained_limb_pairs[i][0]);
//       b = tmp_loc.at(constrained_limb_pairs[i][1]);

//       // (figure out which of a or b should be moved, and calculate the limb score.)

//       // get the limb score and length.
//       double length = vector_abs(std::get<2>(a)-std::get<2>(b),
//     				 std::get<3>(a)-std::get<3>(b),
//     				 std::get<4>(a)-std::get<4>(b));
//       double lscore = limb_score(std::get<5>(a),std::get<5>(b));
      
//       scale += (length / normalized_constraints[i]) * lscore;
//       score_sum += lscore;
//       scores.push_back(lscore);
//     }

#endif // 

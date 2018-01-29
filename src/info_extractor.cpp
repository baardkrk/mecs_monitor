#ifndef INFO_EXTRACTOR_CPP
#define INFO_EXTRACTOR_CPP

#include "openpose_flags.cpp"
#include "info_extractor.h"

// TODO: fix the obvious security holes by initializing the camera_info pointer and the matrices.

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
  // frameDisplayer.displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
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

  // TBD: get RoIs
  
  // Packaging information to message =====================================
  
  e_info.header.stamp = ros::Time::now();
  e_info.header.frame_id = camera_info->header.frame_id;
  e_info.keypoints = keypoint_array;

  return e_info;
};

// Any depth that is nearer to the camera than 0.2m is invalid
std::tuple<double, double, double> InfoExtractor::project_3d(int row, int col)
{
  double Z = blurred_depth.at<double>(row, col), X, Y;
  double fx = camera_info->K[0],
    fy = camera_info->K[4],
    Cx = camera_info->K[2],
    Cy = camera_info->K[5];

  Y = ((double)row - Cy) * Z / fy;
  X = ((double)col - Cx) * Z / fx;

  return std::make_tuple(X, Y, Z);
};

// [Person][Body Part][Point]->[row, col, X,Y,Z, score]
std_msgs::Float64MultiArray InfoExtractor::get_3d_keypoints(op::Array<float> keypoints)
{
  std_msgs::Float64MultiArray kp_arr;
  double row, col, X, Y, Z, score;

  kp_arr.layout.data_offset = 0;

  kp_arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
  kp_arr.layout.dim[0].label = "Person";
  kp_arr.layout.dim[0].size = keypoints.getSize(0);
  kp_arr.layout.dim[0].stride = keypoints.getSize(1) * (keypoints.getSize(2) + 1);
  
  kp_arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
  kp_arr.layout.dim[0].label = "";
  kp_arr.layout.dim[0].size = keypoints.getSize(0);
  kp_arr.layout.dim[0].stride = keypoints.getSize(1) * (keypoints.getSize(2) + 1);
  
  kp_arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
  kp_arr.layout.dim[0].label = "";
  kp_arr.layout.dim[0].size = keypoints.getSize(0);
  kp_arr.layout.dim[0].stride = keypoints.getSize(1) * (keypoints.getSize(2) + 1);

  kp_arr.data.clear();

  for (auto person = 0; person < keypoints.getSize(0); person++) {
    for (auto body_part = 0; body_part < keypoints.getSize(1); body_part++) {

      col = (double)keypoints[{person, body_part, 0}];
      row = (double)keypoints[{person, body_part, 1}];
      score = (double)keypoints[{person, body_part, 2}];

      std::tie(X, Y, Z) = project_3d((int)row, (int)col);

      kp_arr.data.push_back(row);
      kp_arr.data.push_back(col);
      kp_arr.data.push_back(X);
      kp_arr.data.push_back(Y);
      kp_arr.data.push_back(Z);
      kp_arr.data.push_back(score);
    }
  }

  return kp_arr;
};



void InfoExtractor::test()
{
  cv::Mat color_ir;

  cv::cvtColor(ir, color_ir, cv::COLOR_GRAY2BGR);
  

  //std::cout << color_ir << "\n ====================================================================================== \n";

  // cv::Mat check_it;
  // the rgb image took the spotlight (oor, the net input size would be wrong...)
  // cv::hconcat(rgb, color_ir, check_it);
  run_openpose(color_ir, "ir_preview");
  // run_openpose(rgb, "rgb_preview");
  // gaussian_win(5);
};

#endif // INFO_EXTRACTOR_CPP

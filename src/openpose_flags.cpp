#ifndef OPENPOSE_FLAGS
#define OPENPOSE_FLAGS

#define USE_CAFFE

///////// caffe dependencies //////////
#include <gflags/gflags.h>

DEFINE_int32(logging_level, 3,
	     "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
	     " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
	     " low priority messages and 4 for important ones.");

// should possibly input base_name ('kred') as an argument.

DEFINE_string(camera_topic, "/kinect2/sd/image_ir_rect",
	      "Image topic we will subscribe to. This needs to be a RGB image. It should also be "
	      "registered to the depth image you want to combine it with.");

DEFINE_string(depth_topic, "/kinect2/sd/image_depth_rect",
	      "The depth image that is registered to the image in the camera topic");
	      
/////////// OpenPose Definitions ////////////
DEFINE_string(model_folder, "/home/baard/openpose/models/",
	      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");

DEFINE_string(model_pose, "COCO",
	      "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
	      "`MPI_4_layers` (15 keypoints, even faster but less accurate).");

DEFINE_string(net_resolution, "176x144",//"-1x368", //"656x368" // changed to sd(480x400)
	      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
	      " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
	      " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
	      " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");

// DEFINE_string(resolution, "-1x-1",//"1280x720",
// 	      "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
// 	      " default images resolution.");

DEFINE_string(output_resolution, "-1x-1",
	      "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	      " input image resolution.");

DEFINE_int32(num_gpu_start, 0, "GPU device start number.");

DEFINE_double(scale_gap, 0.3,
	      "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
	      " If you want to change the initial scale, you actually want to multiply the"
	      " `net_resolution` by your desired initial scale.");

DEFINE_int32(scale_number, 1, "Number of scales to average.");

// OpenPose Rendering
DEFINE_int32(part_to_show, 0,
	     "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
	     " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
	     " together, 21 for all the PAFs, 22-40 for each body part pair PAF");

DEFINE_bool(disable_blending, false,
	    "If blending is enabled, it will merge the results with the original frame. If disabled, it"
	    " will only display the results.");

DEFINE_double(render_threshold, 0.05,
	      "Only estimated keypoints whose score confidences are higher than this threshold will be"
	      " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	      " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	      " more false positives (i.e. wrong detections).");

DEFINE_double(alpha_pose, 0.6,
	      "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	      " hide it. Only valid for GPU rendering.");

DEFINE_double(alpha_heatmap, 0.7,
	      "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
	      " heatmap, 0 will only show the frame. Only valid for GPU rendering.");

#endif // OPENPOSE_FLAGS

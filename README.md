# MECS Monitor ROS package
This package is meant for extracting information about humans to be used for better human-robot interactions.
Another feature of the MECS project is to fuction as an automatic safety alarm for elderly people. The detected information is as follows:
- Human pose (and 3D location)
- Heart rate
- Respiration rate

The package is written in C++ for ROS Kinetic.

## Installation
Make sure you have a CUDA compatible graphics card (and driver), and install the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html "Installation instructions for CUDA").

Now, install [opencv](https://github.com/opencv/opencv "OpenCV's GitHub repo"). You can also install opencv from your distributions repository. If you build it from source, you can turn on CUDA compatibility to make full use of your graphics card for image processing.
This version of opencv will only be used when you install Caffe, and OpenPose. A different version of OpenCV is included in ROS Kinetic, and is what will be used by the package. (Unless ROS creates a symbolic link to the installed version if it is detected. I'm not sure.)

Now, install ROS if it is not installed. When this package was first written, the program was tested on Ubuntu 16.04 with ROS Kinetic.
I'm currently testing ROS Kinetic on Ubuntu 18.04, but it might not work.

Now, we need to install CMU's [openpose package](https://github.com/CMU-Perceptual-Computing-Lab/openpose). Install it in any location you want, but for the remainder of this guide, we will assume you installed it in
```
$HOME/openpose/
```
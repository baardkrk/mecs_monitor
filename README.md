# MECS Monitor ROS package
This package is meant for extracting information about humans using an RGB-D camera. This information is meant for aiding in better human-robot interactions.
Another feature of the MECS project is to fuction as an automatic safety alarm for elderly people, in case of accidents. The detected information is as follows:

- [x] Human pose (and 3D location)
- [x] Regions of interest (face, chest, hands, more can easily be added)
- [] Mood
- [] Heart rate
- [] Respiration rate

The package is written in C++ for ROS Kinetic and Microsoft's Kinect v2.

## Installation
Make sure you have the latest installation of gcc and g++ installed. (v8 at time of writing.) If you don't, here are some [instructions](https://askubuntu.com/questions/26498/choose-gcc-and-g-version "ubuntu answer for gcc").

Make sure you have a CUDA compatible graphics card (and driver), and install the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html "Installation instructions for CUDA").

Now, install [opencv](https://github.com/opencv/opencv "OpenCV's GitHub repo"). You can also install opencv from your distributions repository. If you build it from source, you can turn on CUDA compatibility to make full use of your graphics card for image processing.
This version of opencv will only be used when you install Caffe, and OpenPose. A different version of OpenCV is included in ROS Kinetic, and is what will be used by the package. (Unless ROS creates a symbolic link to the installed version if it is detected. I'm not sure.)

Now, [install ROS](http://wiki.ros.org/kinetic/Installation/ "ROS Kinetic installation pages") if it is not installed. When this package was first written, the program was tested on Ubuntu 16.04 with ROS Kinetic.
I'm currently testing ROS Kinetic on Ubuntu 18.04, but it might not work.

Install the [iai_kinect package](https://github.com/code-iai/iai_kinect2#install). (Also, it says it will not work with OpenCV 3, but it does.
*TODO: describe fix for opencv3*) The package currently only supports Microsoft's Kinect v2, and therefore needs this package. Support for other RGB-D cameras might be added later.

Now, we need to install CMU's [openpose package](https://github.com/CMU-Perceptual-Computing-Lab/openpose). Install it in any location you want, but for the remainder of this guide, we will assume you installed it in
```
$HOME/openpose/
```
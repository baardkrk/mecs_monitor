# MECS Monitor ROS package
This package is meant for extracting information about humans to be used for better human-robot interactions.
Another feature of the MECS project is to fuction as an automatic safety alarm for elderly people. The detected information is as follows:
- Human pose (and 3D location)
- Heart rate
- Respiration rate

The package is written in C++ for ROS Kinetic.

## Installation
First, we have to install the (CUDA)[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html Installation instructions for CUDA] toolkit.

Then, we need to install the openpose package. Install it in any location you want, but for the remainder of this guide, we will assume you installed it in
```
$HOME/openpose/
```
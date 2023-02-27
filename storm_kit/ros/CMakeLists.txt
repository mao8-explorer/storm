cmake_minimum_required(VERSION 3.5.2)
project(storm)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  tf
  vision_msgs
  visualization_msgs
)


catkin_package(
  CATKIN_DEPENDS geometry_msgs sensor_msgs std_msgs vision_msgs visualization_msgs
#  DEPENDS system_lib
)
#!/bin/bash

source /opt/ros_ws/devel/setup.bash
# rostopic pub -r 10 /stereo_matching_flg std_msgs/String "data: '1'" &
cd /opt/ros_ws/ && roslaunch zed_wrapper zed_camera_mod_serial.launch



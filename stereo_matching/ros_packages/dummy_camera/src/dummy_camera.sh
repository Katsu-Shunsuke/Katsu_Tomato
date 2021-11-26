#!/bin/bash

rosbag play -l /root/catkin_ws/src/dummy_camera/2021-05-20-09-24-08.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-14-01-01.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-14-28-08.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/20211105_112210_zedm.bag &
python /root/catkin_ws/src/dummy_camera/src/dummy_camera.py &
python /root/catkin_ws/src/dummy_camera/src/change_camera_name.py &
rosrun tf2_ros static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 zedA_left_camera_optical_frame zedm_left_camera_optical_frame



#!/bin/bash

rosbag play -l /root/catkin_ws/src/dummy_camera/2021-05-20-09-24-08.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-14-01-01.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-14-28-08.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/20211105_112210_zedm.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-13-58-16.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-14-11-02.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-14-27-29.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-14-21-23.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-13-56-32.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-13-56-45.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/2021-11-03-13-56-57.bag &
# rosbag play -l /root/catkin_ws/src/dummy_camera/20211210_153010_zedm.bag &
python /root/catkin_ws/src/dummy_camera/src/dummy_camera.py &
python /root/catkin_ws/src/dummy_camera/src/change_camera_name.py &
rosrun tf2_ros static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 zedA_left_camera_optical_frame zedm_left_camera_optical_frame



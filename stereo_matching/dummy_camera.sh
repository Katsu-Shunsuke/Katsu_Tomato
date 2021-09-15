#!/bin/bash

rosbag play -l /root/catkin_ws/src/dummy_camera/2021-05-20-09-24-08.bag &
python /root/catkin_ws/src/dummy_camera/src/dummy_camera.py


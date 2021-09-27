# cuda-11.0 requires nvidia-driver>450 but its 430.something on host so cant use it
# FROM nvidia/cuda:11.0-devel-ubuntu20.04

FROM osrf/ros:noetic-desktop-full

# I think default is python3.8 which should be ok
RUN apt update && apt install -y python3-pip tmux vim
RUN ln -s python3.8 /usr/bin/python

#catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws && \
    catkin_make

RUN cd ~/catkin_ws/src/ && catkin_create_pkg synthesis std_msgs sensor_msgs rospy cv_bridge

COPY synthesis_node.py /root/catkin_ws/src/synthesis/src/synthesis_node.py
COPY utils.py /root/catkin_ws/src/synthesis/src/utils.py

# RUN mkdir /root/catkin_ws/src/instance_segmentation/msg
# COPY InstSegRes.msg /root/catkin_ws/src/instance_segmentation/msg/InstSegRes.msg
COPY keep_alive.py /root/keep_alive.py






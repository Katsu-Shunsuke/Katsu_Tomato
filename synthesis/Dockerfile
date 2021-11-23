# cuda-11.0 requires nvidia-driver>450 but its 430.something on host so cant use it
# FROM nvidia/cuda:11.0-devel-ubuntu20.04

FROM osrf/ros:noetic-desktop-full

# I think default is python3.8 which should be ok
RUN apt update && apt install -y python3-pip tmux vim
RUN ln -s python3.8 /usr/bin/python

RUN pip install scikit-learn

#catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws && \
    catkin_make

# RUN mkdir /root/catkin_ws/src/instance_segmentation/msg
# COPY InstSegRes.msg /root/catkin_ws/src/instance_segmentation/msg/InstSegRes.msg
COPY keep_alive.py /root/keep_alive.py

COPY ros_packages/synthesis /root/catkin_ws/src/synthesis

RUN . /opt/ros/noetic/setup.sh && cd ~/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release






# cuda-11.0 requires nvidia-driver>450 but its 430.something on host so cant use it
# FROM nvidia/cuda:11.0-devel-ubuntu20.04

FROM osrf/ros:noetic-desktop-full

COPY cuda-10.1/ /usr/local/cuda-10.1/
RUN ln -s cuda-10.1 /usr/local/cuda

# I think default is python3.8 which should be ok
RUN apt update && apt install -y python3-pip && apt install -y git
RUN pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
RUN pip install torch==1.7.0 torchvision==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN cd && git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && \
    pip install -r requirements/build.txt && \
    pip install -v -e .
RUN ln -s python3.8 /usr/bin/python

RUN apt install -y tmux vim

# cuda path
ENV PATH /usr/local/cuda-10.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda-10.1/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws && \
    catkin_make

RUN cd ~/catkin_ws/src/ && catkin_create_pkg instance_segmentation std_msgs sensor_msgs rospy cv_bridge

COPY epoch_400.pth /root/catkin_ws/src/instance_segmentation/epoch_400.pth
COPY cascade_mask_rcnn_r50_fpn_1x_tomato.py /root/catkin_ws/src/instance_segmentation/cascade_mask_rcnn_r50_fpn_1x_tomato.py
COPY instance_segmentation_node.py /root/catkin_ws/src/instance_segmentation/src/instance_segmentation_node.py
COPY utils.py /root/catkin_ws/src/instance_segmentation/src/utils.py

RUN mkdir /root/catkin_ws/src/instance_segmentation/msg
COPY InstSegRes.msg /root/catkin_ws/src/instance_segmentation/msg/InstSegRes.msg
COPY keep_alive.py /root/keep_alive.py






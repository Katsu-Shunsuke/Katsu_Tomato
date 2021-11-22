FROM stereolabs/zed:3.4-ros-devel-cuda11.1-ubuntu18.04

# cuda path
# ENV PATH /usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

COPY launch_files/zed_camera_mod_serial.launch /opt/ros_ws/src/zed-ros-wrapper/zed_wrapper/launch/zed_camera_mod_serial.launch
COPY launch_files/zed_multi_cam3_serial.launch /opt/ros_ws/src/zed-ros-wrapper/zed_wrapper/launch/zed_multi_cam3_serial.launch
COPY keep_alive.py /root/keep_alive.py

RUN . /opt/ros/melodic/setup.sh && cd /opt/ros_ws && catkin_make -DCMAKE_BUILD_TYPE=Release


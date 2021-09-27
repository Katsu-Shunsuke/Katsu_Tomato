FROM osrf/ros:noetic-desktop-full

# probably best not to just copy cuda
# wrong to do "ADD cuda-8.0.tar.gz /usr/local/" because then gets placed in /usr/local/usr/local/cuda-8.0
# ADD cuda-8.0.tar.gz /
# ADD cuda-9.0.tar.gz /

COPY cuda-10.0/ /usr/local/cuda-10.0/
RUN ln -s cuda-10.0 /usr/local/cuda
# honestly better to mount storage wise but need cuda when doing docker build.
# volume (-v) is mounted during docker run which comes after build.

# PyTorch installation
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.7 && \
    apt install -y python3-pip
RUN ln -s python3.7 /usr/bin/python
RUN python -m pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install numpy -I

# stuffs
RUN apt install -y tmux; apt install -y vim
# COPY aanet/ /root/aanet/
RUN python -m pip install scikit-image
RUN rm -r /usr/lib/python3/dist-packages/PIL/
RUN rm -r /usr/lib/python3/dist-packages/Pillow*    
RUN python -m pip install Pillow
RUN apt update && apt install -y libpython3.7-dev
RUN python -m pip install opencv-python

# gcc-7 and g++-7 installation and softlink reconfiguration
RUN apt install -y gcc-7 ; apt install -y g++-7
RUN rm /usr/bin/gcc && ln -s gcc-7 /usr/bin/gcc ; \
    rm /usr/bin/gcc-ar && ln -s gcc-ar-7 /usr/bin/gcc-ar ; \
    rm /usr/bin/gcc-nm && ln -s gcc-nm-7 /usr/bin/gcc-nm ; \
    rm /usr/bin/gcc-ranlib && ln -s gcc-ranlib-7 /usr/bin/gcc-ranlib ; \
    rm /usr/bin/x86_64-linux-gnu-gcc && ln -s gcc-7 /usr/bin/x86_64-linux-gnu-gcc ; \
    rm /usr/bin/x86_64-linux-gnu-gcc-ar && ln -s gcc-ar-7 /usr/bin/x86_64-linux-gnu-gcc-ar ; \
    rm /usr/bin/x86_64-linux-gnu-gcc-nm && ln -s gcc-nm-7 /usr/bin/x86_64-linux-gnu-gcc-nm ; \
    rm /usr/bin/x86_64-linux-gnu-gcc-ranlib && ln -s gcc-ranlib-7 /usr/bin/x86_64-linux-gnu-gcc-ranlib
RUN rm /usr/bin/g++ && ln -s g++-7 /usr/bin/g++ ; \
    rm /usr/bin/x86_64-linux-gnu-g++ && ln -s g++-7 /usr/bin/x86_64-linux-gnu-g++

# set environment variables for cuda
# RUN cd ~/aanet/nets/deform_conv && bash build.sh
ENV PATH /usr/local/cuda-10.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# Now "bash scripts/aanet_predict.sh" will work.

# catkin_ws
# Note that each RUN command is executed inside a subshell so need to source and run catkin_make together. OR add to ~/.bashrc
# Be careful, /bin/sh is used by docker instead of bash so need to source setup.sh not setup.bash
# also "source" is a bash command, so cannot use with sh. Instead, use "."
RUN . /opt/ros/noetic/setup.sh && \
    mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws && \
    catkin_make

# # copy stereomatching package from host to container and move aanet directory to package
# # Be careful, COPY destination needs to be absolute path, not relative using like ~
# RUN cd ~/catkin_ws/src/ && catkin_create_pkg stereo_matching std_msgs sensor_msgs rospy cv_bridge
# RUN cd ~/catkin_ws/src/ && catkin_create_pkg dummy_camera std_msgs rospy
# COPY stereo_matching_node.py /root/catkin_ws/src/stereo_matching/src/stereo_matching_node.py
# COPY utils.py /root/catkin_ws/src/stereo_matching/src/utils.py
# COPY verify_array.py /root/catkin_ws/src/stereo_matching/src/verify_array.py
# COPY 2021-05-20-09-24-08.bag /root/catkin_ws/src/dummy_camera/2021-05-20-09-24-08.bag
# COPY dummy_camera_node.py /root/catkin_ws/src/dummy_camera/src/dummy_camera_node.py
# COPY dummy_camera.sh /root/catkin_ws/src/dummy_camera/src/dummy_camera.sh
# RUN mv ~/aanet/ ~/catkin_ws/src/stereo_matching/src/
# 
# # need to fix the issue of rosrun not being able to find python script- cmakelist
# RUN . /opt/ros/noetic/setup.sh && \
#     cd ~/catkin_ws && catkin_make
# # run roscore inside a different container




COPY keep_alive.py /root/keep_alive.py

COPY ros_packages/dummy_camera /root/catkin_ws/src/dummy_camera
COPY ros_packages/stereo_matching /root/catkin_ws/src/stereo_matching
RUN . /opt/ros/noetic/setup.sh && cd ~/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release

RUN cd ~/catkin_ws/src/stereo_matching/src/aanet/nets/deform_conv && bash build.sh





















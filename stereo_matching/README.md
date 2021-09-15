## Dockerfile
This image is for running the AANet ROS node.

Container specs are the following:
- Ubuntu 20.04
- ROS Noetic
- Python3.7
- CUDA 10.0
- PyTorch 1.2.0

The following are the requirements:
- nvidia-docker 2.0.3
- docker 18.09.2
- nvidia-driver 430.64

Directory tree is the following:  
```bash
/  
|-root  
  |-catkin_ws  
    |-dev  
    |-build  
    |-src  
      |-CMakeLists.txt 
      |-dummy_camera
        |-CMakeLists.txt
        |-package.xml
        |-src
          |-dummy_camera.py 
      |-stereo_matching
        |-CMakeLists.txt
        |-package.xml
        |-src  
          |-stereo_matching.py  
          |-aanet  
            |-predict_func.py  
            |- ...  
            |-__init__.py  
```

Basically, stereo_matching.py is the ROS node which will subscribe to images coming from the stereo camera (with the best view) and will output the disparity map.  
This will script will import predict_func and make use of functions for loading AANet and making predictions using it.



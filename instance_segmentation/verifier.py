#!/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import numpy as np

from utils import float32_to_numpy, numpy_to_float32
from instance_segmentation.msg import InstSegRes

def callback(msg):
    print("poo")
    float32 = msg.bbox_stem # tuple of length h x w
    array = np.array(float32.data)
    h = float32.layout.dim[0].size
    w = float32.layout.dim[1].size
    array = array.reshape([h, w])
    print(array.shape)

rospy.init_node("verifier")
rospy.Subscriber("/instance_segmentation_output", InstSegRes, callback)
rospy.spin()


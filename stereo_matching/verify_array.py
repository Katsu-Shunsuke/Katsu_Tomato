#!/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import numpy as np

def callback(msg):
    array = msg.data # tuple of length h x w
    array = np.array(array)
    h = msg.layout.dim[0].size
    w = msg.layout.dim[1].size
    array = array.reshape([h, w])
    print(array.shape)

rospy.init_node("array_verifier")
rospy.Subscriber("/aanet_depth_output", Float32MultiArray, callback)
rospy.spin()

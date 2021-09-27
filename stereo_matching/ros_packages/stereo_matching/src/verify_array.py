#!/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

def callback(msg):
    array = msg.data
    print(type(array))
    print(len(array))
    print(msg.layout.dim[0].stride)

rospy.init_node("array_verifier")
rospy.Subscriber("/aanet_depth_output", Float32MultiArray, callback)
rospy.spin()

import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import numpy as np

# def numpy_to_float32(array):
#     msg = Float32MultiArray()
#     msg.data = array.flatten()
#     msg.layout.data_offset = 0
#     msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
#     msg.layout.dim[0].label = "height"
#     msg.layout.dim[0].size = array.shape[0]
#     msg.layout.dim[0].stride = array.shape[0] * array.shape[1]
#     msg.layout.dim[1].label = "width"
#     msg.layout.dim[1].size = array.shape[1]
#     msg.layout.dim[0].stride = array.shape[1]
#     return msg
# 
# def float32_to_numpy(msg):
#     array = np.array(msg.data) # tuple of length h x w
#     h = msg.layout.dim[0].size
#     w = msg.layout.dim[1].size
#     array = array.reshape([h, w])
#     return array

def numpy_to_float32(array):
    msg = Float32MultiArray()
    msg.data = array.flatten().astype(np.float32)
    msg.layout.data_offset = 0
    msg.layout.dim = []
    for i, size in enumerate(array.shape):
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[i].label = str(i)
        msg.layout.dim[i].size = size
        msg.layout.dim[i].stride = prod(array.shape[(i - len(array.shape)):])
    return msg
        
def float32_to_numpy(msg):
    array = np.array(msg.data) # tuple of length h x w
    shape = [i.size for i in msg.layout.dim]
    array = array.reshape(shape)
    return array

def prod(list_or_tuple):
    res = 1
    for i in list_or_tuple:
        res *= i
    return res

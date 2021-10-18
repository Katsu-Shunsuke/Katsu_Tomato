import rospy
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, MultiArrayDimension

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

def numpy_to_rosarray(array, dtype):
    if dtype == "float32":
        msg = Float32MultiArray()
        datatype = np.float32
    elif dtype == "uint8":
        msg = UInt8MultiArray()
        datatype = np.uint8
    else:
        raise Exception("unrecognized dtype")
    # MultiArrayDimension bit
    msg.data = tuple(array.flatten().astype(datatype))
    msg.layout.data_offset = 0
    msg.layout.dim = []
    for i, size in enumerate(array.shape):
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[i].label = str(i)
        msg.layout.dim[i].size = size
        msg.layout.dim[i].stride = prod(array.shape[(i - len(array.shape)):])
    return msg
        
def rosarray_to_numpy(msg):
    array = np.array(msg.data) # tuple of length h x w
    shape = [i.size for i in msg.layout.dim]
    array = array.reshape(shape)
    return array

def prod(list_or_tuple):
    res = 1
    for i in list_or_tuple:
        res *= i
    return res

def stereo_reconstruction(d, b=63, f=2.8):
    # b [mm]
    # f [mm]
   
    r_x = d.shape[1] # number of pixels in horizontal direction
    r_y = d.shape[0] # number of pixels in vertical direction
    p_x = 0.002 * 1920 / r_x # pixel size in mm, horizontally. Need to readjust for resized image (zed camera at 1920x1080 is 0.002 mm per pixel)
    p_y = 0.002 * 1080 / r_y # pixel size in mm, vertically.
    d[d==0] = 0.0000000001 # to prevent dividing by zero
   
    z = b * (f/p_x) / d # d is in pixels, so z is in mm
   
    xv, yv = np.meshgrid(np.arange(r_x), np.arange(r_y))
    x = (xv - r_x/2) * p_x * z / f
    y = (yv - r_y/2) * p_y * z / f
   
    xyz = np.dstack([x, y, z])
   
    return xyz

def polynomial_derivative(coefs):
    deg = len(coefs) - 1
    deriv_coefs = []
    for i, poly_coef in enumerate(coefs[:-1]):
        deriv_coefs.append(poly_coef * (deg - i))
    return deriv_coefs

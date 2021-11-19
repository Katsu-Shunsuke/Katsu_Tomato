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

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    credit to: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

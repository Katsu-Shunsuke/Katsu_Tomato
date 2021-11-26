import rospy
from std_msgs.msg import Header, Float32MultiArray, UInt8MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import struct
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

def stereo_reconstruction(d, b=63):
    # camera parameters and info given below:
    # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
    # b [mm]
    # f [mm]
   
    r_x = d.shape[1] # number of pixels in horizontal direction
    r_y = d.shape[0] # number of pixels in vertical direction
    if r_x == 1920 and r_y == 1080:
        f = 1400 # focal length in pixels
        p = 0.002 # pixel size in mm
    elif r_x == 1280 and r_y == 720:
        f = 700 # focal length in pixels
        p = 0.004 # pixel size in mm
    elif r_x == 672 and r_y == 376:
        f = 350 # focal length in pixels
        p = 0.008 # pixel size in mm
    else:
        raise Exception("Unsupported resolution.")

    f = f * p # focal length in mm

    # probably wrong to take pixel size and resolution as linear...
#    p_x = 0.002 * 1920 / r_x # pixel size in mm, horizontally. Need to readjust for resized image (zed camera at 1920x1080 is 0.002 mm per pixel)
#    p_y = 0.002 * 1080 / r_y # pixel size in mm, vertically.
    d[d==0] = 0.0000000001 # to prevent dividing by zero
   
    z = b * (f / p) / d # d is in pixels, so z is in mm
   
    xv, yv = np.meshgrid(np.arange(r_x), np.arange(r_y))
    x = (xv - r_x/2) * p * z / f
    y = (yv - r_y/2) * p * z / f
   
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

# def generate_pc2_message(xyz, rgb):
#     header = Header(frame_id="/zedA_left_camera_optical_frame")
#     fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#               PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#               PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
#               PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
#  
#     assert(xyz.shape == rgb.shape)
#     xyz = xyz.reshape((-1, 3))
#     rgb = rgb.reshape((-1, 3))
#     print(xyz.shape)
#     print(rgb.shape)
# 
#     points = [[0.3, 0.0, 0.0, 0xff0000],
#               [0.0, 0.3, 0.0, 0x00ff00],
#               [0.0, 0.0, 0.3, 0x0000ff]]
#  
#     point_cloud = pc2.create_cloud(header, fields, points)
#  
#     return point_cloud



# def generate_pc2_message(xyz, rgb):
#     header = Header(frame_id="/zedA_left_camera_optical_frame")
#     fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#               PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#               PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
#               PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
#  
#     assert(xyz.shape == rgb.shape)
#     xyz = xyz.reshape((-1, 3))
#     rgb = rgb.reshape((-1, 3))
# 
#     hex_color = []
#     for i in rgb:
#         r, g, b = i
#         a = 255
#         rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
#         hex_color.append(rgb)
#         print(rgb)
# 
#     hex_color = np.array(hex_color).reshape((-1, 1))
#     points = list(np.hstack((xyz, hex_color)))
#     print(points)
#  
#     points = [[0.3, 0.0, 0.0, 0xff0000],
#               [0.0, 0.3, 0.0, 0x00ff00],
#               [0.0, 0.0, 0.3, 0x0000ff]]
#  
#     point_cloud = pc2.create_cloud(header, fields, points)
#  
#     return point_cloud


def generate_pc2_message(xyz, rgb):
    header = Header(frame_id="/zedm_left_camera_optical_frame")
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
              PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
 
    assert(xyz.shape == rgb.shape)
    xyz = xyz.reshape((-1, 3))
    rgb = rgb.reshape((-1, 3))
    size = round(0.1 * xyz.shape[0])
    idx = np.random.randint(xyz.shape[0], size=size)
    xyz = xyz[idx, :] * 10**(-3) # mm to m
    rgb = rgb[idx, :]

    points = xyz.tolist()
    for i, ting in enumerate(rgb):
        r, g, b = ting
        color = '0x%02x%02x%02x' % (r, g, b)
        color = int(color, 16)
        points[i].append(color)
 
    point_cloud = pc2.create_cloud(header, fields, points)
 
    return point_cloud

# def generate_pc2_message(xyz, rgb):
#     header = Header(frame_id="/zedA_left_camera_optical_frame")
#     fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#               PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#               PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
#               PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
#  
#     assert(xyz.shape == rgb.shape)
#     xyz = xyz.reshape((-1, 3))
#     rgb = rgb.reshape((-1, 3))
#     hex_color = np.array([0xff0000] * xyz.shape[0]).reshape((-1,1))
# 
#     points = list(np.hstack((xyz, hex_color)))
#     print(points)
#  
#     point_cloud = pc2.create_cloud(header, fields, points)
#  
#     return point_cloud

def Color(red, green, blue, white = 0):
    """Convert the provided red, green, blue color to a 24-bit color value.
    Each color component should be a value 0-255 where 0 is the lowest intensity
    and 255 is the highest intensity.
    """
    return (white << 24) | (red << 16)| (green << 8) | blue







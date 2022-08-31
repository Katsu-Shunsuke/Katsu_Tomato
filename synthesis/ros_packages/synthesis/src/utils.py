import rospy
from std_msgs.msg import Header, Float32MultiArray, UInt8MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import struct
import numpy as np

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

def generate_pc2_message(xyz, rgb, sampling_prop=0.1):
    header = Header(frame_id="zedm_left_camera_optical_frame")
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
              PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
 
    assert(xyz.shape == rgb.shape)
    xyz = xyz.reshape((-1, 3))
    rgb = rgb.reshape((-1, 3))
    size = round(sampling_prop * xyz.shape[0])
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

def filter_instseg(bbox, mask, threshold):
    """
    INPUTS
    bbox: (n_instances, 5), indices 0-3 are for bbox, index 4 is confidence score
    mask: (n_points, 3), first 2 columns are i and j, third column specifies which instance
    threshold: float
    OUTPUTS
    bbox: (n_instances_filtered, 5)
    mask: list of (n_points, 2), list of length=n_instances_filtered
    """
    # convert mask to list form
    n_instances = np.max(mask[:,2]).astype(int) + 1 if mask.size else 0 # it is possible that no pedicels are detected
    mask_list = [mask[mask[:,2]==i][:, :2] for i in range(n_instances)] #i since self.mask_pedicel starts at zero
    
    assert(n_instances == len(bbox))
    if n_instances > 0:
        bbox_out = []
        mask_out = []
        for i, instance in enumerate(bbox):
            if instance[4] > threshold:
                bbox_out.append(instance)
                mask_out.append(mask_list[i])
        return np.vstack(bbox_out), mask_out
    else:
        return bbox, mask_list





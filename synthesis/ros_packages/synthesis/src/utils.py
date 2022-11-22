import rospy
from std_msgs.msg import Header, Float32MultiArray, UInt8MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import io
import cv2
import struct
import numpy as np

from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.decomposition import PCA

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
        if bbox_out: # possible that n_instance!=0 but threshold not met
            return np.vstack(bbox_out), mask_out
        else:
            return bbox, mask_list
    else:
        return bbox, mask_list

# based on the one in instance segmentation node with very slight mods
def visualize_output(image, result, threshold_per_class=[0.2, 0.8, 0.4, 0.7], show_bbox=True, save_im=False):
    # image should be numpy array (bgr)
    # result should be the output of inference_detector
    fig = plt.figure(figsize=(25,15))
    ax = fig.add_subplot(111)

    ax.imshow(image)
    # bbox
    c = ["r", "c", "m", "y"]
    classes = ["stem", "tomato", "pedicel", "sepal"]

    for i, bbox_per_class in enumerate(result[0]):
        for j, bbox in enumerate(bbox_per_class):
            if bbox[4] >= threshold_per_class[i]:
                # so bbox array is: [x_top_left, y_top_left, x_bottom_right, y_bottom_right, confidence_score] which is different from coco
                if show_bbox:
                    ax.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]],
                            [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]], c[i], linewidth=4.0)
                # plot segmenation mask
                indices = result[1][i][j]
                ax.scatter(indices[:,1], indices[:,0], c=c[i], s=5, alpha=0.07, marker=".")

    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    img = get_img_from_fig(fig, dpi=60)

    if save_im:
        ax.savefig("test_im_" + str(datetime.datetime.now()) + ".png")

    return img

def get_img_from_fig(fig, dpi=80):
    """
    credit to:
    https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def curve_fitting(x, y, z, mode="polynomial", tomato_center=None, tomato_r=None):
    """
    INPUTS
    x: (n_points,)
    y: (n_points,)
    z: (n_points,)
    mode: str
    OUTPUTS
    cutpoint:
    dir_vector: 
    pedicel_end:
    fitted_curve:
    """
    pedicel_cut_prop = rospy.get_param("pedicel_cut_prop", 1)
    
    if mode == "polynomial":
        if tomato_center is None or tomato_r is None:
            raise Exception("Must provide tomato_center and/or tomato_r")
        deg = rospy.get_param("deg", 2)
        # polynomial regeression
        coefs_yx = np.polyfit(y, x, deg=deg)
        coefs_yz = np.polyfit(y, z, deg=deg)

        # calculate cumulative length of polynomial and y_cut
        y_grid = np.linspace(np.min(y), np.max(y), 100)
        x_grid = np.polyval(coefs_yx, y_grid)
        z_grid = np.polyval(coefs_yz, y_grid)
        curve_y_sorted = np.vstack((x_grid, y_grid, z_grid)).T
        dists = np.linalg.norm(curve_y_sorted[1:, :] - curve_y_sorted[:-1, :], axis=1)
        cumlen = np.cumsum(dists)
        y_cut = y_grid[np.argmin(np.abs(pedicel_cut_prop * cumlen[-1] - cumlen))] # len(y_grid) and len(cumlen) differ by 1 but doesnt matter

        # need to think about coordinate system
#        index = int((pedicel_cut_prop) * len(y))
#        y_cut = np.partition(y, index)[index]
#        y_cut = pedicel_cut_prop * (np.max(y) - np.min(y)) + np.min(y)

        # predict
        x_pred = np.polyval(coefs_yx, y_cut)
        z_pred = np.polyval(coefs_yz, y_cut)
        # pedicel xyz position
        cut_point = np.array([x_pred, y_cut, z_pred])
        # tangent vector (3D)
        deriv_coefs_yx = polynomial_derivative(coefs_yx)
        deriv_coefs_yz = polynomial_derivative(coefs_yz)
        deriv_yx = np.polyval(deriv_coefs_yx, y_cut)
        deriv_yz = np.polyval(deriv_coefs_yz, y_cut)
        dir_vector = np.array([deriv_yx, 1, deriv_yz])
        dir_vector /= np.linalg.norm(dir_vector) # unit vector
    
        # pedicel_end
        pedicel_end_y =  np.max(y)
        pedicel_end_x = np.polyval(coefs_yx, pedicel_end_y)
        pedicel_end_z = np.polyval(coefs_yz, pedicel_end_y)
#        pedicel_end_z = tomato_center[2] - np.sqrt(tomato_r**2 - (pedicel_end_x - tomato_center[0])**2 - (pedicel_end_y - tomato_center[1])**2)
        pedicel_end = np.array([pedicel_end_x, pedicel_end_y, pedicel_end_z])

        # fitted curve for visualization
        x_curve = np.polyval(coefs_yx, y)
        z_curve = np.polyval(coefs_yz, y)
        curve = np.vstack((x_curve, y, z_curve)).T
    elif mode == "spline":
        # WIP
        raise Exception("Don't use this mode, still a work in progress.")
        n = 1000
        s = round(len(x) / 10)
        k = 5
        tck, u = interpolate.splprep([x, y, z], s=s, k=k)
        u_fine = np.linspace(0, 1, n)
        xyz_curve = interpolate.splev(u, tck)
        curve = np.vstack(xyz_curve).T
    elif mode == "PCA":
        pass
    else:
        raise Exception("Unrecognized mode.")

    return cut_point, dir_vector, pedicel_end, curve 

def pca(xyz):
    pca = PCA(n_components=3)
    pca.fit(xyz)
    M = pca.components_
    return M

def calc_mean_point(xyz):
    p_mean = np.mean(xyz, axis=0)
    i_mean = np.argmin(
        np.sum(
            (xyz - np.tile(p_mean, (len(xyz), 1))) ** 2,
            axis=1
        )
    )
    xyz_mean = xyz[i_mean, :]
    return xyz_mean

def visualize_eigen_vectors(p, M):
    p1 = np.vstack([p + c * M[0, :] for c in range(40)]) # largest eigen value
    p2 = np.vstack([p + c * M[1, :] for c in range(17)])
    p3 = np.vstack([p + c * M[2, :] for c in range(8)]) # smallest eigen value
    eigen_pc = np.vstack((p, p1, p2, p3))
    return eigen_pc

def indices_within_circle(im_shape, c, r_max):
    j, i = np.meshgrid(np.arange(im_shape[1]), np.arange(im_shape[0]))
    r = np.sqrt((i - c[0])**2 + (j - c[1])**2)
    within_circle = r < r_max # boolean mask
    return within_circle






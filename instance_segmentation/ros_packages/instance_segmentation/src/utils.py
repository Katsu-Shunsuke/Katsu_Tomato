import rospy
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, MultiArrayDimension

import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def get_img_from_fig(fig, dpi=180):
    """
    credit to:
    https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def visualize_output(image, result, threshold_per_class=[0.2, 0.8, 0.4, 0.7], show_bbox=True, save_im=False):
    # image should be numpy array (bgr)
    # result should be the output of inference_detector
    fig = plt.figure(figsize=(25,15))
    ax = fig.add_subplot(111)

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # bbox
    c = ["r", "c", "m", "y"]
    classes = ["stem", "tomato", "pedicel", "sepal"]
   
    for i, bbox_per_class in enumerate(result[0]):
        for j, bbox in enumerate(bbox_per_class):
            if bbox[4] >= threshold_per_class[i]:
                # so bbox array is: [x_top_left, y_top_left, x_bottom_right, y_bottom_right, confidence_score] which is different from coco
                if show_bbox:
                    ax.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]],
                            [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]], c[i])
                # plot segmenation mask
                indices = np.nonzero(result[1][i][j])
                ax.scatter(indices[1], indices[0], c=c[i], s=5, alpha=0.07, marker=".")
#     plt.xlim([0,1240])
#     plt.ylim([720,0])

    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    img = get_img_from_fig(fig, dpi=60)

    if save_im:
        ax.savefig("test_im_" + str(datetime.datetime.now()) + ".png")

    return img


#!/usr/bin/env python

import os
import sys
import rospy

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension, Header
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import mmcv
from mmdet.apis import init_detector, inference_detector

from utils import numpy_to_rosarray, visualize_output
from instance_segmentation.msg import InstSegRes, ExitCode # need to edit CMakeLists.txt and package.xml
class InstanceSegmentation:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedm/zed_node/left/image_rect_color" # left image because disparity map is on left image.
#        self.flg_topic = "instance_segmentation_flg"
        self.flg_topic = "stereo_matching_flg"
        self.result_arr_topic = "instance_segmentation_array_output"
        self.result_im_topic = "instance_segmentation_image_output"
        self.config_file = '../cascade_mask_rcnn_r50_fpn_1x_tomato.py'
        self.checkpoint_file = '../epoch_2000.pth'
        # output of callback methods
        self.im = None
        self.im_array = None
        self.result = None
        self.maskrcnn = None
        self.flg = None
        self.result_arr_msg = None
        self.result_im_msg = None
        self.exit_code_pub = rospy.Publisher("large_tomato/exit_code", ExitCode, queue_size=1)

    def im_callback(self, msg):
        self.im = msg      
        cv_image = CvBridge().imgmsg_to_cv2(self.im, "bgr8")
#        im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#        self.im_array = np.array(im_rgb)
        self.im_array = np.array(cv_image) #mmdetection model was trined in bgr not rgb!!!!
    
    def main_callback(self):
        if self.maskrcnn is None:
            self.maskrcnn = init_detector(self.config_file, self.checkpoint_file, device='cuda:0') # change device acordingly
        print("running maskrcnn")
        self.result = inference_detector(self.maskrcnn, self.im_array) # list of list of array
        self.result_arr_msg = self.to_InstSegRes(self.result)

#            result_im = visualize_output(self.im_array, self.result, threshold_per_class=[0.2, 0.8, 0.4, 0.7], show_bbox=True, save_im=False)
        result_im = visualize_output(self.im_array, self.result, threshold_per_class=[0.0, 0.0, 0.0, 0.0], show_bbox=True, save_im=False)
        self.result_im_msg = CvBridge().cv2_to_imgmsg(result_im, "rgb8")
        print(result_im.shape)

    def update_flg(self, msg):
        if msg.data == "1" and self.im_array is not None:
            self.flg = "1"

    def to_InstSegRes(self, result):
        msg = InstSegRes()
        # bbox
        msg.bbox_stem = numpy_to_rosarray(self.result[0][0], "float32")
        msg.bbox_tomato = numpy_to_rosarray(self.result[0][1], "float32")
        msg.bbox_pedicel = numpy_to_rosarray(self.result[0][2], "float32")
        msg.bbox_sepal = numpy_to_rosarray(self.result[0][3], "float32")
        # mask - need to consider the case when no instance is detected and hence cannot concatenate
        i_stem  = np.transpose(np.nonzero(np.dstack(self.result[1][0]) if self.result[1][0] else []))
        i_tomato = np.transpose(np.nonzero(np.dstack(self.result[1][1]) if self.result[1][1] else []))
        i_pedicel = np.transpose(np.nonzero(np.dstack(self.result[1][2]) if self.result[1][2] else []))
        i_sepal = np.transpose(np.nonzero(np.dstack(self.result[1][3]) if self.result[1][3] else []))
        msg.mask_stem = numpy_to_rosarray(i_stem, "float32")
        msg.mask_tomato = numpy_to_rosarray(i_tomato, "float32")
        msg.mask_pedicel = numpy_to_rosarray(i_pedicel, "float32")
        msg.mask_sepal = numpy_to_rosarray(i_sepal, "float32")
#        msg.mask_stem = numpy_to_rosarray(np.dstack(self.result[1][0]), "uint8")
#        msg.mask_tomato = numpy_to_rosarray(np.dstack(self.result[1][1]), "uint8")
#        msg.mask_pedicel = numpy_to_rosarray(np.dstack(self.result[1][2]), "uint8")
#        msg.mask_sepal = numpy_to_rosarray(np.dstack(self.result[1][3]), "uint8")
        # header
        h = Header()
        h.stamp = rospy.Time.now()
        h.seq = 0
        h.frame_id = "poo"
        msg.header = h
        return msg
    
def main():
    rospy.init_node("instance_segmentation", anonymous=True)
    model = InstanceSegmentation()
    rospy.Subscriber(model.im_topic, Image, model.im_callback)
    rospy.Subscriber(model.flg_topic, String, model.update_flg)
    pub_arr = rospy.Publisher(model.result_arr_topic, InstSegRes, queue_size=1)
    pub_im = rospy.Publisher(model.result_im_topic, Image, queue_size=1)
    exit_code = ExitCode()
#    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if model.flg == "1":
            model.main_callback()
            if model.result_arr_msg is not None and model.result_im_msg is not None:
    #            rospy.loginfo(model.result_msg)
                pub_arr.publish(model.result_arr_msg)
                pub_im.publish(model.result_im_msg)
    #            r.sleep()
            else:
                rospy.loginfo("Instance segmentation is failed.")
                exit_code.exit_code = ExitCode.CODE_PEDICLE_INSTSEG_FAILED
                model.exit_code_pub.publish(exit_code)
            model.flg = "0"


#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()






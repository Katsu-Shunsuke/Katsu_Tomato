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

from utils import numpy_to_rosarray 
from instance_segmentation.msg import InstSegRes # need to edit CMakeLists.txt and package.xml

class InstanceSegmentation:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedA/zed_node_A/left/image_rect_color" # left image because disparity map is on left image.
#        self.flg_topic = "instance_segmentation_flg"
        self.flg_topic = "stereo_matching_flg"
        self.result_topic = "instance_segmentation_output"
        self.config_file = '../cascade_mask_rcnn_r50_fpn_1x_tomato.py'
        self.checkpoint_file = '../epoch_2000.pth'
        # output of callback methods
        self.im = None
        self.im_array = None
        self.result = None
        self.maskrcnn = None
        self.flg = None
        self.result_msg = None

    def im_callback(self, msg):
        self.im = msg      
        cv_image = CvBridge().imgmsg_to_cv2(self.im, "bgr8")
        self.im_array = np.array(cv_image)
    
    def main_callback(self, msg):
        if msg.data == "1" and self.im_array is not None:
            self.flg = "1"
            if self.maskrcnn is None:
                self.maskrcnn = init_detector(self.config_file, self.checkpoint_file, device='cuda:0') # change device acordingly
            print("running maskrcnn")
            self.result = inference_detector(self.maskrcnn, self.im_array) # list of list of array
            self.result_msg = self.to_InstSegRes(self.result)
    
    def to_InstSegRes(self, result):
        msg = InstSegRes()
        # bbox
        msg.bbox_stem = numpy_to_rosarray(self.result[0][0], "float32")
        msg.bbox_tomato = numpy_to_rosarray(self.result[0][1], "float32")
        msg.bbox_pedicel = numpy_to_rosarray(self.result[0][2], "float32")
        msg.bbox_sepal = numpy_to_rosarray(self.result[0][3], "float32")
        # mask
        i_stem  = np.transpose(np.nonzero(np.dstack(self.result[1][0])))
        i_tomato = np.transpose(np.nonzero(np.dstack(self.result[1][1])))
        i_pedicel = np.transpose(np.nonzero(np.dstack(self.result[1][2])))
        i_sepal = np.transpose(np.nonzero(np.dstack(self.result[1][3])))
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
    rospy.Subscriber(model.flg_topic, String, model.main_callback)
    pub = rospy.Publisher(model.result_topic, InstSegRes, queue_size=1)
#    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if model.result_msg is not None and model.flg=="1":
#            rospy.loginfo(model.result_msg)
            pub.publish(model.result_msg)
#            r.sleep()
            model.flg = "0"

#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()






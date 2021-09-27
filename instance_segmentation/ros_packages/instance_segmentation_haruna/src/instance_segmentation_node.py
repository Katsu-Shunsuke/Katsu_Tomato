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

# import mmcv
# from mmdet.apis import init_detector, inference_detector

from utils import float32_to_numpy, numpy_to_float32
from instance_segmentation.msg import InstSegRes # need to edit CMakeLists.txt and package.xml

class InstanceSegmentation:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedA/zed_node_A/left/image_rect_color" # left image because disparity map is on left image.
#        self.flg_topic = "instance_segmentation_flg"
        self.flg_topic = "stereo_matching_flg"
        self.result_topic = "instance_segmentation_haruna_output"
        self.config_file = '../cascade_mask_rcnn_r50_fpn_1x_tomato.py'
        self.checkpoint_file = '../epoch_400.pth'
        # output of callback methods
        self.im = None
        self.im_array = None
        self.result = None
        self.maskrcnn = None
        self.flg = None
        self.bbox_stem = None
        self.bbox_tomato = None
        self.bbox_pedicel = None
        self.bbox_sepal = None
        self.mask_stem = None
        self.mask_tomato = None
        self.mask_pedicel = None
        self.mask_sepal = None
        self.result_msg = None

    def im_callback(self, msg):
        self.im = msg      
        cv_image = CvBridge().imgmsg_to_cv2(self.im, "bgr8")
        self.im_array = np.array(cv_image)
    
    def main_callback(self, msg):
        # if msg.data == "1" and self.im_array is not None:
        #     self.flg = "1"
        #     if self.maskrcnn is None:
        #         self.maskrcnn = init_detector(self.config_file, self.checkpoint_file, device='cuda:0') # change device acordingly
        #     self.result = inference_detector(self.maskrcnn, self.im_array) # list of list of array
        #     self.result_msg = self.to_InstSegRes(self.result)
        self.result_msg = InstSegRes()
        h = Header()
        h.stamp = rospy.Time.now()
        h.seq = 0
        h.frame_id = "poo"
        self.result_msg.header = h

    def to_InstSegRes(self, result):
        msg = InstSegRes()
        # bbox
        self.bbox_stem = numpy_to_float32(self.result[0][0])
        self.bbox_tomato = numpy_to_float32(self.result[0][1])
        self.bbox_pedicel = numpy_to_float32(self.result[0][2])
        self.bbox_sepal = numpy_to_float32(self.result[0][3])
        # mask
        self.mask_stem = numpy_to_float32(np.dstack(self.result[1][0]))
        self.mask_tomato = numpy_to_float32(np.dstack(self.result[1][1]))
        self.mask_pedicel = numpy_to_float32(np.dstack(self.result[1][2]))
        self.mask_sepal = numpy_to_float32(np.dstack(self.result[1][3]))
        # assign to each field in msg
        msg.bbox_stem = self.bbox_stem
        msg.bbox_tomato = self.bbox_tomato
        msg.bbox_pedicel = self.bbox_pedicel
        msg.bbox_sepal = self.bbox_sepal
        msg.mask_stem = self.mask_stem
        msg.mask_tomato = self.mask_tomato
        msg.mask_pedicel = self.mask_pedicel
        msg.mask_sepal = self.mask_sepal
        h = Header()
        h.stamp = rospy.Time.now()
        h.seq = 0
        h.frame_id = "poo"
        msg.header = h
        return msg
    
def main():
    rospy.init_node("instance_segmentation_haruna", anonymous=True)
    model = InstanceSegmentation()
    rospy.Subscriber(model.im_topic, Image, model.im_callback)
    rospy.Subscriber(model.flg_topic, String, model.main_callback)
    pub = rospy.Publisher(model.result_topic, InstSegRes, queue_size=10)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if model.result_msg is not None:
            rospy.loginfo(model.result_msg)
            pub.publish(model.result_msg)
            r.sleep()
#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()





# def image_loader(im_right_topic, im_left_topic):
#     rospy.init_node("stereo_matching", anonymous=True)
#     im_right = message_filters.Subscriber(im_right_topic, Image)
#     im_left = message_filters.Subscriber(im_left_topic, Image)
# 
#     ts = message_filters.TimeSynchronizer([image_right, im_left], 5) # better to use message_filters.ApproximateTimeSynchronizer ? lets do queue_size=5 for now. 
#     ts.registerCallback(callback)
#     rospy.spin()



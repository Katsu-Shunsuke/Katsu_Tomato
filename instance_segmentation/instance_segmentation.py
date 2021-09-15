#!/usr/bin/env python

import os
import sys
import rospy

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import mmcv
from mmdet.apis import init_detector, inference_detector

class InstanceSegmentation:
    def __init__(self):
        # topics to subscribe and publish to
        self.camera_topic = "best_view_camera"
        self.im_topic = "/zedA/zed_node_A/left/image_rect_color" # left image because disparity map is on left image.
        self.flg_topic = "instance_segmentation_flg"
        self.depth_topic = "aanet_depth_output"
        # output of callback methods
        self.camera_name = None
        self.im_right = None
        self.im_left = None
        self.array_right = None
        self.array_left = None
        self.depth = None
        self.aanet = None
        self.device = None
        self.flg = None

    def camera_name_callback(self, msg):
        self.camera_name = msg.data

    def right_callback(self, msg):
        self.im_right = msg     
        cv_image = CvBridge().imgmsg_to_cv2(self.im_right, "bgr8")
        self.array_right = np.array(cv_image)
    
    def left_callback(self, msg):
        self.im_left = msg      
        cv_image = CvBridge().imgmsg_to_cv2(self.im_left, "bgr8")
        self.array_left = np.array(cv_image)
    
    def main_callback(self, msg):
        if msg.data == "1" and self.array_right is not None and self.array_left is not None:
            self.flg = "1"
            if self.aanet is None or self.device is None:
                self.aanet, self.device = load_aanet(pretrained_aanet="aanet/pretrained/aanet_kitti15-fb2a0d23.pth")
            self.depth = aanet_predict(self.array_right, self.array_left,
                                       self.aanet, self.device).astype(np.float32) # to publish as array has to be float32 for some reason
    
def main():
    rospy.init_node("stereo_matching", anonymous=True)
    sm = StereoMatching()
    rospy.Subscriber(sm.camera_topic, String, sm.camera_name_callback)
    rospy.Subscriber(sm.right_topic, Image, sm.right_callback)
    rospy.Subscriber(sm.left_topic, Image, sm.left_callback)
    rospy.Subscriber(sm.flg_topic, String, sm.main_callback)
    pub = rospy.Publisher(sm.depth_topic, numpy_msg(Floats), queue_size=10)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(sm.depth)
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



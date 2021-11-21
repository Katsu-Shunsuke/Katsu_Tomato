#!/usr/bin/env python

import os
import sys
import rospy

from matplotlib import pyplot as plt

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray, Float64MultiArray, MultiArrayDimension
# from rospy.numpy_msg import numpy_msg
# from rospy_tutorials.msg import Floats

sys.path.append("/root/catkin_ws/src/stereo_matching/src/aanet")
from aanet import load_aanet, aanet_predict

from ros_utils import numpy_to_float

class StereoMatching:
    def __init__(self):
        # topics to subscribe and publish to
        self.camera_topic = "best_view_camera"
#        self.right_topic = "best_view_im_right"
#        self.left_topic = "best_view_im_left"
        self.right_topic = "/zedA/zed_node_A/right/image_rect_color"
        self.left_topic = "/zedA/zed_node_A/left/image_rect_color"
        self.flg_topic = "stereo_matching_flg"
        self.depth_arr_topic = "aanet_depth_array_output"
        self.depth_im_topic = "aanet_depth_image_output"
#        self.pretrained_aanet = "aanet/pretrained/aanet_sceneflow-5aa5a24e.pth"
        self.pretrained_aanet = "aanet/pretrained/aanet+_sceneflow-d3e13ef0.pth"
#        self.in_shape = (1280, 720) # BE CAREFUL, dimensions are flipped for c2.resize()
        self.in_shape = (960, 540) # BE CAREFUL, dimensions are flipped for c2.resize()
        self.out_shape = (1920, 1080) # must be consistent with instance segmentation 
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
        self.depth_arr_msg = None
        self.depth_im_msg = None

    def camera_name_callback(self, msg):
        self.camera_name = msg.data

    def right_callback(self, msg):
        self.im_right = msg     
        cv_image = CvBridge().imgmsg_to_cv2(self.im_right, "bgr8")
        im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im_rgb, dsize=self.in_shape)
        self.array_right = np.array(im_resized)
    
    def left_callback(self, msg):
        self.im_left = msg      
        cv_image = CvBridge().imgmsg_to_cv2(self.im_left, "bgr8")
        im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im_rgb, dsize=self.in_shape)
        self.array_left = np.array(im_resized)
    
    def main_callback(self, msg):
        if msg.data == "1" and self.array_right is not None and self.array_left is not None:
            self.flg = "1"
            if self.aanet is None or self.device is None:
                self.aanet, self.device = load_aanet(pretrained_aanet=self.pretrained_aanet)
            print("running aanet")
            depth_raw = aanet_predict(self.array_right, self.array_left, self.aanet, self.device).astype(np.float32) # to publish as array has to be float32 for some reason
            self.depth = cv2.resize(depth_raw, dsize=self.out_shape)
            plt.imshow(self.depth)
            plt.savefig("depth.png")
            self.depth_arr_msg = numpy_to_float(self.depth, "float32")
            print(np.min(self.depth))
            print(np.max(self.depth))

#            depth_im = Image()
#            depth_im_array = (self.depth / np.max(self.depth) * 255).astype(np.uint8)
#            depth_im.data = tuple(depth_im_array.flatten())
#            depth_im.height, depth_im.width = depth_im_array.shape
#            depth_im.step = depth_im.width
#            self.depth_im_msg = depth_im

            depth_im_array = (self.depth / np.max(self.depth) * (2**16 - 1)).astype(np.uint16)
            self.depth_im_msg = CvBridge().cv2_to_imgmsg(depth_im_array, "mono16")
        


def main():
    rospy.init_node("stereo_matching", anonymous=True)
    sm = StereoMatching()
    rospy.Subscriber(sm.camera_topic, String, sm.camera_name_callback)
    rospy.Subscriber(sm.right_topic, Image, sm.right_callback)
    rospy.Subscriber(sm.left_topic, Image, sm.left_callback)
    rospy.Subscriber(sm.flg_topic, String, sm.main_callback)
    pub_depth_arr = rospy.Publisher(sm.depth_arr_topic, Float32MultiArray, queue_size=1)
    pub_depth_im = rospy.Publisher(sm.depth_im_topic, Image, queue_size=1)
#    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if sm.depth_arr_msg is not None and sm.depth_im_msg is not None and sm.flg=="1":
            pub_depth_arr.publish(sm.depth_arr_msg)
            pub_depth_im.publish(sm.depth_im_msg)
#            r.sleep()
            sm.flg = "0"

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



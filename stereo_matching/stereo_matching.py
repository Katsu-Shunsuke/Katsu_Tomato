#!/usr/bin/env python

import os
import sys
import rospy

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
# from rospy.numpy_msg import numpy_msg
# from rospy_tutorials.msg import Floats

sys.path.append("/root/catkin_ws/src/stereo_matching/src/aanet")
from aanet import load_aanet, aanet_predict

class StereoMatching:
    def __init__(self):
        # topics to subscribe and publish to
        self.camera_topic = "best_view_camera"
#        self.right_topic = "best_view_im_right"
#        self.left_topic = "best_view_im_left"
        self.right_topic = "/zedA/zed_node_A/right/image_rect_color"
        self.left_topic = "/zedA/zed_node_A/left/image_rect_color"
        self.flg_topic = "stereo_matching_flg"
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
        self.depth_msg = None

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
            self.depth_msg = self.to_ros_array(self.depth)

    def to_ros_array(self, array):
        msg = Float32MultiArray()
        msg.data = array.flatten()
        msg.layout.data_offset = 0
#        msg.layout.dim = [MultiArrayDimension() for i in range(array.shape[0])]
        msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        msg.layout.dim[0].label = "height"
        msg.layout.dim[0].size = array.shape[0]
        msg.layout.dim[0].stride = array.shape[0] * array.shape[1]
        msg.layout.dim[1].label = "width"
        msg.layout.dim[1].size = array.shape[1]
        msg.layout.dim[0].stride = array.shape[1]
        return msg
        


def main():
    rospy.init_node("stereo_matching", anonymous=True)
    sm = StereoMatching()
    rospy.Subscriber(sm.camera_topic, String, sm.camera_name_callback)
    rospy.Subscriber(sm.right_topic, Image, sm.right_callback)
    rospy.Subscriber(sm.left_topic, Image, sm.left_callback)
    rospy.Subscriber(sm.flg_topic, String, sm.main_callback)
    pub = rospy.Publisher(sm.depth_topic, Float32MultiArray, queue_size=10)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if sm.depth_msg is not None:
            pub.publish(sm.depth_msg)
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



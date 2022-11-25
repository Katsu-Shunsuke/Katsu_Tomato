#!/usr/bin/env python

import os
import sys
import rospy

import numpy as np
from std_msgs.msg import String, Int32MultiArray


def main():
    camera_topic = "best_view_camera"
    camera = "A"
    flg_topic = "stereo_matching_flg"
    flg = "1"

    rospy.init_node("dummy_camera", anonymous=True)
    pub1 = rospy.Publisher(camera_topic, String, queue_size=10)
    pub2 = rospy.Publisher(flg_topic, String, queue_size=10)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pub1.publish(camera)
#        pub2.publish(flg)
        rate.sleep()
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



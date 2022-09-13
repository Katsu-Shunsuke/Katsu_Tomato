#/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image

class ImageSynchronizer:
    def __init__(self):
        # if zedB is the camera to be used then change accordingly.
        # topics in 
        self.left_image_in = "/zedm/zed_node/left/image_rect_color"
        self.right_image_in = "/zedm/zed_node/right/image_rect_color"
        # topics out
        self.left_image_out = "/zedm/zed_node/left/image_rect_color_synchronized"
        self.right_image_out = "/zedm/zed_node/right/image_rect_color_synchronized"
        # msg
        self.left_image_msg = None
        self.right_image_msg = None

    def image_callback(self, left_msg, right_msg):
        self.left_image_msg = left_msg
        self.right_image_msg = right_msg

def main():
    rospy.init_node("image_synchronizer", anonymous=True)
    synchronizer = ImageSynchronizer()
    # subscribers
    left_sub = message_filters.Subscriber(synchronizer.left_image_in, Image)
    right_sub = message_filters.Subscriber(synchronizer.right_image_in, Image)
    ts = message_filters.TimeSynchronizer([left_sub, right_sub], 10)
    ts.registerCallback(synchronizer.image_callback)
    # publishers
    pub_left_image = rospy.Publisher(synchronizer.left_image_out, Image, queue_size=1)
    pub_right_image = rospy.Publisher(synchronizer.right_image_out, Image, queue_size=1)
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        if synchronizer.left_image_msg is not None and synchronizer.right_image_msg is not None:
            pub_left_image.publish(synchronizer.left_image_msg)
            pub_right_image.publish(synchronizer.right_image_msg)
            rate.sleep()
    rospy.spin()

if __name__=="__main__":
    main()





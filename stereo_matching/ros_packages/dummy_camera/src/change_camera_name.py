#/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo

class ChangeCameraName:
    def __init__(self):
        # if zedB is the camera to be used then change accordingly.
        # topics in 
        self.left_image_in = "/zedA/zed_node_A/left/image_rect_color"
        self.right_image_in = "/zedA/zed_node_A/right/image_rect_color"
        self.left_camera_info_in = "/zedA/zed_node_A/left/camera_info"
        # topics out
        self.left_image_out = "/zedm/zed_node/left/image_rect_color"
        self.right_image_out = "/zedm/zed_node/right/image_rect_color"
        self.left_camera_info_out = "/zedm/zed_node/left/camera_info"
        # msg
        self.left_image_msg = None
        self.right_image_msg = None
        self.left_camera_info_msg = None

    def left_callback(self, msg):
        self.left_image_msg = msg

    def right_callback(self, msg):
        self.right_image_msg = msg

    def left_camera_info_callback(self, msg):
        self.left_camera_info_msg = msg
        self.left_camera_info_msg.header.frame_id = "zedm_left_camera_optical_frame"

def main():
    rospy.init_node("change_camera_name", anonymous=True)
    ccn = ChangeCameraName()
    # subscribers
    rospy.Subscriber(ccn.left_image_in, Image, ccn.left_callback)
    rospy.Subscriber(ccn.right_image_in, Image, ccn.right_callback)
    rospy.Subscriber(ccn.left_camera_info_in, CameraInfo, ccn.left_camera_info_callback)
    # publishers
    pub_left_image = rospy.Publisher(ccn.left_image_out, Image, queue_size=1)
    pub_right_image = rospy.Publisher(ccn.right_image_out, Image, queue_size=1)
    pub_left_camera_info = rospy.Publisher(ccn.left_camera_info_out, CameraInfo, queue_size=1)
    while not rospy.is_shutdown():
        if ccn.left_image_msg is not None and ccn.right_image_msg is not None and ccn.left_camera_info_msg is not None:
            pub_left_image.publish(ccn.left_image_msg)
            pub_right_image.publish(ccn.right_image_msg)
            pub_left_camera_info.publish(ccn.left_camera_info_msg)
    rospy.spin()

if __name__=="__main__":
    main()




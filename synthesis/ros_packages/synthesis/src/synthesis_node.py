#!/usr/bin/env python

import os
import sys
import rospy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension, Header
from geometry_msgs.msg import Point32
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils import rosarray_to_numpy, stereo_reconstruction
from synthesis.msg import InstSegRes # need to edit CMakeLists.txt and package.xml

class Synthesis:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedA/zed_node_A/left/image_rect_color" # left image because disparity map is on left image.
#        self.flg_topic = "synthesis_flg"
        self.flg_topic = "stereo_matching_flg"
        self.result_topic = "synthesis_output"
        self.depth_topic = "aanet_depth_output"
        self.instseg_topic = "instance_segmentation_output"
        # output of callback methods
        self.depth = None
        self.xyz = None
        self.im_array = None
        self.result = None
        self.flg = None
        self.result_msg = None
        self.bbox_stem = None
        self.bbox_tomato = None
        self.bbox_pedicel = None
        self.bbox_sepal = None
        self.mask_stem = None
        self.mask_tomato = None
        self.mask_pedicel = None
        self.mask_sepal = None
        self.this_pedicel = None
        self.cut_point = None

    def im_callback(self, msg):
        print("received image")
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.im_array = np.array(im_rgb)

    def depth_callback(self, msg):
        print("received disparity map")
        self.depth = rosarray_to_numpy(msg)
        # convert to world coordinates
        self.xyz = stereo_reconstruction(self.depth)
        
    def instseg_callback(self, msg):
        print("received instsegres")
        self.bbox_stem = rosarray_to_numpy(msg.bbox_stem)
        self.bbox_tomato = rosarray_to_numpy(msg.bbox_tomato)
        self.bbox_pedicel = rosarray_to_numpy(msg.bbox_pedicel)
        self.bbox_sepal = rosarray_to_numpy(msg.bbox_sepal)
        self.mask_stem = rosarray_to_numpy(msg.mask_stem) # indices not bool array
        self.mask_tomato = rosarray_to_numpy(msg.mask_tomato)
        self.mask_pedicel = rosarray_to_numpy(msg.mask_pedicel)
        self.mask_sepal = rosarray_to_numpy(msg.mask_sepal)

    def main_callback(self, msg):
        if msg.data == "1" and self.xyz is not None and self.mask_sepal is not None and self.im_array is not None:
            print("running main callback")
            self.flg = "1"
            bbox_top = 0.5
            ripeness_threshold = 1
            pedicel_cut_prop = 0.3
            ripeness_percentile = 0.25
            n_pedicels = np.max(self.mask_pedicel[:,2]).astype(int)
            for i in range(n_pedicels):
                # choose a pedicel (cannot loop for every pedicel in the image because cog can change)
                this_pedicel = self.mask_pedicel[self.mask_pedicel[:,2]==i]
                # its possible for no pedicels to be detected in an image 
                x = this_pedicel[:,1].astype("int") # actually, better to send msg as uint32
                y = this_pedicel[:,0].astype("int")
                # obtain the end with smaller y value
                x_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,1] # index zero since there are probs multiple
                y_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,0] # index zero since there are probs multiple
                # if that end point is within a tomato bbox, obtain a small patch centered around the tomato mask
                tomato_is_ripe = False
                for j, this_tomato in enumerate(self.bbox_tomato):
                    x_min, y_min, x_max, y_max = this_tomato[0], this_tomato[1], this_tomato[2], this_tomato[3]
                    if (x_end > x_min and x_end < x_max) and (y_end > y_min and y_end < bbox_top * (y_max - y_min) + y_min):
                        # compute ripeness, and if ripeness is above a certain threshold return coordniate of point half way along the pedicel
                        mask_indices = self.mask_tomato[self.mask_tomato[:,2]==j].astype(int)
                        # probably unnecessary to reduce if just using rgb info
                        tomato_pixels = self.im_array[mask_indices[:,0], mask_indices[:,1]] # should be nx3
                        r, g, b = tomato_pixels[:,0], tomato_pixels[:,1], tomato_pixels[:,2]
                        ripeness = np.sort((r - g)/(r + g + b))
                        lower_index = int(ripeness_percentile * len(ripeness))
                        upper_index = int((1 - ripeness_percentile) * len(ripeness))
                        ripeness = np.mean(ripeness[lower_index: upper_index])
                        if ripeness < ripeness_threshold:
                            # send this info to the manipulator  
                            self.this_pedicel = this_pedicel

                            index = int((1 - pedicel_cut_prop) * len(y))
                            y_cut = np.partition(y, index)[index]
                            # run polynomial fitting
                            coefs = np.polyfit(y, x, 6) # fifth deg for now
                            # predict
                            x_pred = np.polyval(coefs, y_cut).astype("int")
                            self.cut_point = self.xyz[y_cut, x_pred, :]
                            res = Point32()
                            res.x, res.y, res.z = self.cut_point[0], self.cut_point[1], self.cut_point[2]
                            self.result_msg = res 
                            # tangent vector (3D)

#                            plt.imshow((self.depth / np.max(self.depth) * 255).astype(np.uint8), cmap="gray")
                            plt.imshow(self.im_array)
                            plt.plot(np.polyval(coefs, y), y, "bo", ms=0.5)
                            plt.plot(x_pred, y_cut, "ro", ms=0.5)
#                            plt.annotate(str(ripeness), (x_pred, y_cut))
                            plt.savefig("images/test" + str(i) + str(j) + ".png", dpi=300)
                            tomato_is_ripe = True

#                            break
#                if tomato_is_ripe:
#                    break
                

    
def main():
    rospy.init_node("synthesis", anonymous=True)
    synthesizer = Synthesis()
    rospy.Subscriber(synthesizer.im_topic, Image, synthesizer.im_callback)
    rospy.Subscriber(synthesizer.flg_topic, String, synthesizer.main_callback)
    rospy.Subscriber(synthesizer.depth_topic, Float32MultiArray, synthesizer.depth_callback)
    rospy.Subscriber(synthesizer.instseg_topic, InstSegRes, synthesizer.instseg_callback)
    pub = rospy.Publisher(synthesizer.result_topic, Point32, queue_size=1)
#    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if synthesizer.result_msg is not None and synthesizer.flg=="1":
#            rospy.loginfo(model.result_msg)
            pub.publish(synthesizer.result_msg)
#            r.sleep()
            synthesizer.flg = "0"

#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()






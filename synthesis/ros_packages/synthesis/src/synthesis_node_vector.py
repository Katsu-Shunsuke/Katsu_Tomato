#!/usr/bin/env python

import os
import sys
import rospy
import tf

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.io import savemat
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String, Float32MultiArray, Float64MultiArray, MultiArrayDimension, Header
from geometry_msgs.msg import Point32, Vector3
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils import rosarray_to_numpy, stereo_reconstruction, polynomial_derivative, rotation_matrix_from_vectors, generate_pc2_message
from synthesis.msg import InstSegRes, CutPoint, ExitCode # need to edit CMakeLists.txt and package.xml

class Synthesis:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedm/zed_node/left/image_rect_color" # left image because disparity map is on left image.
#        self.flg_topic = "synthesis_flg"
        self.flg_topic = "stereo_matching_flg"
        self.result_topic = "synthesis_cutpoint_output"
        self.depth_topic = "aanet_depth_array_output"
        self.instseg_topic = "instance_segmentation_array_output"
        self.pc2_topic = "synthesis_pc2_output"
        self.exit_code_pub = rospy.Publisher("large_tomato/exit_code", ExitCode, queue_size=1)
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
        self.quaternion = None
        self.translation = None
        self.point_cloud = None
        self.instseg_finished = False
        self.sm_finished = False

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
        if self.xyz is None:
            exit_code = ExitCode()
            exit_code.exit_code = ExitCode.CODE_PEDICEL_STEREO_MATCHING_FAILED
            self.exit_code_pub.publish(exit_code)
            return
        self.sm_finished = True
        
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
        if self.mask_sepal is None:
            exit_code = ExitCode()
            exit_code.exit_code = ExitCode.CODE_PEDICEL_INSTSEG_FAILED
            self.exit_code_pub.publish(exit_code)
            return
        self.instseg_finished = True

    def update_flg(self, msg):
        if msg.data == "1":
            self.flg = "1"

    def main_callback(self):
        if self.xyz is not None and self.mask_sepal is not None and self.im_array is not None:
            print("running main callback")
            bbox_top = 0.5
            ripeness_threshold = 10
            pedicel_cut_prop = 0.5
            ripeness_percentile = 0.25
            deg = 5
            n_pedicels = np.max(self.mask_pedicel[:,2]).astype(int) if self.mask_pedicel.size else 0
            
            # publish test pointcloud2 message
            self.point_cloud = generate_pc2_message(self.xyz, self.im_array)

            # sort pedicels
            mask_pedicel_list = [self.mask_pedicel[self.mask_pedicel[:,2]==i][:, :2] for i in range(n_pedicels)] #i since self.mask_pedicel starts at zero
            min_y = [np.mean(i[:,0]) for i in mask_pedicel_list]
            mask_pedicel_sorted = [i for _, i in sorted(zip(min_y, mask_pedicel_list))] # pedicels sorted from small y-values (vertically higher) first
            print(n_pedicels)
            print(min_y)

            for this_pedicel in mask_pedicel_sorted:
                # choose a pedicel (cannot loop for every pedicel in the image because cog can change)
                if this_pedicel.size:
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
                            rgb_not_zero = np.sum(tomato_pixels, axis=1).astype("bool")
                            tomato_pixels = tomato_pixels[rgb_not_zero, :]
                            r, g, b = tomato_pixels[:,0], tomato_pixels[:,1], tomato_pixels[:,2]
                            ripeness = np.sort((r - g)/(r + g + b))
                            lower_index = int(ripeness_percentile * len(ripeness))
                            upper_index = int((1 - ripeness_percentile) * len(ripeness))
                            ripeness = np.mean(ripeness[lower_index: upper_index])
                            if ripeness < ripeness_threshold:
                                # send this info to the manipulator  
                                self.this_pedicel = this_pedicel
                                pedicel_xyz = self.xyz[y, x, :]
                                x_glob, y_glob, z_glob = pedicel_xyz[:, 0], pedicel_xyz[:, 1], pedicel_xyz[:, 2]
                                # polynomial regeression
                                coefs_yx = np.polyfit(y_glob, x_glob, deg=deg)
                                coefs_yz = np.polyfit(y_glob, z_glob, deg=deg)
                                # need to think about coordinate system
                                index = int((pedicel_cut_prop) * len(y_glob))
                                y_cut = np.partition(y_glob, index)[index]
                                # predict
                                x_pred = np.polyval(coefs_yx, y_cut)
                                z_pred = np.polyval(coefs_yz, y_cut)
                                # pedicel xyz position
                                cut_point = CutPoint()
                                point = Point32()
                                point.x, point.y, point.z = x_pred, y_cut, z_pred
                                cut_point.xyz = point
                                # tangent vector (3D)
                                deriv_coefs_yx = polynomial_derivative(coefs_yx)
                                deriv_coefs_yz = polynomial_derivative(coefs_yz)
                                deriv_yx = np.polyval(deriv_coefs_yx, y_cut)
                                deriv_yz = np.polyval(deriv_coefs_yz, y_cut)
                                r = np.array([deriv_yx, 1, deriv_yz])
                                r = r / np.linalg.norm(r) # unit vector
                                dir_vector = Vector3()
                                dir_vector.x, dir_vector.y, dir_vector.z = r
                                cut_point.tangent = dir_vector
                                # custom message with point and tangent vector
                                self.result_msg = cut_point
                                
                                # calculate rotation matrix to align pedicel in scissor coordinate y-direction and tangent vector
                                vec1 = np.array([0.0, 1.0, 0.0]) # camera coordinates
                                vec2 = r # scissor coordinates
                                rot = rotation_matrix_from_vectors(vec1, vec2)
                                # quaternion and translation
                                rot_eye = np.eye(4)
                                rot_eye[:3, :3] = rot # rotation matrix has to be 4x4 for the tf function
                                self.quaternion = tf.transformations.quaternion_from_matrix(rot_eye) # no need to convert for quaternion because its just direction
                                self.translation = tuple(np.array([x_pred, y_cut, z_pred]) * 10**(-3)) # mm to m
    
    
    #                            # save result as matlab matrices
    #                            savemat("to_local/point_clouds/xyz.mat", {"xyz": self.xyz})
    #                            x_curve = np.polyval(coefs_yx, y_glob)
    #                            z_curve = np.polyval(coefs_yz, y_glob)
    #                            curve = np.vstack((x_curve, y_glob, z_curve)).T
    #                            savemat("to_local/point_clouds/curve" + str(j) + ".mat", {"curve": curve})
    #                            savemat("to_local/point_clouds/point" + str(j) + ".mat", {"point": [x_pred, y_cut, z_pred]})
    #                            mag_max = 20
    #                            t = np.array([x_pred, y_cut, z_pred])
    #                            tangent_line = np.vstack([mag * r + t for mag in np.linspace(-mag_max, mag_max, 30)])
    #                            savemat("to_local/point_clouds/tangent" + str(j) + ".mat", {"tangent": tangent_line})
    #                            savemat("to_local/point_clouds/image.mat", {"image": self.im_array})
    #
    #                            # save result as numpy arrays
    #                            np.save("to_local/point_clouds/xyz.npy", self.xyz)
    #                            np.save("to_local/point_clouds/curve" + str(j) + ".npy", curve)
    #                            np.save("to_local/point_clouds/point" + str(j) + ".npy", [x_pred, y_cut, z_pred])
    #                            np.save("to_local/point_clouds/tangent" + str(j) + ".npy", tangent_line)
    #                            np.save("to_local/point_clouds/image.npy", self.im_array)
    #                            
    #                            plt.imshow(self.depth)
    #                            plt.savefig("to_local/depth.png")
    #                            
    #                            np.save("to_local/xyz.npy", self.xyz)
    #                            print(np.min(self.xyz[:,:,2]))
    #                            print(np.max(self.xyz[:,:,2]))
    
                                tomato_is_ripe = True
    
                                break
                    if tomato_is_ripe:
                        break
    
def main():
    rospy.init_node("synthesis", anonymous=True)
    synthesizer = Synthesis()
    rospy.Subscriber(synthesizer.im_topic, Image, synthesizer.im_callback)
    rospy.Subscriber(synthesizer.flg_topic, String, synthesizer.update_flg)
    rospy.Subscriber(synthesizer.depth_topic, Float32MultiArray, synthesizer.depth_callback)
    rospy.Subscriber(synthesizer.instseg_topic, InstSegRes, synthesizer.instseg_callback)
    pub_cutpoint = rospy.Publisher(synthesizer.result_topic, CutPoint, queue_size=1)
    pub_pointcloud = rospy.Publisher(synthesizer.pc2_topic, PointCloud2, queue_size=1)
#    r = rospy.Rate(10)
    br = tf.TransformBroadcaster()
    exit_code = ExitCode()
    while not rospy.is_shutdown():
        if synthesizer.flg == "1":
            if synthesizer.instseg_finished and synthesizer.sm_finished:
                rospy.loginfo("Start synthesis.")
                synthesizer.main_callback()

                if synthesizer.point_cloud is not None:
                    pub_pointcloud.publish(synthesizer.point_cloud)

                if synthesizer.quaternion is not None and synthesizer.translation is not None and synthesizer.point_cloud is not None:
        #        if synthesizer.result_msg is not None and synthesizer.flg=="1":
        #            rospy.loginfo(model.result_msg)
                    pub_cutpoint.publish(synthesizer.result_msg)
                    pub_pointcloud.publish(synthesizer.point_cloud)
        #            r.sleep()
                    br.sendTransform(synthesizer.translation, synthesizer.quaternion, rospy.Time.now(), "/tomato_pedicel", "/zedm_left_camera_optical_frame")
                    exit_code.exit_code = ExitCode.CODE_PEDICEL_DETECTION_SUCCESS
                    synthesizer.exit_code_pub.publish(exit_code)
                else:
                    rospy.loginfo("pedicel is not detected.")
                    exit_code.exit_code = ExitCode.CODE_PEDICEL_DETECTION_FAILED
                    synthesizer.exit_code_pub.publish(exit_code)
            
                synthesizer.flg = "0"


#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()







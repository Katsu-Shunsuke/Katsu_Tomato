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

from utils import rosarray_to_numpy, stereo_reconstruction, polynomial_derivative, generate_pc2_message, filter_instseg, visualize_output, curve_fitting
from pedicel_quaternion import calc_pedicel_quaternion, calc_tomato_center, remove_outliers 
from synthesis.msg import InstSegRes, CutPoint, ExitCode # need to edit CMakeLists.txt and package.xml

class Synthesis:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedm/zed_node/left/image_rect_color_synchronized" # left image because disparity map is on left image.
#        self.flg_topic = "synthesis_flg"
        self.flg_topic = "stereo_matching_flg"
        self.result_topic = "synthesis_cutpoint_output"
        self.depth_topic = "aanet_depth_array_output"
        self.instseg_topic = "instance_segmentation_array_output"
        self.image_pc2_topic = "synthesis_image_pc2_output"
        self.polynomial_pc2_topic = "synthesis_polynomial_pc2_output"
        self.tomato_center_pc2_topic = "synthesis_tomato_center_pc2_output"
        self.pedicel_end_pc2_topic = "synthesis_pedicel_end_pc2_output"
        self.instseg_im_filtered_topic = "instance_segmentation_filtered_image_output"
        self.exit_code_pub = rospy.Publisher("large_tomato/exit_code", ExitCode, queue_size=1)
        self.publish_filtered_instseg_image = False
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
        self.image_point_cloud = None
        self.polynomial_point_cloud = None
        self.tomato_center_point_cloud = None
        self.pedicel_end_point_cloud = None
        self.instseg_im_filtered = None
        self.instseg_finished = False
        self.sm_finished = False
        self.tf_computed = False

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
        # dont think this is the best way but this callback must wait until im_callback has finished.
        if self.publish_filtered_instseg_image:
            while self.im_array is None:
                pass
        print("received instsegres")
        threshold_stem = rospy.get_param("threshold_stem", 0.3)
        threshold_tomato = rospy.get_param("threshold_tomato", 0.1)
        threshold_pedicel = rospy.get_param("threshold_pedicel", 0.1)
        threshold_sepal = rospy.get_param("threshold_sepal", 0.2)
        self.bbox_stem, self.mask_stem = filter_instseg(rosarray_to_numpy(msg.bbox_stem),
                                                        rosarray_to_numpy(msg.mask_stem), threshold_stem)
        self.bbox_tomato, self.mask_tomato = filter_instseg(rosarray_to_numpy(msg.bbox_tomato),
                                                            rosarray_to_numpy(msg.mask_tomato), threshold_tomato)
        self.bbox_pedicel, self.mask_pedicel = filter_instseg(rosarray_to_numpy(msg.bbox_pedicel),
                                                              rosarray_to_numpy(msg.mask_pedicel), threshold_pedicel)
        self.bbox_sepal, self.mask_sepal = filter_instseg(rosarray_to_numpy(msg.bbox_sepal),
                                                          rosarray_to_numpy(msg.mask_sepal), threshold_sepal)

        # visualize filtered result
        if self.publish_filtered_instseg_image:
            instseg_result = [[self.bbox_stem, self.bbox_tomato, self.bbox_pedicel, self.bbox_sepal],
                              [self.mask_stem, self.mask_tomato, self.mask_pedicel, self.mask_sepal]]
            instseg_im_filtered = visualize_output(self.im_array, instseg_result, threshold_per_class=[0.0, 0.0, 0.0, 0.0], show_bbox=True, save_im=False)
            self.instseg_im_filtered = CvBridge().cv2_to_imgmsg(instseg_im_filtered, "rgb8")

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
        print("running main callback")
        bbox_top = 0.5
        ripeness_threshold = rospy.get_param("ripeness_threshold", 10)
        ripeness_percentile = 0.25
        pedicel_calc_mode = rospy.get_param("pedicel_calc_mode", 3)
        which_pedicel = rospy.get_param("which_pedicel", 0)
        
        print("\nbbox_top: {}\nripeness_threshold: {}\nripeness_percentile: {}\npedicel_calc_mode: {}\nwhich_pedicel: {}\n".format(bbox_top,
            ripeness_threshold, ripeness_percentile, pedicel_calc_mode, which_pedicel))

        # publish test pointcloud2 message
        self.image_point_cloud = generate_pc2_message(self.xyz, self.im_array)

        # sort pedicels
        n_pedicels = int(len(self.mask_pedicel))
        min_y = [np.mean(i[:,0]) for i in self.mask_pedicel]
        mask_pedicel_sorted = [i for _, i in sorted(zip(min_y, self.mask_pedicel))] # pedicels sorted from small y-values (vertically higher) first
        print("n_pedicels:", n_pedicels, "\n")

        if n_pedicels != 0:
            if which_pedicel > n_pedicels - 1:
                raise Exception("which_pedicel must be an integer in the range 0, ..., n_pedicels-1")
            mask_pedicel_sorted = mask_pedicel_sorted[which_pedicel:]

        # if mask_pedicel_sorted is empty then this loop is skipped. 
        for this_pedicel in mask_pedicel_sorted:
            # choose a pedicel (cannot loop for every pedicel in the image because cog can change)
            x = this_pedicel[:,1].astype("int") # actually, better to send msg as uint32
            y = this_pedicel[:,0].astype("int")
            # obtain the end with smaller y value
            x_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,1] # index zero since there are probs multiple
            y_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,0] # index zero since there are probs multiple
            # if that end point is within a tomato bbox, obtain a small patch centered around the tomato mask
            overlapping_tomatoes = []
            xy_centers = []
            for j, this_tomato in enumerate(self.bbox_tomato):
#                x_min, y_min, w, h = this_tomato[:4]
#                x_max = x_min + w
#                y_max = y_min + h
#                x_center = x_min + w/2
#                y_center = y_min + h/2
                x_min, y_min, x_max, y_max = this_tomato[:4]
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                if (x_end > x_min and x_end < x_max) and (y_end > y_min and y_end < bbox_top * (y_max - y_min) + y_min):
                    overlapping_tomatoes.append(j)
                    xy_centers.append([x_center, y_center])
            
            dists = []
            if len(overlapping_tomatoes) > 1:
                for xy_center in xy_centers:
                    dist = np.sqrt((xy_center[0] - x_end)**2 + (xy_center[1] - y_end)**2)
                    #dist = np.abs(x_center - x_end)
                    dists.append(dist)
                j_final = overlapping_tomatoes[dists.index(min(dists))]
            elif len(overlapping_tomatoes) == 1:
                j_final = overlapping_tomatoes[0]
            else: # zero
                j_final = None

            print("overlapping_tomatoes", overlapping_tomatoes)
            print("j_final", j_final)
            print("dists", dists)

            if j_final is not None:
                # compute ripeness, and if ripeness is above a certain threshold return coordniate of point half way along the pedicel
                mask_indices = self.mask_tomato[j_final].astype(int)
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

                    # calculate tomato center
                    tomato_xyz = self.xyz[mask_indices[:,0], mask_indices[:,1], :] # should be nx3
                    tomato_center, tomato_r = calc_tomato_center(tomato_xyz)
                    print("tomato_center:", tomato_center)
                    self.tomato_center_point_cloud = generate_pc2_message(tomato_center, np.array([0, 255, 255]), sampling_prop=1)

                    cut_point, dir_vector, pedicel_end, curve = curve_fitting(x_glob, y_glob, z_glob, mode="polynomial", tomato_center=tomato_center, tomato_r=tomato_r)

                    # cutpoint
                    cut_point_msg = CutPoint()
                    point = Point32()
                    point.x, point.y, point.z = cut_point
                    cut_point_msg.xyz = point
                    # direction vector
                    dir_vector_msg = Vector3()
                    dir_vector_msg.x, dir_vector_msg.y, dir_vector_msg.z = dir_vector
                    cut_point_msg.tangent = dir_vector_msg
                    self.result_msg = cut_point_msg

                    # calculate rotation matrix to align pedicel in scissor coordinate y-direction and tangent vector
                    vec1 = np.array([0.0, 1.0, 0.0]) # camera coordinates
                    vec2 = dir_vector # scissor coordinates
                    self.pedicel_end_point_cloud = generate_pc2_message(pedicel_end, np.array([255, 0, 255]), sampling_prop=1)
                    self.quaternion = calc_pedicel_quaternion(vec1, vec2, cutpoint=cut_point, tomato_center=tomato_center, pedicel_end=pedicel_end, mode=pedicel_calc_mode)
                    self.translation = tuple(cut_point * 10**(-3)) # mm to m
    
                    # visualize curve-fitted polynomial
                    rgb = np.tile(np.array([255,0,0]), (len(curve), 1))
                    self.polynomial_point_cloud = generate_pc2_message(curve, rgb)
    
    #                # save result as matlab matrices
    #                savemat("to_local/point_clouds/xyz.mat", {"xyz": self.xyz})
    #                x_curve = np.polyval(coefs_yx, y_glob)
    #                z_curve = np.polyval(coefs_yz, y_glob)
    #                curve = np.vstack((x_curve, y_glob, z_curve)).T
    #                savemat("to_local/point_clouds/curve" + str(j) + ".mat", {"curve": curve})
    #                savemat("to_local/point_clouds/point" + str(j) + ".mat", {"point": [x_pred, y_cut, z_pred]})
    #                mag_max = 20
    #                t = np.array([x_pred, y_cut, z_pred])
    #                tangent_line = np.vstack([mag * r + t for mag in np.linspace(-mag_max, mag_max, 30)])
    #                savemat("to_local/point_clouds/tangent" + str(j) + ".mat", {"tangent": tangent_line})
    #                savemat("to_local/point_clouds/image.mat", {"image": self.im_array})
    #
    #                # save result as numpy arrays
    #                np.save("to_local/point_clouds/xyz.npy", self.xyz)
    #                np.save("to_local/point_clouds/curve" + str(j) + ".npy", curve)
    #                np.save("to_local/point_clouds/point" + str(j) + ".npy", [x_pred, y_cut, z_pred])
    #                np.save("to_local/point_clouds/tangent" + str(j) + ".npy", tangent_line)
    #                np.save("to_local/point_clouds/image.npy", self.im_array)
    #                
    #                plt.imshow(self.depth)
    #                plt.savefig("to_local/depth.png")
    #                
    #                np.save("to_local/xyz.npy", self.xyz)
    #                print(np.min(self.xyz[:,:,2]))
    #                print(np.max(self.xyz[:,:,2]))
    
                    self.tf_computed = True
            break
    
    
def main():
    rospy.init_node("synthesis", anonymous=True)
    synthesizer = Synthesis()
    rospy.Subscriber(synthesizer.im_topic, Image, synthesizer.im_callback)
    rospy.Subscriber(synthesizer.flg_topic, String, synthesizer.update_flg)
    rospy.Subscriber(synthesizer.depth_topic, Float32MultiArray, synthesizer.depth_callback)
    rospy.Subscriber(synthesizer.instseg_topic, InstSegRes, synthesizer.instseg_callback)
    pub_cutpoint = rospy.Publisher(synthesizer.result_topic, CutPoint, queue_size=1)
    pub_image_pointcloud = rospy.Publisher(synthesizer.image_pc2_topic, PointCloud2, queue_size=1)
    pub_polynomial_pointcloud = rospy.Publisher(synthesizer.polynomial_pc2_topic, PointCloud2, queue_size=1)
    pub_tomato_center_pointcloud = rospy.Publisher(synthesizer.tomato_center_pc2_topic, PointCloud2, queue_size=1)
    pub_pedicel_end_pointcloud = rospy.Publisher(synthesizer.pedicel_end_pc2_topic, PointCloud2, queue_size=1)
    if synthesizer.publish_filtered_instseg_image:
        pub_instseg_im_filtered = rospy.Publisher(synthesizer.instseg_im_filtered_topic, Image, queue_size=1)
#    r = rospy.Rate(10)
    br = tf.TransformBroadcaster()
    exit_code = ExitCode()
    while not rospy.is_shutdown():
        if synthesizer.flg == "1":
            if synthesizer.instseg_finished and synthesizer.sm_finished and synthesizer.im_array is not None:
                rospy.loginfo("Start synthesis.")
                synthesizer.main_callback()

                pub_image_pointcloud.publish(synthesizer.image_point_cloud)

                if synthesizer.publish_filtered_instseg_image:
                    pub_instseg_im_filtered.publish(synthesizer.instseg_im_filtered)

                if synthesizer.tf_computed:
        #        if synthesizer.result_msg is not None and synthesizer.flg=="1":
        #            rospy.loginfo(model.result_msg)
                    pub_cutpoint.publish(synthesizer.result_msg)
                    pub_image_pointcloud.publish(synthesizer.image_point_cloud)
                    pub_polynomial_pointcloud.publish(synthesizer.polynomial_point_cloud)
                    pub_tomato_center_pointcloud.publish(synthesizer.tomato_center_point_cloud)
                    pub_pedicel_end_pointcloud.publish(synthesizer.pedicel_end_point_cloud)
        #            r.sleep()
                    br.sendTransform(synthesizer.translation, synthesizer.quaternion, rospy.Time.now(), "/tomato_pedicel", "/zedm_left_camera_optical_frame")
                    exit_code.exit_code = ExitCode.CODE_PEDICEL_DETECTION_SUCCESS
                    synthesizer.exit_code_pub.publish(exit_code)
                else:
                    rospy.loginfo("pedicel is not detected.")
                    exit_code.exit_code = ExitCode.CODE_PEDICEL_DETECTION_FAILED
                    synthesizer.exit_code_pub.publish(exit_code)
            
                synthesizer.flg = "0"
                synthesizer.instseg_finished = False
                synthesizer.sm_finished = False
                synthesizer.tf_computed = False



#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()







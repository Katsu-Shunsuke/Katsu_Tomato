#!/usr/bin/env python

import os
import sys
import rospy
import tf
import warnings

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
from visualization_msgs.msg import Marker

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils import (rosarray_to_numpy, stereo_reconstruction, polynomial_derivative, generate_pc2_message, filter_instseg, visualize_output,
curve_fitting, pca, calc_mean_point, visualize_eigen_vectors)
from pedicel_quaternion import calc_pedicel_quaternion, calc_tomato_center, remove_outliers, calc_all_pedicel_quaternions, remove_outliers, select_mode_and_cutpoint
from synthesis.msg import InstSegRes, CutPoint, ExitCode # need to edit CMakeLists.txt and package.xml

class Synthesis:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedm/zed_node/left/image_rect_color_synchronized" # left image because disparity map is on left image.
#        self.flg_topic = "synthesis_flg"
        self.result_topic = "synthesis_cutpoint_output"
        self.depth_topic = "aanet_depth_array_output"
        self.instseg_topic = "instance_segmentation_array_output"
        self.image_pc2_topic = "synthesis_image_pc2_output"
        self.polynomial_pc2_topic = "synthesis_polynomial_pc2_output"
        self.tomato_center_pc2_topic = "synthesis_tomato_center_pc2_output"
        self.pedicel_end_pc2_topic = "synthesis_pedicel_end_pc2_output"
        self.instseg_im_filtered_topic = "instance_segmentation_filtered_image_output"
        self.pca_eigen_raw_topic = "pca_eigen_raw"
        self.pca_eigen_corrected_topic = "pca_eigen_corrected"
        self.pedicel_xyz_topic = "pedicel_xyz"
        self.pedicel_end_minmax_xyz_topic = "pedicel_end_minmax_xyz"
        self.exit_code_pub = rospy.Publisher("large_tomato/exit_code", ExitCode, queue_size=1)
        self.publish_filtered_instseg_image = False
        self.calc_all_modes = True
        # output of callback methods
        self.depth = None
        self.xyz = None
        self.im_array = None
        self.result = None
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
        self.eigen_computed = False
        self.quaternions_using_all_modes = None
        self.pca_eigen_raw = None
        self.pca_eigen_corrected = None
        self.pedicel_xyz = None
        self.pedicel_end_minmax_xyz = None

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

    def main_callback(self):
        print("running main callback")
        bbox_top = rospy.get_param("bbox_top", 1)
        ripeness_threshold = rospy.get_param("ripeness_threshold", 10)
        ripeness_percentile = rospy.get_param("ripeness_percentile", 0.25)
        pedicel_calc_mode = rospy.get_param("pedicel_calc_mode", 4)
        which_pedicel = rospy.get_param("which_pedicel", 0)
        max_dist = rospy.get_param("max_dist", 999)
        
        print("\nbbox_top: {}\nripeness_threshold: {}\nripeness_percentile: {}\npedicel_calc_mode: {}\nwhich_pedicel: {}\n".format(
            bbox_top, ripeness_threshold, ripeness_percentile, pedicel_calc_mode, which_pedicel))

        # publish test pointcloud2 message
        self.image_point_cloud = generate_pc2_message(self.xyz, self.im_array)

        # sort pedicels
        n_pedicels = int(len(self.mask_pedicel))
        min_y = [np.mean(i[:,0]) for i in self.mask_pedicel]
        mask_pedicel_sorted = [i for _, i in sorted(zip(min_y, self.mask_pedicel))] # pedicels sorted from small y-values (vertically higher) first
        print("n_pedicels:", n_pedicels, "\n")

        if n_pedicels != 0:
            if which_pedicel > n_pedicels - 1:
                warnings.warn("which_pedicel >= n_pedicels-1; index out of range so index 0 is used instead.")
                which_pedicel = 0
#                which_pedicel = n_pedicels - 1
#                raise Exception("which_pedicel must be an integer in the range 0, ..., n_pedicels-1")
            mask_pedicel_sorted = mask_pedicel_sorted[which_pedicel:]

        # if mask_pedicel_sorted is empty then this loop is skipped. 
        for this_pedicel in mask_pedicel_sorted:
            # choose a pedicel (cannot loop for every pedicel in the image because cog can change)
            x = this_pedicel[:, 1].astype("int") # actually, better to send msg as uint32
            y = this_pedicel[:, 0].astype("int")

            # ==============================
            # Make changes from here on
            pedicel_xyz = self.xyz[y, x, :]
#            within_stds = remove_outliers(pedicel_xyz[:,2], max_deviations=1)
            within_stds = remove_outliers(np.linalg.norm(pedicel_xyz, axis=1), max_deviations=0.7)
            pedicel_xyz = pedicel_xyz[within_stds, :] # remove incorrect points at object boundary
            this_pedicel = this_pedicel[within_stds, :]

            # perform pca. M is matrix of eigen vectors, eigen are eigen vector point clouds for visualization
            pedicel_xyz_mean = calc_mean_point(pedicel_xyz)
            pedicel_dist = np.linalg.norm(pedicel_xyz_mean)
            print("pedicel_dist:", pedicel_dist, "[mm]")
            if pedicel_dist < max_dist:
                M = pca(pedicel_xyz)
                eigen = visualize_eigen_vectors(pedicel_xyz_mean, M)
                self.pca_eigen_raw = generate_pc2_message(
                    eigen,
                    np.tile(np.array([0, 255, 255]), (len(eigen),1)),
                    sampling_prop=1
                )
                self.pedicel_xyz = generate_pc2_message(
                    pedicel_xyz,
                    np.tile(np.array([255, 0, 255]), (len(pedicel_xyz),1))
                )
    
                # find the two endpoints of pedicel along direction with largest eigen value
                pedicel_xyz_transformed = pedicel_xyz @ M.T
                i_max = np.argmax(pedicel_xyz_transformed[:, 0])
                i_min = np.argmin(pedicel_xyz_transformed[:, 0])
                pedicel_end_max = pedicel_xyz_transformed[i_max, :]
                pedicel_end_min = pedicel_xyz_transformed[i_min, :]
                pedicel_end_max_2d = this_pedicel[i_max, :]
                pedicel_end_min_2d = this_pedicel[i_min, :]
#                plt.figure()
#                plt.imshow(self.im_array)
#                plt.plot(*pedicel_end_max_2d[::-1], "yo", ms=1)
#                plt.plot(*pedicel_end_min_2d[::-1], "yo", ms=1)
#                plt.savefig("pedicel_ends.png")
    
                self.pedicel_end_minmax_xyz = generate_pc2_message(
                    np.vstack((pedicel_end_max, pedicel_end_min)) @ np.linalg.inv(M.T),
                    np.tile(np.array([255, 0, 255]), (2, 1)),
                    sampling_prop=1
                )
    
                self.eigen_computed = True
    
                # search if there is a sepal attached to either of the pedicel ends
                d_search = 40 # in pixels
                intersect_end_max = []
                intersect_end_min = []
                intersect_end_max_sepal = []
                intersect_end_min_sepal = []
                for this_sepal in self.mask_sepal:
                    sepal_center = np.mean(this_sepal, axis=0)
                    d_max_end = np.linalg.norm(sepal_center - pedicel_end_max_2d)
                    d_min_end = np.linalg.norm(sepal_center - pedicel_end_min_2d)
                    if d_max_end < d_search:
                        intersect_end_max.append(d_max_end)
                        intersect_end_max_sepal.append(sepal_center)
                    if d_min_end < d_search:
                        intersect_end_min.append(d_min_end)
                        intersect_end_min_sepal.append(sepal_center)
                
                print("intersect_end_max:", intersect_end_max)
                print("intersect_end_min:", intersect_end_min)
                if len(intersect_end_max) > 0 and len(intersect_end_min) == 0:
                    pedicel_end_with_sepal_ij = pedicel_end_max_2d
                    sepal_center_pedicel_end = intersect_end_max_sepal[np.argmin(intersect_end_max)]
                    if len(intersect_end_max) > 1:
                        warnings.warn("More than 1 overlapping sepal at pedicel_end_max")
                elif len(intersect_end_min) > 0 and len(intersect_end_max) == 0:
                    pedicel_end_with_sepal_ij = pedicel_end_min_2d
                    sepal_center_pedicel_end = intersect_end_min_sepal[np.argmin(intersect_end_min)]
                    if len(intersect_end_min) > 1:
                        warnings.warn("More than 1 overlapping sepal at pedicel_end_min")
                elif len(intersect_end_min) > 0 and len(intersect_end_max) > 0:
                    pedicel_end_with_sepal_ij = pedicel_end_max_2d if min(intersect_end_max) < min(intersect_end_min) else pedicel_end_min_2d
                    sepal_center_pedicel_end = intersect_end_max_sepal[np.argmin(intersect_end_max)] if (
                        min(intersect_end_max) < min(intersect_end_min)
                    ) else intersect_end_min_sepal[np.argmin(intersect_end_min)]
                    warnings.warn("Both pedicel ends intersect with a sepal.")
                else: # both equal zero
                    pedicel_end_with_sepal_ij = None
                    sepal_center_pedicel_end = None

                print("pedicel_end_with_sepal_ij:", pedicel_end_with_sepal_ij)
                print("sepal_center_pedicel_end:", sepal_center_pedicel_end)
                    
                if pedicel_end_with_sepal_ij is not None and sepal_center_pedicel_end is not None:
                    # find if any tomato overlaps
                    y_end, x_end = pedicel_end_with_sepal_ij.astype("int")
                    y_sepal, x_sepal = sepal_center_pedicel_end.astype("int")
                    overlapping_tomatoes = []
                    xy_centers = []
                    for j, this_tomato in enumerate(self.bbox_tomato):
                        x_min, y_min, x_max, y_max = this_tomato[:4]
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        if (x_sepal > x_min and x_sepal < x_max) and (y_sepal > y_min and y_sepal < bbox_top * (y_max - y_min) + y_min):
                            overlapping_tomatoes.append(j)
                            xy_centers.append(np.array([y_center, x_center]))
                    
                    dists = []
                    if len(overlapping_tomatoes) > 1:
                        for xy_center in xy_centers:
                            dist = np.linalg.norm(xy_center - sepal_center_pedicel_end)
                            #dist = np.abs(x_center - x_sepal)
                            dists.append(dist)
                        j_final = overlapping_tomatoes[np.argmin(dists)]
                    elif len(overlapping_tomatoes) == 1:
                        j_final = overlapping_tomatoes[0]
                    else: # zero
                        j_final = None
        
                    print("\noverlapping_tomatoes:", overlapping_tomatoes)
                    print("j_final:", j_final)
                    print("dists:", dists)
        
                    if j_final is not None:
                        mask_indices = self.mask_tomato[j_final].astype(int)
                        # probably unnecessary to reduce if just using rgb info
                        tomato_pixels = self.im_array[mask_indices[:, 0], mask_indices[:, 1]].astype(np.float64) # should be nx3, also must be float because uint8 causes overflow
                        rgb_not_zero = np.sum(tomato_pixels, axis=1).astype("bool")
                        tomato_pixels = tomato_pixels[rgb_not_zero, :]
                        r, g, b = tomato_pixels[:, 0], tomato_pixels[:, 1], tomato_pixels[:, 2]
                        ripeness = np.sort((r - g) / (r + g + b))
                        lower_index = int(ripeness_percentile * len(ripeness))
                        upper_index = int((1 - ripeness_percentile) * len(ripeness))
                        ripeness = np.mean(ripeness[lower_index: upper_index])
                        print("ripeness:", ripeness)
                        if ripeness < ripeness_threshold:
                            # send this info to the manipulator  
                            self.this_pedicel = this_pedicel
        
                            # calculate tomato center
                            tomato_xyz = self.xyz[mask_indices[:, 0], mask_indices[:, 1], :] # should be nx3
                            tomato_center, tomato_r = calc_tomato_center(tomato_xyz)
                            print("tomato_dia:", tomato_r * 2 * 0.001, "[m]")
                            self.tomato_center_point_cloud = generate_pc2_message(tomato_center, np.array([0, 255, 255]), sampling_prop=1)
        
                            # curve fitting
                            # first figure out if largest eigen value vector is pointing in which direction
                            pedicel_end_minmax = self.xyz[y_end, x_end, :]
                            if np.linalg.norm(pedicel_xyz_mean + M[0, :] - pedicel_end_minmax) > np.linalg.norm(pedicel_xyz_mean - pedicel_end_minmax):
                                M[0, :] *= -1
                            if (np.cross(M[0, :], M[1, :]) / M[2, :] < 0).all():
                                M[2, :] *= -1
                            eigen_corrected = visualize_eigen_vectors(pedicel_xyz_mean, M) # visualize before the axes get rearranged
                            M[[0, 1, 2], :] = M[[2, 0, 1], :]
                            self.pca_eigen_corrected = generate_pc2_message(
                                eigen_corrected,
                                np.tile(np.array([0, 255, 255]), (len(eigen_corrected),1)),
                                sampling_prop=1
                            )
                            x_glob, y_glob, z_glob = (pedicel_xyz @ M.T).T
                            cut_points, dir_vectors, pedicel_end, pedicel_start, curve, curve_length = curve_fitting(x_glob, y_glob, z_glob, mode="polynomial")
                            cut_points, dir_vectors, pedicel_end, pedicel_start, curve = [ting @ np.linalg.inv(M.T) for ting in [cut_points, dir_vectors, pedicel_end, pedicel_start, curve]]

                            mode_preference, cut_prop_i = select_mode_and_cutpoint(
                                cutpoints=cut_points,
                                tomato_center=tomato_center,
                                tomato_r=tomato_r,
                                pedicel_start=pedicel_start,
                                curve_length=curve_length
                            )
                            print("mode_preference:", mode_preference)
                            print("cut_prop_i:", cut_prop_i)
                            cut_point = cut_points[cut_prop_i, :]
                            dir_vector = dir_vectors[cut_prop_i, :]

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
                            if self.calc_all_modes:
                                self.quaternions_using_all_modes = calc_all_pedicel_quaternions(
                                    vec1, vec2,
                                    cutpoint=cut_point,
                                    tomato_center=tomato_center,
                                    pedicel_end=pedicel_end
                                ) 
                            else:
                                self.quaternion = calc_pedicel_quaternion(
                                    vec1, vec2,
                                    cutpoint=cut_point,
                                    tomato_center=tomato_center,
                                    pedicel_end=pedicel_end,
                                    mode=pedicel_calc_mode
                                )
                            self.translation = tuple(cut_point * 10**(-3)) # mm to m
            
                            # visualize curve-fitted polynomial
                            rgb = np.tile(np.array([255,0,0]), (len(curve), 1))
                            self.polynomial_point_cloud = generate_pc2_message(curve, rgb, sampling_prop=1)
            
                            self.tf_computed = True
                            break
            print("\nPedicel has no tomato attached to it. Moving onto the next pedicel.\n")
    
def main():
    rospy.init_node("synthesis", anonymous=True)
    synthesizer = Synthesis()
    rospy.Subscriber(synthesizer.im_topic, Image, synthesizer.im_callback)
    rospy.Subscriber(synthesizer.depth_topic, Float32MultiArray, synthesizer.depth_callback)
    rospy.Subscriber(synthesizer.instseg_topic, InstSegRes, synthesizer.instseg_callback)
    pub_cutpoint = rospy.Publisher(synthesizer.result_topic, CutPoint, queue_size=1)
    pub_image_pointcloud = rospy.Publisher(synthesizer.image_pc2_topic, PointCloud2, queue_size=1)
    pub_polynomial_pointcloud = rospy.Publisher(synthesizer.polynomial_pc2_topic, PointCloud2, queue_size=1)
    pub_tomato_center_pointcloud = rospy.Publisher(synthesizer.tomato_center_pc2_topic, PointCloud2, queue_size=1)
    pub_pedicel_end_pointcloud = rospy.Publisher(synthesizer.pedicel_end_pc2_topic, PointCloud2, queue_size=1)
    pub_pca_eigen_raw = rospy.Publisher(synthesizer.pca_eigen_raw_topic, PointCloud2, queue_size=1)
    pub_pca_eigen_corrected = rospy.Publisher(synthesizer.pca_eigen_corrected_topic, PointCloud2, queue_size=1)
    pub_pedicel_xyz = rospy.Publisher(synthesizer.pedicel_xyz_topic, PointCloud2, queue_size=1)
    pub_pedicel_end_minmax_xyz = rospy.Publisher(synthesizer.pedicel_end_minmax_xyz_topic, PointCloud2, queue_size=1)
    if synthesizer.publish_filtered_instseg_image:
        pub_instseg_im_filtered = rospy.Publisher(synthesizer.instseg_im_filtered_topic, Image, queue_size=1)
#    r = rospy.Rate(10)
    br = tf.TransformBroadcaster()
    exit_code = ExitCode()
    while not rospy.is_shutdown():
        if synthesizer.instseg_finished and synthesizer.sm_finished and synthesizer.im_array is not None:
            rospy.loginfo("Start synthesis.")
            synthesizer.main_callback()

            pub_image_pointcloud.publish(synthesizer.image_point_cloud)

            if synthesizer.publish_filtered_instseg_image:
                pub_instseg_im_filtered.publish(synthesizer.instseg_im_filtered)

            if synthesizer.eigen_computed:
                pub_pca_eigen_raw.publish(synthesizer.pca_eigen_raw)
                pub_pedicel_xyz.publish(synthesizer.pedicel_xyz)
                pub_pedicel_end_minmax_xyz.publish(synthesizer.pedicel_end_minmax_xyz)

            if synthesizer.tf_computed:
    #        if synthesizer.result_msg is not None and synthesizer.flg=="1":
    #            rospy.loginfo(model.result_msg)
                pub_cutpoint.publish(synthesizer.result_msg)
                pub_image_pointcloud.publish(synthesizer.image_point_cloud)
                pub_polynomial_pointcloud.publish(synthesizer.polynomial_point_cloud)
                pub_tomato_center_pointcloud.publish(synthesizer.tomato_center_point_cloud)
                pub_pedicel_end_pointcloud.publish(synthesizer.pedicel_end_point_cloud)
                pub_pca_eigen_corrected.publish(synthesizer.pca_eigen_corrected)
    #            r.sleep()
                if synthesizer.calc_all_modes:
                    for i, quaternion in enumerate(synthesizer.quaternions_using_all_modes):
                        br.sendTransform(synthesizer.translation, quaternion, rospy.Time.now(), "/tomato_pedicel_mode_" + str(i), "/zedm_left_camera_optical_frame")
                else:
                    br.sendTransform(synthesizer.translation, synthesizer.quaternion, rospy.Time.now(), "/tomato_pedicel", "/zedm_left_camera_optical_frame")
                exit_code.exit_code = ExitCode.CODE_PEDICEL_DETECTION_SUCCESS
                synthesizer.exit_code_pub.publish(exit_code)
            else:
                rospy.loginfo("pedicel is not detected.")
                exit_code.exit_code = ExitCode.CODE_PEDICEL_DETECTION_FAILED
                synthesizer.exit_code_pub.publish(exit_code)
        
            synthesizer.im_array = None
            synthesizer.instseg_finished = False
            synthesizer.sm_finished = False
            synthesizer.tf_computed = False
            synthesizer.eigen_computed = False
            print("====================================================")



#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()







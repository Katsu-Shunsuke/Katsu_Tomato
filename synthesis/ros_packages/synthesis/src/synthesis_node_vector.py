#!/user/bin/env python
#test
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

from utils import rosarray_to_numpy, stereo_reconstruction, polynomial_derivative, generate_pc2_message, filter_instseg, visualize_output, curve_fitting
from pedicel_quaternion import calc_pedicel_quaternion, remove_outliers, calc_all_pedicel_quaternions
from synthesis.msg import InstSegRes, CutPoint, ExitCode # need to edit CMakeLists.txt and package.xml
from functions import mask_to_xyz, index_to_xyz, index_to_xyz_all, remove_outliers, calc_tomato_center,new_e, new_field, back_field, hand_box, fit_plane, twist,calc_modify_y, new_hand_arm_rotaion, Box_new_tidy,detect_interference, surface_pedicel, calc_sepal_center
from calc import calculate, calculate2
from relation_list import pedicel_sepal, sepal_tomato, pedicel_tomato, check_relation_list, blank_list
from make_model import calc_g, calc_pedicel_end, calc_pedicel_end2, calc_delimination, generate_marker_message, straight_pedicel

class Synthesis:
    def __init__(self):
        # topics to subscribe and publish to
        self.im_topic = "/zedm/zed_node/left/image_rect_color_synchronized" # left image because disparity map is on left image.
#        self.flg_topic = "synthesis_flg"
        self.result_topic = "synthesis_cutpoint_output"
        self.depth_topic = "aanet_depth_array_output"
        self.instseg_topic = "instance_segmentation_array_output"
        self.image_pc2_topic = "synthesis_image_pc2_output"
        
        self.insert_point_pc2_topic = "synthesis_insert_point_pc2_output"
        self.set_point_pc2_topic = "synthesis_set_point_pc2_output"
        self.set_point_tw_pc2_topic = "synthesis_set_point_tw_pc2_output"

        self.pedicel_pc2_topic = "synthesis_pedicel_pc2_output"
        self.tomato_center_pc2_topic = "synthesis_tomato_center_pc2_output"
        self.end_xyz_pc2_topic = "synthesis_end_xyz_pc2_output"
        self.start_xyz_pc2_topic = "synthesis_start_xyz_pc2_output"

        self.delimination_pc2_topic = "synthesis_delimination_pc2_output"

        self.fitted_sphere_topic = "fitted_sphere"

        self.box_pc2_topic = "synthesis_box_pc2_output"
        self.box_tw_pc2_topic = "synthesis_box_tw_pc2_output"

        self.instseg_im_filtered_topic = "instance_segmentation_filtered_image_output"
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

        self.tomato_center_point_cloud = None
        self.end_xyz_point_cloud = None
        self.start_xyz_point_cloud = None
        self.pedicel_point_cloud = None
        self.delimination_point_cloud = None

        self.box_point_cloud = None
        self.box_tw_point_cloud = None

        self.fitted_sphere = None

        self.image_point_cloud = None

        self.insert_point = None
        self.set_point = None
        self.set_point_tw = None

        self.insert_point_cloud = None
        self.set_point_cloud = None
        self.set_point_tw_cloud = None

        self.quaternion_insert = None
        self.quaternion_tw = None
 
        self.instseg_im_filtered = None
        self.instseg_finished = False
        self.sm_finished = False
        self.tf_computed = False
        self.quaternions_using_all_modes = None

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
        bbox_top = rospy.get_param("bbox_top", 0.6)
        ripeness_threshold = rospy.get_param("ripeness_threshold", 10)
        ripeness_percentile = rospy.get_param("ripeness_percentile", 0.25)
        pedicel_calc_mode = rospy.get_param("pedicel_calc_mode", 4)#収穫方法のパラメータ
        which_pedicel = rospy.get_param("which_pedicel", 0)
        which_tomato = rospy.get_param("which_tomato", 0)#どのトマトをとるか
        
        print("\nbbox_top: {}\nripeness_threshold: {}\nripeness_percentile: {}\npedicel_calc_mode: {}\nwhich_pedicel: {}\n".format(bbox_top,
            ripeness_threshold, ripeness_percentile, pedicel_calc_mode, which_pedicel))

        # publish test pointcloud2 message
        self.image_point_cloud = generate_pc2_message(self.xyz, self.im_array)#点群の生成

        # sort pedicels
        n_pedicels = int(len(self.mask_pedicel))
        size_p_mask = [len(i) for i in self.mask_pedicel]
        mask_pedicel_sorted = [i for _, i in sorted(zip(size_p_mask, self.mask_pedicel))][::-1] # pedicels sorted from small y-values (vertically higher) first
        print("n_pedicels:", n_pedicels, "\n")
        n_sepals = int(len(self.mask_sepal))
        size_s_mask = [np.min(i[:,0]) for i in self.mask_sepal]
        mask_sepal_sorted = [i for _, i in sorted(zip(size_s_mask, self.mask_sepal))]
        bbox_sepal_sorted = [i for _, i in sorted(zip(size_s_mask, self.bbox_sepal))]
        print("n_sepals:", n_sepals, "\n" )
        n_tomatoes = int(len(self.mask_tomato))
        size_t_mask = [np.min(i[:,0]) for i in self.mask_tomato]
        mask_tomato_sorted = [i for _, i in sorted(zip(size_t_mask, self.mask_tomato))]
        bbox_tomato_sorted = [i for _, i in sorted(zip(size_t_mask, self.bbox_tomato))]
        print("n_tomatoes:", n_tomatoes, "\n")


        pedicel_sepal_list = pedicel_sepal(mask_sepal_sorted, mask_pedicel_sorted, bbox_sepal_sorted)
        sepal_tomato_list = sepal_tomato(mask_tomato_sorted, mask_sepal_sorted, bbox_tomato_sorted)
        pedicel_tomato_list = pedicel_tomato(mask_tomato_sorted, mask_pedicel_sorted, bbox_tomato_sorted)
        
#        pedicel_tomato_list = []
#        for i in range(n_pedicels):
#            pedicel_tomato_list.append([])

#        for i in range(len(tomato_pedicel_list)):
#            if len(tomato_pedicel_list[i]) > 0 :
#                index = tomato_pedicel_list[i][0]
#                pedicel_tomato_list[index].append(i)

        print("pedicel_sepal_list" + str(pedicel_sepal_list))
        print("sepal_tomato_list" + str(sepal_tomato_list))
        print("pedicel_tomato_list" + str(pedicel_tomato_list) + "\n")

        list_sum = check_relation_list(pedicel_sepal_list, sepal_tomato_list, pedicel_tomato_list, n_tomatoes, n_sepals, n_pedicels)

        print("summerize")
        print(str(list_sum) + "\n")


        ########################################################
        #######    全てのデータのxyz(cut)  #####################
        ########################################################

        tomato_all = index_to_xyz_all(self.xyz, mask_tomato_sorted)
        pedicel_all = index_to_xyz_all(self.xyz, mask_pedicel_sorted)
        sepal_all = index_to_xyz_all(self.xyz, mask_sepal_sorted)
        
        tomato_all_cut = []
        pedicel_all_cut = []
        sepal_all_cut = []
        for i in range(len(tomato_all)):
            max_deviations = 0.5
            tomato_all_cut.append( tomato_all[i][remove_outliers(tomato_all[i][:,2], max_deviations),:] )

        for i in range(len(pedicel_all)):
            max_deviations = 0.05
            pedicel_all_cut.append( pedicel_all[i][remove_outliers(pedicel_all[i][:,2], max_deviations),:] )

        for i in range(len(sepal_all)):
            max_deviations = 0.05
            sepal_all_cut.append( sepal_all[i][remove_outliers(sepal_all[i][:,2], max_deviations),:] )

        center_end_dis = []
        tomato_center_all = blank_list(n_tomatoes)
        tomato_r_all = blank_list(n_tomatoes)
        sepal_center_all = blank_list(n_sepals)
        start_xyz_all = blank_list(n_pedicels)
        end_xyz_all = blank_list(n_pedicels)
        dela_xyz_all = blank_list(n_pedicels)

        for i in range(n_tomatoes):
            _, tomato_center_i, tomato_r_i = calc_tomato_center(tomato_all[i], 0.025)
            tomato_center_all[i] = tomato_center_i
            tomato_r_all[i] = tomato_r_i
        
        for i in range(n_sepals):
            sepal_center_all[i] = calc_sepal_center(sepal_all_cut[i])

        for i in range(len(list_sum)):
            t_index = list_sum[i][0]
            s_index = list_sum[i][1]
            p_index = list_sum[i][2]

            start_xyz_i = pedicel_all_cut[p_index][np.argmax(np.sum((pedicel_all_cut[p_index] - sepal_center_all[s_index])**2, axis=1))]
            start_xyz_all[p_index] = start_xyz_i

            end_xyz = sepal_center_all[s_index]
            end_xyz = calc_pedicel_end2(pedicel_all_cut[p_index], end_xyz)
            length = np.dot((sepal_center_all[s_index] - tomato_center_all[t_index]),(end_xyz-tomato_center_all[t_index]))/np.linalg.norm(end_xyz-tomato_center_all[t_index])
            end_xyz_i = (end_xyz - tomato_center_all[t_index])*(length)/np.linalg.norm(end_xyz - tomato_center_all[t_index]) + tomato_center_all[t_index]
            end_xyz_all[p_index] = end_xyz_i

            dela_xyz_all[p_index] = calc_delimination(pedicel_all_cut[p_index], start_xyz_i, end_xyz_i, tomato_center_all[t_index])


        #################################################
        #######    interference  ########################
        #################################################
#
#        interference_all = []
#        #eye_all = []
#        #eye_tw_all = []
#
        for i in range(len(list_sum)):

            t_index = list_sum[i][0]
            s_index = list_sum[i][1]
            p_index = list_sum[i][2]

            plane_v = tomato_center_all[t_index] - end_xyz_all[p_index]
            p_xyz = pedicel_all_cut[p_index]
            t_xyz = tomato_all_cut[t_index]
            start = start_xyz_all[p_index]
            end = end_xyz_all[p_index]
            P = new_e(plane_v)
            p_xyz_new = new_field(P, p_xyz)
            t_xyz_new = new_field(P, t_xyz)
            start_new = new_field(P, start)
            end_new       = new_field(P, end)
            coefs_xz_new = np.polyfit(p_xyz_new[:,0], p_xyz_new[:,2], deg=1)
            insert_new = np.array([1, 0, coefs_xz_new[0]])
            if np.dot(insert_new, start_new - end_new) < 0:
                insert_new = insert_new * -1 

            t_upper_new_y = t_xyz_new[np.argsort(t_xyz_new[:,1])][:int(len(t_xyz_new)*0.1)][:,1].mean()
    
#    
#            tomato_index  = t
#            
#            if len(tomato_pedicel_list[t])==1:
#                pedicel_index = tomato_pedicel_list[t][0]
#            elif len(tomato_pedicel_list[t]) > 1:
#                end_to_center_compare = []
#                for p in tomato_pedicel_list[t]:
#                    end_to_center_compare.append(np.linalg.norm(end_xyz_all[p][0]-tomato_center_all[t]))
#                pedicel_index = tomato_pedicel_list[t][np.argmin(end_to_center_compare)]
#                print("tomato "+ str(t) + " near  pedicel" + str(pedicel_index))
#                    
#                    
#            interference=[]
#            
#            print("tomato " + str(t))
#
#            _, _, _, eye, eye_tw, _, Box_new, _,  P_calc = calculate(tomato_all_cut[tomato_index],pedicel_all_cut[pedicel_index],self.xyz, end_xyz, start_xyz, 0.025)
#            
#            #eye_all.append(eye)
#            #eye_tw_all.append(eye_tw)
#            
#            for i in range(len(self.mask_tomato)):
#
#                if i != tomato_index:
#                    tomato_new = new_field(P_calc, tomato_all_cut[i])
#                    R_box = Box_new_tidy(Box_new)
#                    num_inf = detect_interference(Box_new, tomato_new, R_box)
#                    inf_p = num_inf/len(tomato_new)*100
#                    if inf_p <= 5:
#                        print("not interference" + str(i) + " " + str(inf_p)+ "%")
#                        interference.append(False)
#                    else:
#                        print("detect interference" + str(i) + " " + str(inf_p)+ "%")
#                        interference.append(True)
#                else:
#                    interference.append(False)
#            interference_all.append(interference)
#            print("\n")

        ################################################
        ######      approach     ########################
        ################################################

        if len(list_sum) != 0:
            t_i_final = which_tomato
            if (list_sum[:,0] == t_i_final).any():
                
                list_approach = list_sum[np.where(list_sum[:,0]==t_i_final)][0]
                p_i_final = list_approach[2]
                s_i_final = list_approach[1]

                print("tomato_index : " + str(t_i_final))
                print("sepal_index : " + str(s_i_final))
                print("pedicel_index : " + str(p_i_final) + "\n")

                tomato_xyz = tomato_all_cut[t_i_final]
                tomato_center = tomato_center_all[t_i_final]
                tomato_r = tomato_r_all[t_i_final]
                self.tomato_center_point_cloud = generate_pc2_message(tomato_center, np.array([255,0,255]), sampling_prop=1)
                self.fitted_sphere = generate_marker_message(tomato_center, tomato_r*2)

                sepal_g = sepal_center_all[s_i_final]
                pedicel_xyz = pedicel_all_cut[p_i_final]
                start_xyz = start_xyz_all[p_i_final]
                #surface_pedicel_xyz = np.array(surface_pedicel(pedicel_xyz, level=0.5))
                
                end_xyz = end_xyz_all[p_i_final]
                delimination = dela_xyz_all[p_i_final]

                straight_pedicel(start_xyz, end_xyz, delimination)

                if delimination is not None:
                    self.delimination_point_cloud = generate_pc2_message(delimination,np.array([255,255,255]), sampling_prop=1) 
                    
#                   if np.array([interference_all[t_i_final]]).any():
#                       print("warning: may get caught on other tomatoes")
#                   else:
#                       print("no obstale!!")

                insert_point, set_point, set_point_tw, eye, eye_tw, Box, Box_new, Box_tw, P_calc = calculate2(t_i_final, tomato_center_all, tomato_r_all, end_xyz, start_xyz, pedicel_xyz, tomato_xyz)

                #insert_point, set_point, set_point_tw, eye, eye_tw, Box, Box_new, Box_tw, P_calc = calculate(tomato_xyz,pedicel_xyz,self.xyz,end_xyz,start_xyz,0.5)
                self.end_xyz_point_cloud = generate_pc2_message(end_xyz, np.array([255,0,255]), sampling_prop=1)

                self.start_xyz_point_cloud = generate_pc2_message(start_xyz,np.array([255,0,0]),sampling_prop=1)

                pedicel_color = np.zeros((len(pedicel_xyz),3), dtype=np.int16)
                self.pedicel_point_cloud = generate_pc2_message(pedicel_xyz, pedicel_color, sampling_prop=1)

                box_color = np.zeros((len(Box),3), dtype=np.int16)
                box_color_tw = np.zeros((len(Box_tw),3), dtype=np.int16)
                self.box_point_cloud = generate_pc2_message(Box, box_color, sampling_prop=1)
                self.box_tw_point_cloud = generate_pc2_message(Box_tw, box_color_tw, sampling_prop=1)

                self.insert_point = tuple(insert_point * 10**(-3))
                print("insert_point : " + str(insert_point))
                self.set_point = tuple(set_point * 10**(-3))
                self.set_point_tw = tuple(set_point_tw * 10**(-3))

                self.insert_point_cloud = generate_pc2_message(insert_point, np.array([255, 0, 255]), sampling_prop=1)
                self.set_point_cloud = generate_pc2_message(set_point, np.array([255, 0, 255]), sampling_prop=1)
                self.set_point_tw_cloud = generate_pc2_message(set_point_tw, np.array([255, 0, 255]), sampling_prop=1)


                self.quaternion_insert = tf.transformations.quaternion_from_matrix(eye)
                self.quaternion_tw = tf.transformations.quaternion_from_matrix(eye_tw)

                self.tf_computed = True
                
            else:
                print("select tomato is not found or not attached any pedicel or sepal")


def main():
    rospy.init_node("synthesis", anonymous=True)
    synthesizer = Synthesis()
    rospy.Subscriber(synthesizer.im_topic, Image, synthesizer.im_callback) 
    #/zedm/zed_node/left/image_rect_color_synchronizedをsubscribeしたら画像配列

    rospy.Subscriber(synthesizer.depth_topic, Float32MultiArray, synthesizer.depth_callback) 
    #aanet_depth_array_outputをsubscribeしたら深度マップ

    rospy.Subscriber(synthesizer.instseg_topic, InstSegRes, synthesizer.instseg_callback) 
    #instance_segmentation_array_outputとsubscribeしたらインスタンスセグメンテーション
    
    pub_image_pointcloud = rospy.Publisher(synthesizer.image_pc2_topic, PointCloud2, queue_size=1)
    pub_insert_point_pointcloud = rospy.Publisher(synthesizer.insert_point_pc2_topic, PointCloud2, queue_size=1)
    pub_set_point_pointcloud = rospy.Publisher(synthesizer.set_point_pc2_topic, PointCloud2, queue_size=1)
    pub_set_point_tw_pointcloud = rospy.Publisher(synthesizer.set_point_tw_pc2_topic, PointCloud2, queue_size=1)
    pub_tomato_center_pointcloud = rospy.Publisher(synthesizer.tomato_center_pc2_topic, PointCloud2, queue_size=1)
    pub_end_xyz_pointcloud = rospy.Publisher(synthesizer.end_xyz_pc2_topic, PointCloud2, queue_size=1)
    pub_box_pointcloud = rospy.Publisher(synthesizer.box_pc2_topic, PointCloud2, queue_size=1)
    pub_box_tw_pointcloud = rospy.Publisher(synthesizer.box_tw_pc2_topic, PointCloud2, queue_size=1)
    pub_pedicel_pointcloud = rospy.Publisher(synthesizer.pedicel_pc2_topic, PointCloud2, queue_size=1)
    pub_delimination_pointcloud = rospy.Publisher(synthesizer.delimination_pc2_topic, PointCloud2, queue_size=1)
    pub_start_xyz_pointcloud = rospy.Publisher(synthesizer.start_xyz_pc2_topic, PointCloud2, queue_size=1)
    pub_fitted_sphere = rospy.Publisher(synthesizer.fitted_sphere_topic, Marker, queue_size=1)
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

            if synthesizer.tf_computed:
    #        if synthesizer.result_msg is not None and synthesizer.flg=="1":
    #            rospy.loginfo(model.result_msg)
                pub_image_pointcloud.publish(synthesizer.image_point_cloud)
                pub_insert_point_pointcloud.publish(synthesizer.insert_point_cloud)
                pub_set_point_pointcloud.publish(synthesizer.set_point_cloud)
                pub_set_point_tw_pointcloud.publish(synthesizer.set_point_tw_cloud)
                pub_tomato_center_pointcloud.publish(synthesizer.tomato_center_point_cloud)
                pub_end_xyz_pointcloud.publish(synthesizer.end_xyz_point_cloud)
                pub_box_pointcloud.publish(synthesizer.box_point_cloud)
                pub_box_tw_pointcloud.publish(synthesizer.box_tw_point_cloud)
                pub_pedicel_pointcloud.publish(synthesizer.pedicel_point_cloud)
                if synthesizer.delimination_point_cloud is not None:
                    pub_delimination_pointcloud.publish(synthesizer.delimination_point_cloud)
                pub_start_xyz_pointcloud.publish(synthesizer.start_xyz_point_cloud)
                pub_fitted_sphere.publish(synthesizer.fitted_sphere)

                br.sendTransform(synthesizer.insert_point, synthesizer.quaternion_insert, rospy.Time.now(), "/insert_pedicel", "/zedm_left_camera_optical_frame")
                br.sendTransform(synthesizer.set_point, synthesizer.quaternion_insert, rospy.Time.now(), "/insert", "/zedm_left_camera_optical_frame")
                br.sendTransform(synthesizer.set_point_tw, synthesizer.quaternion_tw, rospy.Time.now(), "/twist", "/zedm_left_camera_optical_frame")
    #            r.sleep()
                
            else:
                rospy.loginfo("tomato or pedicel is not detected.")
                exit_code.exit_code = ExitCode.CODE_PEDICEL_DETECTION_FAILED
                synthesizer.exit_code_pub.publish(exit_code)
        
            synthesizer.im_array = None
            synthesizer.instseg_finished = False
            synthesizer.sm_finished = False
            synthesizer.tf_computed = False



#    if sm.flg == "1":
#        pub.publish(sm.depth)
#        sm.flg = "0"
    rospy.spin()

if __name__ == "__main__":
    main()

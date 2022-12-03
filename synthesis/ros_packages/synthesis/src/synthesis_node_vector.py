#!/user/bin/env python

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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils import rosarray_to_numpy, stereo_reconstruction, polynomial_derivative, generate_pc2_message, filter_instseg, visualize_output, curve_fitting
from pedicel_quaternion import calc_pedicel_quaternion, calc_tomato_center, remove_outliers, calc_all_pedicel_quaternions
from synthesis.msg import InstSegRes, CutPoint, ExitCode # need to edit CMakeLists.txt and package.xml
from functions import mask_to_xyz, index_to_xyz, index_to_xyz_all, remove_outliers, calc_tomato_center,new_e, new_field, back_field, hand_box, twist_hand, fit_plane, twist_x, twist_y,calc_modify_y, new_hand_arm_rotaion, Box_new_tidy,detect_interference
from calc import calculate

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
        bbox_top = rospy.get_param("bbox_top", 0.5)
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
        min_y = [np.mean(i[:,0]) for i in self.mask_pedicel]
        mask_pedicel_sorted = [i for _, i in sorted(zip(min_y, self.mask_pedicel))] # pedicels sorted from small y-values (vertically higher) first
        print("n_pedicels:", n_pedicels, "\n")

        tomato_pedicel_list = []
        for n in range(len(self.mask_tomato)):
            tomato_pedicel_list.append([])

        # if mask_pedicel_sorted is empty then this loop is skipped. 
        for pedicel_index, this_pedicel in enumerate(mask_pedicel_sorted):
            # choose a pedicel (cannot loop for every pedicel in the image because cog can change)
            x = this_pedicel[:,1].astype("int") # actually, better to send msg as uint32
            y = this_pedicel[:,0].astype("int")
            # obtain the end with smaller y value
            x_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,1] # index zero since there are probs multiple
            y_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,0] # index zero since there are probs multiple
            # if that end point is within a tomato bbox, obtain a small patch centered around the tomato mask
            overlapping_tomatoes = []#選択した小花柄がはいっているトマト
            xy_centers = []#そのトマトの中心
            for j, this_tomato in enumerate(self.bbox_tomato):#選択した小花柄がトマトのbboxに入っているか
#                x_min, y_min, w, h = this_tomato[:4]
#                x_max = x_min + w
#                y_max = y_min + h
#                x_center = x_min + w/2
#                y_center = y_min + h/2
                x_min, y_min, x_max, y_max = this_tomato[:4]
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                if (x_end > x_min and x_end < x_max) and (y_end > y_min and y_end < bbox_top * (y_max - y_min) + y_min):#デフォルト値0.5
                    overlapping_tomatoes.append(j)
                    xy_centers.append([x_center, y_center])
            
            dists = []#トマトと小花柄の下端との距離
            if len(overlapping_tomatoes) > 1:
                for xy_center in xy_centers:
                    dist = np.sqrt((xy_center[0] - x_end)**2 + (xy_center[1] - y_end)**2)
                    #dist = np.abs(x_center - x_end)
                    dists.append(dist)
                j_final = overlapping_tomatoes[dists.index(min(dists))]#一番距離が近いものが収穫するトマト
                tomato_pedicel_list[j_final].append(pedicel_index)
            elif len(overlapping_tomatoes) == 1:
                j_final = overlapping_tomatoes[0]
                tomato_pedicel_list[j_final].append(pedicel_index)
            else: # zero
                j_final = None

            print("overlapping_tomatoes", overlapping_tomatoes)
            print("j_final", j_final)
            print("dists", dists, "\n")
            print("tomato_pedicel_list", tomato_pedicel_list)
            print("\n")

        ##全てのデータのxyz(cut)
        tomato_all = index_to_xyz_all(self.xyz, self.mask_tomato)
        #pedicel_all = index_to_xyz_all(xyz, mask_pedicel)

        tomato_all_cut = []
        #pedicel_all_cut = []
        for i in range(len(tomato_all)):
            max_deviations = 0.025
            tomato_all_cut.append( tomato_all[i][remove_outliers(tomato_all[i][:,2], max_deviations),:] )

        #for i in range(len(pedicel_all)):
        #    max_deviations = 0.5
        #    pedicel_all_cut.append( pedicel_all[i][remove_outliers(pedicel_all[i][:,2], max_deviations),:] )



        interference_all = []
        #eye_all = []
        #eye_tw_all = []

        for t in range(len(self.mask_tomato)):
    
            tomato_index  = t
            pedicel_index = tomato_pedicel_list[t][which_pedicel]
            interference=[]
            
            print("tomato " + str(t))

            _, _, _, eye, eye_tw, Box_new, P_calc = calculate(tomato_index,pedicel_index,self.xyz, self.mask_tomato, self.mask_pedicel, 0.025)
            
            #eye_all.append(eye)
            #eye_tw_all.append(eye_tw)
            
            for i in range(len(self.mask_tomato)):

                if i != tomato_index:
                    tomato_new = new_field(P_calc, tomato_all_cut[i])
                    R_box = Box_new_tidy(Box_new)
                    num_inf = detect_interference(Box_new, tomato_new, R_box)
                    inf_p = num_inf/len(tomato_new)*100
                    if inf_p <= 5:
                        print("not interference" + str(i) + " " + str(inf_p)+ "%")
                        interference.append(False)
                    else:
                        print("detect interference" + str(i) + " " + str(inf_p)+ "%")
                        interference.append(True)
                else:
                    interference.append(False)
            interference_all.append(interference)
            print("\n")

        t_i_final = which_tomato
        p_i_final = tomato_pedicel_list[which_tomato][which_pedicel]
        insert_point, set_point, set_point_tw, eye, eye_tw, Box_new, P_calc = calculate(t_i_final,p_i_final,self.xyz, self.mask_tomato, self.mask_pedicel, 0.025)
        
        self.insert_point = tuple(insert_point * 10**(-3))
        self.set_point = tuple(set_point * 10**(-3))
        self.set_point_tw = tuple(set_point_tw * 10**(-3))

        self.insert_point_cloud = generate_pc2_message(insert_point, np.array([255, 0, 255]), sampling_prop=1)
        self.set_point_cloud = generate_pc2_message(set_point, np.array([255, 0, 255]), sampling_prop=1)
        self.set_point_tw_cloud = generate_pc2_message(set_point_tw, np.array([255, 0, 255]), sampling_prop=1)


        self.quaternion_insert = eye
        self.quaternion_tw = eye_tw

        self.tf_computed = True



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

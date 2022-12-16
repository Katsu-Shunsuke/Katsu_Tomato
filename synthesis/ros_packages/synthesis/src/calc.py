#!/user/bin/env python

import numpy as np
import math
import rospy

from functions import mask_to_xyz, index_to_xyz, index_to_xyz_all, remove_outliers, calc_tomato_center,new_e, new_field, back_field, hand_box, fit_plane, twist,calc_modify_y, new_hand_arm_rotaion, Box_new_tidy,detect_interference, surface_pedicel, calc_pedicel_end
from utils import curve_fitting

def calculate(tomato_index, pedicel_index, xyz, mask_tomato, mask_pedicel, max_deviations, visualize=False, ax=False):
    
    #トマトと小花柄のxyzを取得
    tomato_xyz = index_to_xyz(tomato_index, xyz, mask_tomato)
    pedicel_xyz_pre = index_to_xyz(pedicel_index, xyz, mask_pedicel)
    #球体フィッティング・トマトの外れ値除外　#小花柄の外れ値除外
    tomato_cut, center, r = calc_tomato_center(tomato_xyz, max_deviations)
    pedicel_xyz = pedicel_xyz_pre[remove_outliers(pedicel_xyz_pre[:,2], max_deviations*2),:]
    pedicel_sf = surface_pedicel(pedicel_xyz)

    #小花柄のトマト側-end #小花柄の茎側-start
    #_,_, end_xyz, _ = curve_fitting(pedicel_xyz[:,0], pedicel_xyz[:,1], pedicel_xyz[:,2], mode="polynomial", tomato_center=center, tomato_r=r)
    start_xyz = pedicel_sf[np.argmax(np.sum((pedicel_sf - center)**2, axis=1))]
    end_xyz = calc_pedicel_end(pedicel_xyz, center) #,start_xyz)
    
    #トマトの垂直ベクトルの求め方
    #1: トマト中心と小花柄endを結んだベクトル
    #2: トマト上面（小花柄endから近いトマトの点群の平面近似-法線ベクトル
    tomato_tilt_mode = 1
    if tomato_tilt_mode == 1:
        plane_v =  center - end_xyz
    elif tomato_tilt_mode == 2:   
        tomato_dis = np.sqrt(np.sum((tomato_xyz - end_xyz)**2, axis=1))
        tomato_upper = tomato_xyz[np.where(tomato_dis < 40)]
        plane_v = fit_plane(tomato_upper)

    #トマトが垂直になるように座標変換
    P = new_e(plane_v)
    p_xyz_new = new_field(P, pedicel_xyz)
    t_xyz_new = new_field(P, tomato_xyz)
    end_xyz_new = new_field(P, end_xyz)
    start_xyz_new = new_field(P, start_xyz)
    
    tomato_upper_new_y = t_xyz_new[np.argsort(t_xyz_new[:,1])][:int(len(t_xyz_new)*0.1)][:,1].mean()
    tomato_delamination_new = np.array([end_xyz_new[0], tomato_upper_new_y - 14, end_xyz_new[2]])
    touch_point_news = p_xyz_new[np.where((p_xyz_new[:,1] < tomato_upper_new_y - 14) & (p_xyz_new[:,1] > tomato_upper_new_y -15))]
    
    coefs_xz_new = np.polyfit(p_xyz_new[:,0], p_xyz_new[:,2], deg=1)
    insert_new = np.array([1, 0, coefs_xz_new[0]]) 

    if np.dot((start_xyz_new - end_xyz_new), insert_new) < 0:
        insert_new = insert_new * -1

    Box_new = hand_box(tomato_upper_new_y, end_xyz_new, insert_new)
    

    #欲しかった情報
    #tomato_delamination: 離層の位置（トマト上面から5mm）
    #insert: 挿入ベクトル
    #Box: ハンドの位置
    tomato_delamination = back_field(P, tomato_delamination_new)
    insert = back_field(P, insert_new)
    Box = back_field(P, Box_new)
                      
    #向きの調整
    if np.dot((start_xyz-end_xyz), insert) < 0:
        insert = insert * -1
    
    #正規化
    vec_y = - plane_v / np.linalg.norm(plane_v)
    vec_x = np.cross( - plane_v, insert) / np.linalg.norm(np.cross( - plane_v, insert))
    vec_z = insert / np.linalg.norm(insert)
    
    ####
    #姿勢の修正←起動生成できるように
    vec_x_new = vec_x
    vec_y_new = vec_y
    vec_z_new = vec_z
    
    new_hand_upper_thick = 5
    interval = 203.01
    #insert_point = end_xyz + vec_y * new_hand_upper_thick
    insert_point = tomato_delamination
    
    #なるべく正面からアプローチしたいから許容範囲(0 deg ~ 60? deg)で修正
#    front_limit = 45 #deg
#    if vec_z_new[2] /np.linalg.norm(np.array([vec_z_new[0], vec_z_new[2]])) < np.sin( front_limit * math.pi /180):
#        
#        theta_mod_x = front_limit - np.arcsin( vec_z_new[2] /np.linalg.norm(np.array([vec_z_new[0], vec_z_new[2]])) )
#        
#        if vec_z_new[0] > 0:#小花柄の向きで正負決めるべきでは？
#            theta_mod_x_n = calc_modify_y(vec_y_new, vec_z_new, theta_mod_x)
#            print("modify v : " + str(theta_mod_x) + " →  " + str(theta_mod_x_n))
#            
#        else:
#            theta_mod_x_n = -1 * calc_modify_y(vec_y_new, vec_z_new, - theta_mod_x)
#            print("modify v : " + str(- theta_mod_x) + " →  " + str(theta_mod_x_n))
#        R = twist(vec_y_new, theta_mod_x_n)
#        vec_x_new = np.dot(R, vec_x_new.T).T
#        vec_y_new = np.dot(R, vec_y_new.T).T
#        vec_z_new = np.dot(R, vec_z_new.T).T
#        #vec_x_new,vec_y_new,vec_z_new,R_y = twist_y(vec_x_new,vec_y_new,vec_z_new,theta_mod_x_n)
#        Box = np.dot(R, (Box - insert_point).T).T + insert_point

        
#    #なるべく水平にアプローチしたいから許容範囲で修正
#    horizon_limit = 60 #deg
#    if abs ( (vec_z_new[0] ** 2 + vec_z_new[2] ** 2) ** 0.5 / np.linalg.norm(vec_z_new) ) < np.cos(60 * math.pi / 180):
#        theta_mod_h = np.arccos(vec_z_new[0] ** 2 + vec_z_new[2] ** 2) ** 0.5 / np.linalg.norm(vec_z_new) - 60
#        if vec_z_new[1] > 0:
#            R = twist(vec_x_new, theta_mod_h)
#        else:
#            R = twist(vec_x_new, - theta_mod_h)
#        vec_y_new = np.dot(R, vec_y_new.T).T
#        vec_z_new = np.dot(R, vec_z_new.T).T
#        Box = np.dot(R, (Box - insert_point).T).T + insert_point
#        print("modify h : " + str(theta_mod_h))
#        
    insert_h_deg = np.arcsin( vec_z_new[1] / np.linalg.norm(vec_z_new)) * 180 / math.pi
    insert_v_deg = np.arcsin( vec_z_new[2] /np.linalg.norm(np.array([vec_z_new[0], vec_z_new[2]])) ) * 180 / math.pi
    print("insert_v_deg(int) : " + str(int(insert_v_deg)))
    print("insert_h_deg(int) : " + str(int(insert_h_deg)))
    
    
    #ロボットの目標位置
    set_point = insert_point - vec_z_new * interval
    
    #ローラーの押し込み距離計算
    touch_points = back_field(P, touch_point_news)
    if len(touch_points) > 0:
        dis_to_touch = np.min(np.linalg.norm(touch_points - set_point, axis=1))
        print("dis_to_touch : " + str(dis_to_touch))
    else:
        print("warning air shot")
    
    #print("tomato_xyz")
    #print(tomato_xyz.shape)
    #print("tomato_cut")
    #print(tomato_cut.shape)
    #print("set_point")
    #print(set_point)
    #print("insert_point")
    #print(insert_point)
    #print("半径")
    #print(r)
    
    
    ### ひねり動作 ###
    harvest_mode = rospy.get_param("harvest_mode", 0)
    harvest_deg = rospy.get_param("harvest_deg", 45)
    if harvest_mode == 0:
        R_harvest = twist(vec_x, harvest_deg)
    else:
        R_harvest = twist(vec_x_new, harvest_deg)

    vec_x_tw = np.dot(R_harvest, vec_x_new.T)
    vec_y_tw = np.dot(R_harvest, vec_y_new.T)
    vec_z_tw = np.dot(R_harvest, vec_z_new.T)
    set_point_tw = np.dot(R_harvest, (set_point - insert_point).T).T + insert_point
    Box_tw = np.dot(R_harvest, (Box-insert_point).T).T + insert_point


    ### カメラで認識するときと収穫するときは奥行き方向に90(deg)回転
    vec_x_final, vec_y_final, vec_z_final = new_hand_arm_rotaion( - vec_x_new, - vec_y_new, vec_z_new)
    vec_x_tw_final, vec_y_tw_final, vec_z_tw_final = new_hand_arm_rotaion( - vec_x_tw, - vec_y_tw, vec_z_tw)
        
    #set_point_tw  = insert_point - vec_z_tw * interval
    
    insert_vector_mode = rospy.get_param("insert_vector_mode", 0)

###必要な情報
#
#
    # insert_point
    # set_point, set_point_tw
    # vec_x_final, vec_y_final, vec_z_final
    # vec_x_tw_final, vec_y_tw_final, vec_z_tw_final
    # Box, Box_tw

    if insert_vector_mode == 0:
    
        vec_final = np.array([vec_x_final, vec_y_final, vec_z_final])
        vec_tw_final = np.array([vec_x_tw_final, vec_y_tw_final, vec_z_tw_final])

    else:
        vec_final = np.array([vec_x_new, vec_y_new, vec_z_new])
        vec_tw_final = np.array([vec_x_tw, vec_y_tw, vec_z_tw])

    eye = np.eye(4)
    eye_tw = np.eye(4)
    eye[:3,:3] = vec_final.T
    eye_tw[:3,:3] = vec_tw_final.T
 
    return insert_point, set_point, set_point_tw, eye, eye_tw, Box, Box_new, Box_tw, P, end_xyz


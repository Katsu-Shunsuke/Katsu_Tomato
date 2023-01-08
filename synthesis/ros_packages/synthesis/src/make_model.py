import numpy as np
import math
import rospy

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def calc_g(xyz):
    x_g = np.mean(xyz[:,0])
    y_g = np.mean(xyz[:,1])
    z_g = np.mean(xyz[:,2])
    return np.array([x_g, y_g, z_g])

def calc_pedicel_end(pedicel, sepal_g):
    a = np.polyfit(pedicel[:,0], pedicel[:,1], deg=1)[0]
    if np.abs(a) < 1:
        coefs_xy = np.polyfit(pedicel[:,0], pedicel[:,1], deg=2)
        coefs_xz = np.polyfit(pedicel[:,0], pedicel[:,2], deg=2)
        x = sepal_g[0]
        y = np.polyval(coefs_xy, x)
        z = np.polyval(coefs_xz, x)
    
    else:
        coefs_yx = np.polyfit(pedicel[:,1], pedicel[:,0], deg=2)
        coefs_yz = np.polyfit(pedicel[:,1], pedicel[:,2], deg=2)
        y = sepal_g[1]
        x = np.polyval(coefs_yx, y)
        z = np.polyval(coefs_yz, y)

    return np.array([x,y,z])

#def calc_pedicel_end2(pedicel, end_pre):
#    pedicel_near = pedicel[np.argsort(np.sum((pedicel - end_pre)**2, axis=1))[:int(len(pedicel)*0.3)]]
#    coefs_xy = np.polyfit(pedicel_near[:,0], pedicel_near[:,1], deg=1)
#    coefs_xz = np.polyfit(pedicel_near[:,0], pedicel_near[:,2], deg=1)
#    A_xy = pedicel_near[0][:2]
#    n_xy = np.array([1,coefs_xy[0]])
#    b_xy = end_pre[:2] - A_xy
#    t_xy = np.dot(n_xy, b_xy)
#    P_xy = A_xy + t_xy * n_xy
#    
#    end_1 = np.array([P_xy[0], end_pre[2]])
#
#    A_xz = np.array([pedicel_near[0][0], pedicel_near[0][2]])
#    n_xz = np.array([1,coefs_xz[0]])
#    b_xz = end_1 - A_xz
#    t_xz = np.dot(n_xz, b_xz)
#    P_xz = A_xz + t_xz * n_xz
#
#
#   return np.array([P_xz[0], P_xy[1], P_xz[1]])

def calc_pedicel_end2(pedicel, end_pre):
    pedicel_near = pedicel[np.argsort(np.sum((pedicel - end_pre)**2, axis=1))[:int(len(pedicel)*0.3)]]
    coefs_xy = np.polyfit(pedicel_near[:,0], pedicel_near[:,1], deg=1)
    coefs_xz = np.polyfit(pedicel_near[:,0], pedicel_near[:,2], deg=1)
    n = np.array([1, coefs_xy[0], coefs_xz[0]])
    A = pedicel_near[0]
    b = end_pre - A
    t = np.dot(n,b)
    P = A + t*n

    return P


def calc_delimination(pedicel, start, end, center):
    deg = np.arccos(np.dot(start-end, center-end)/np.linalg.norm(start-end)/np.linalg.norm(center-end))
    p_for_d = pedicel[np.arccos((np.dot(pedicel-end,center-end)/np.linalg.norm(pedicel-end,axis=1)/np.linalg.norm(center-end)))>deg]
    if len(p_for_d) != 0: 
        p=0.5
        dis_start = np.sqrt(np.sum((p_for_d-start)**2,axis=1))
        dis_end = np.sqrt(np.sum((p_for_d-end)**2,axis=1))
        delimination = p_for_d[np.argmax(dis_start + dis_end - np.abs(dis_start - dis_end*2)*p)]
        return delimination
    else:
        print("delimination is not detected normally")
        return None

def generate_marker_message(xyz, dia):
        point = xyz * 10**(-3) # mm to m
        dia *= 10**(-3)
        marker_data = Marker()
        marker_data.header.frame_id = "zedm_left_camera_optical_frame"
        marker_data.header.stamp = rospy.Time.now()

        marker_data.ns = "basic_shapes"
        marker_data.id = 0

        marker_data.action = Marker.ADD
        marker_data.pose.position.x = point[0]
        marker_data.pose.position.y = point[1]
        marker_data.pose.position.z = point[2]
        marker_data.pose.orientation.x = 0.0
        marker_data.pose.orientation.y = 0.0
        marker_data.pose.orientation.z = 0.0
        marker_data.pose.orientation.w = 1.0
        marker_data.color.r = 0.0
        marker_data.color.g = 0.5
        marker_data.color.b = 0.5
        marker_data.color.a = 0.4
        
        marker_data.scale.x = dia
        marker_data.scale.y = dia
        marker_data.scale.z = dia
        marker_data.lifetime = rospy.Duration()
        marker_data.type = Marker.SPHERE 
        return marker_data

def straight_pedicel(start, end, dela):
    deg = np.arccos( np.dot((start - dela),(end - dela)) / np.linalg.norm(start - dela)/np.linalg.norm(end - dela)) * 180 / math.pi
    length = np.linalg.norm(dela - end) / np.linalg.norm(start - end)
    if deg > 120 or length < 0.15:
        print("warning ----- this pedicel might be straight")
        print("deg : " + str(deg) + "length : " + str(length))
#!/usr/bin/env python

import os
import sys
import rospy
import tf

import cv2
import numpy as np

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    credit to: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def calc_pedicel_quaternion(vec1, vec2, cutpoint=None, tomato_center=None, pedicel_end=None, mode=3):
    """
    calculate rotation matrix to align pedicel in scissor coordinate y-direction and tangent vector
    mode 0: no constraint
    mode 1: set euler[0] and euler[1] to zero
    mode 2: automatically calculate optimal quaternion by rotating about pedicel direction vector
    mode 3: further rotate from mode2 about pedicel x axis
    mode 4: compute mode 0, then rotate about pedicel x axis according to tangent plane; basically mode 3 but skips the rotation bit about y axis.
    """
    rot = rotation_matrix_from_vectors(vec1, vec2)
    rot_eye = np.eye(4)
    rot_eye[:3, :3] = rot # rotation matrix has to be 4x4 for the tf function
    if mode == 0:
        quaternion = tf.transformations.quaternion_from_matrix(rot_eye)
    elif mode == 1:
        euler = tf.transformations.euler_from_matrix(rot_eye)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, euler[2]) # no need to convert for quaternion because its just direction
    elif mode == 2 or mode == 3 or mode == 4:
        if cutpoint is None or tomato_center is None:
            raise Exception("Must provide cutpoint and/or tomato_center")
        if mode == 2 or mode == 3:
            theta_deg = calc_theta(vec2, tomato_center, cutpoint, rot)
            print("theta_deg:", theta_deg)
            rot_about_pedicel_y = calc_rotation_matrix_about_arbitrary_axis(vec2, theta_deg)
            rot = rot_about_pedicel_y @ rot
        if mode == 3 or mode == 4:
            if pedicel_end is None:
                raise Exception("Must provide pedicel_end")
            pedicel_x = rot @ np.array([1, 0, 0])
            pedicel_z = rot @ np.array([0, 0, 1])
            phi_deg = np.arcsin(np.abs((pedicel_end - tomato_center) @ pedicel_z) / (np.linalg.norm(pedicel_end - tomato_center) * np.linalg.norm(pedicel_z))) * (180 / np.pi) # angle between tangent plane and pedicel z axis
            rot_about_pedicel_x = calc_rotation_matrix_about_arbitrary_axis(pedicel_x, phi_deg)
            rot = rot_about_pedicel_x @ rot
        rot_eye[:3, :3] = rot
        quaternion = tf.transformations.quaternion_from_matrix(rot_eye)
    else:
        raise Exception("Mode not recognized.")
    return quaternion

def calc_rotation_matrix_about_arbitrary_axis(u, theta_deg):
    """
    u: axis (through origin) about which to rotate R, i.e. pedicel tangent vector. must be unit vector!!
    theta_deg: amount by which to rotate in degrees
    credit to: https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python
    """
    from numpy import sin, cos, pi
    u = u / np.linalg.norm(u) # unit vector
    theta = theta_deg * (pi / 180)
    R_about_u = np.array([[cos(theta) + u[0]**2 * (1-cos(theta))           , u[0] * u[1] * (1-cos(theta)) - u[2] * sin(theta), u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
                          [u[0] * u[1] * (1-cos(theta)) + u[2] * sin(theta), cos(theta) + u[1]**2 * (1-cos(theta))           , u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
                          [u[0] * u[2] * (1-cos(theta)) - u[1] * sin(theta), u[1] * u[2] * (1-cos(theta)) + u[0] * sin(theta), cos(theta) + u[2]**2 * (1-cos(theta))]])
    return R_about_u

def calc_theta(v_p, v_t, v_c, rot, adjust_angle_deg=0):
    """
    v_p: pedicel direction vector
    v_t: tomato center position vector
    v_c: cutpoint position vector
    """
    v_pt = v_t - v_c - (v_t - v_c) @ v_p / np.linalg.norm(v_p)**2 * v_p # normal vector to v_p through tomato center (v_t)
    v_pz = rot @ np.array([0, 0, 1]) # z-axis in pedicel coordinate frame
    theta = np.arccos(v_pt @ v_pz / (np.linalg.norm(v_pt) * np.linalg.norm(v_pz)))
    cross_prod = np.cross(v_pt, v_pz)
    if ((cross_prod / v_p) > 0).all():
        # if cross_prod and cutpoint in same direction then negative rotation
        theta *= -1
    return theta * (180 / np.pi) + adjust_angle_deg

def calc_tomato_center(xyz):
    """
    xyz: set of points to fit the sphere (n,3)
    we essentially find the least square solution to the equation, f=Ac
    https://jekel.me/2015/Least-Squares-Sphere-Fit/
    """
    xyz = xyz[remove_outliers(xyz[:,2]), :]
    A = np.ones((xyz.shape[0], 4))
    A[:,:3] = 2 * xyz # add column of ones 
    f = np.sum(xyz ** 2, axis=1)
    c, residules, rank, singval = np.linalg.lstsq(A, f)
    r = np.sqrt(c[0]**2 + c[1]**2 + c[2]**2 + c[3])
    return c[:3], r

def remove_outliers(x, max_deviations=0.5):
    mean = np.mean(x)
    std = np.std(x)
    centered = x - mean
    within_stds = centered < max_deviations * std # boolean array, True means within deviation
    return within_stds

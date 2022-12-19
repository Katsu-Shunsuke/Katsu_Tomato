import numpy as np
import math
import rospy

def hand_box2(point, z_vec):
    z = z_vec / np.linalg.norm(z_vec)
    x = np.array([-z[2], 0, z[0]])
    y = np.array([0,1,0])
    width_1 = 24
    width_2 = 12
    head = 15
    blank = 5
    length = 35
    a1 = point + x * width_1 - y * head
    a2 = point + x * width_2 - y * head
    a3 = point + x * width_2 - y * blank
    a4 = point + x * width_1 - y * blank
    a5 = point - x * width_2 - y * head
    a6 = point - x * width_1 - y * head
    a7 = point - x * width_1 - y * blank
    a8 = point - x * width_2 - y * blank
    a = np.vstack((a1,a2,a3,a4,a5,a6,a7,a8))
    box = np.vstack((a1,a2,a3,a4,a5,a6,a7,a8))
    for i in range(int(length)):
        z_i = (i+1)
        b = a - z * z_i
        box = np.vstack((box, b))
    
    return box
mport numpy as np
import math

def mask_to_xyz(xyz,mask):
    return xyz[mask[:,0], mask[:,1], :] 

def index_to_xyz(i,xyz,mask):

    mask_i = mask[i].astype('int32')

    xyz_i = mask_to_xyz(xyz, mask_i)
    
    return xyz_i

def index_to_xyz_all(xyz, mask):
    
    xyz_all = []
    
    for i in range(len(mask)):
        xyz_all.append(index_to_xyz(i,xyz,mask))
    
    return xyz_all

def remove_outliers(x, max_deviations):
    mean = np.mean(x)
    std = np.std(x)
    centered = x - mean
    within_stds = centered < max_deviations * std # boolean array, True means within deviation
    return within_stds

def calc_tomato_center(xyz, max_deviations):#tomatoのセンターと半径を求める xyz=1x3
    """
    xyz: set of points to fit the sphere (n,3)
    we essentially find the least square solution to the equation, f=Ac
    https://jekel.me/2015/Least-Squares-Sphere-Fit/
    """
    xyz = xyz[remove_outliers(xyz[:,2], max_deviations), :]
    A = np.ones((xyz.shape[0], 4))
    A[:,:3] = 2 * xyz # add column of ones 
    f = np.sum(xyz ** 2, axis=1)
    c, residules, rank, singval = np.linalg.lstsq(A, f)
    r = np.sqrt(c[0]**2 + c[1]**2 + c[2]**2 + c[3])
    return xyz, c[:3], r

def new_e(vec):
    e_x = np.array([1, - vec[0] / vec[1], 0])
    e_z = np.array([vec[0]*vec[2]/vec[1], vec[2], - vec[0] ** 2 / vec[1] - vec[1]])
    
    e_x_unit = (e_x / np.linalg.norm(e_x)).reshape((3,1))
    e_y_unit = (vec / np.linalg.norm(vec)).reshape((3,1))
    e_z_unit = (e_z / np.linalg.norm(e_z)).reshape((3,1))
    
    P = np.hstack((e_x_unit, e_y_unit, e_z_unit))
    return P

def new_field(P, point):
    P_i = np.linalg.inv(P)
    return np.dot(point, P_i.T)


def back_field(P, point):
    return np.dot(point, P.T)

#def nvector(axes, vector, loc,
#                  vcolor="black", alpha=0.5):
#    # 法線ベクトルをプロット
#    axes.quiver(loc[0], loc[1], loc[2],
#                vector[0], vector[1], vector[2],
#                color = vcolor, length = 50, arrow_length_ratio = 0.2)
    
#def sphere(ax, center,r,color):
#    theta1,theta2 = np.mgrid[0:2*np.pi:30j,0:2*np.pi:30j] #mgrid関数でmesh gridを定義
#    x = r*np.cos(theta1)*np.sin(theta2) + center[0] #xの極座標表示 x=r*sin(theta2)cos(theta1)
#    y = r*np.sin(theta1)*np.sin(theta2) + center[1] #yの極座標表示 y=r*sin(theta1)cos(theta2)
#    z = r*np.cos(theta2) + center[2] #xの極座標表示 y=r*cos(theta1)
#
#    #plot_surface(面)とplot_wireframe(グリッド線)の両方の描写を知っておくと便利。
#    #ax.plot_surface(x, y, z,rstride=1, cstride=1, cmap='hsv') #shade=False
#    ax.plot_wireframe(x, y, z, color=color, linewidth=0.5) #linewidthは細め
    
def hand_box(tomato_upper, end_new, start_new, z_vec_new):
    thick = 30
    
    if np.dot((start_new-end_new), z_vec_new) < 0:
        z_vec_new = z_vec_new * -1
        
    z = z_vec_new / np.linalg.norm(z_vec_new)
    x = np.array([-z[2], 0, z[0]])
    p = np.array([end_new[0], tomato_upper, end_new[2]])
    a1 = p + x * thick + z * 10 + np.array([0, -10, 0])
    a2 = p - x * thick + z * 10 + np.array([0, -10, 0])
    a3 = p - x * thick - z * 175 + np.array([0, -10, 0])
    a4 = p + x * thick - z * 175 + np.array([0, -10, 0])
    a5 = a1 + np.array([0,175,0])
    a6 = a2 + np.array([0,175,0])
    a7 = a3 + np.array([0,175,0])
    a8 = a4 + np.array([0,175,0])
    Box = np.vstack((a1,a2,a3,a4,a5,a6,a7,a8))
    return Box

#def Box_shape(Z, ax, facecolor):
#    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
#    verts = [[Z[0],Z[1],Z[2],Z[3]],
#             [Z[4],Z[5],Z[6],Z[7]],
#             [Z[0],Z[1],Z[5],Z[4]],
#             [Z[2],Z[3],Z[7],Z[6]],
#             [Z[1],Z[2],Z[6],Z[5]],
#             [Z[4],Z[7],Z[3],Z[0]]]
#   ax.add_collection3d(Poly3DCollection(verts, facecolors=facecolor, linewidths=1, edgecolors='r', alpha=.20))

def twist_y(vec_x, vec_y, vec_z, theta):
    t = theta * math.pi /180
    n1 = vec_y[0]
    n2 = vec_y[1]
    n3 = vec_y[2]
    c = np.cos(t)
    s = np.sin(t)
    R = np.array([[c + n1**2 * (1 - c), n1 * n2 * (1 - c) - n3 * s, n1 * n3 *(1 -c) + n2 * s],
                 [n2 * n1 * (1 - c) + n3 * s, c + n2 ** 2 * (1 - c), n2 * n3 *(1 - c) - n1 * s],
                 [n3 * n1 * (1 - c) - n2 * s, n3 * n2 *(1 - c) + n1 * s, c + n3 ** 2 * (1 - c)]])
    return np.dot(R, vec_x.T).T, vec_y, np.dot(R, vec_z.T).T, R

def twist_x(vec_x, vec_y, vec_z, theta):
    t = theta * math.pi /180
    n1 = vec_x[0]
    n2 = vec_x[1]
    n3 = vec_x[2]
    c = np.cos(t)
    s = np.sin(t)
    R = np.array([[c + n1**2 * (1 - c), n1 * n2 * (1 - c) - n3 * s, n1 * n3 *(1 -c) + n2 * s],
                 [n2 * n1 * (1 - c) + n3 * s, c + n2 ** 2 * (1 - c), n2 * n3 *(1 - c) - n1 * s],
                 [n3 * n1 * (1 - c) - n2 * s, n3 * n2 *(1 - c) + n1 * s, c + n3 ** 2 * (1 - c)]])
    return vec_x, np.dot(R, vec_y.T).T, np.dot(R, vec_z.T).T, R

def fit_plane(point_cloud):
    """
    入力
        point_cloud : xyzのリストnumpy.array型
    出力
        plane_v : 法線ベクトルの向き(単位ベクトル)
        com : 重心　近似平面が通る点
    """

    com = np.sum(point_cloud, axis=0) / len(point_cloud)
    # 重心を計算
    q = point_cloud - com
    # 重心を原点に移動し、同様に全点群を平行移動する  pythonのブロードキャスト機能使用
    Q = np.dot(q.T, q)
    # 3x3行列を計算する 行列計算で和の形になるため総和になる
    la, vectors = np.linalg.eig(Q)
    # 固有値、固有ベクトルを計算　固有ベクトルは縦のベクトルが横に並ぶ形式
    plane_v = vectors.T[np.argmin(la)]
    if plane_v[1] > 0:
        plane_v = -1 * plane_v
    # 固有値が最小となるベクトルの成分を抽出

    return plane_v

def twist_hand(Box,R, insert_point):
    Box_o = Box - insert_point
    Box_tw = np.dot(R, Box_o.T).T
    return Box_tw + insert_point

def calc_modify_y(vec_y, vec_z, t):
    theta = t * math.pi /180
    R = np.array([[np.cos(theta), - np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
    xz = np.dot(R, np.array([vec_z[0], vec_z[2]]).T).T
    y = - (vec_y[0] * xz[0] + vec_y[2] * xz[1])/ vec_y[1] #垂直にするため
    new_vec = np.array([xz[0], y, xz[1]])
    theta_mod = np.arccos(np.dot(new_vec, vec_z)/(np.linalg.norm(new_vec) * np.linalg.norm(vec_z))) * 180 / math.pi
    return theta_mod

def new_hand_arm_rotaion(vec_x, vec_y, vec_z):
    vec_x_final = np.array([-vec_x[1], vec_x[0], vec_x[2]])
    vec_y_final = np.array([-vec_y[1], vec_y[0], vec_y[2]])
    vec_z_final = np.array([-vec_z[1], vec_z[0], vec_z[2]])
    return vec_x_final, vec_y_final, vec_z_final

def Box_new_tidy(Box_new):
    vector = Box_new[0] - Box_new[3]
    theta =  abs(np.arcsin(vector[2]/np.linalg.norm(vector)))
    
    if vector[0]*vector[2] >=  0:
        R_box = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
    else:
        R_box = np.array([[np.cos(theta), 0, -np.sin(theta)],
                          [0, 1, 0],
                          [np.sin(theta), 0, np.cos(theta)]])
    return R_box

def detect_interference(Box_new, xyz_new, R_box):#Box_new(8x3)xyz_new(nx3)
    Box_tidy = np.dot(R_box, Box_new.T).T
    xyz_tidy = np.dot(R_box, xyz_new.T).T
    max_x = np.max(Box_tidy[:,0])
    min_x = np.min(Box_tidy[:,0])
    max_y = np.max(Box_tidy[:,1])
    min_y = np.min(Box_tidy[:,1])
    max_z = np.max(Box_tidy[:,2])
    min_z = np.min(Box_tidy[:,2])
    center = np.array([(max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2])
    xyz_tidy_1 = (xyz_tidy - center) / (np.array([max_x - min_x, max_y - min_y, max_z - min_z])/2)
    return np.count_nonzero((abs(xyz_tidy_1) < 1).all(axis=1)) #bool

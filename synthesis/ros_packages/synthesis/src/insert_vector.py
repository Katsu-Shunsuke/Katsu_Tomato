mport numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
%matplotlib notebook
import math

def calculate(tomato_index, pedicel_index, xyz, mask_tomato, mask_pedicel, max_deviations, ax=False):
    
    tomato_xyz = index_to_xyz(tomato_index-1, xyz, mask_tomato)
    pedicel_xyz = index_to_xyz(pedicel_index-1, xyz, mask_pedicel)ro
    tomato_cut, center, r = calc_tomato_center(tomato_xyz, max_deviations) #球体フィッティング・トマトの外れ値除外
    pedicel_xyz = pedicel_xyz[remove_outliers(pedicel_xyz[:,2], max_deviations),:] #小花柄の外れ値除外

    end_xyz = pedicel_xyz[np.argmin(np.sum((pedicel_xyz - center)**2, axis=1))]#小花柄のトマト側
    start_xyz = pedicel_xyz[np.argmax(np.sum((pedicel_xyz - center)**2, axis=1))]#小花柄の茎側
    
    tomato_tilt_mode = 1
    if tomato_tilt_mode == 1:
        plane_v =  center - end_xyz
    elif tomato_tilt_mode == 2:   
        tomato_dis = np.sqrt(np.sum((tomato_xyz - end_xyz)**2, axis=1))
        tomato_upper = tomato_xyz[np.where(tomato_dis < 40)]
        plane_v = fit_plane(tomato_upper)

    P = new_e(plane_v)
    p_xyz_new = new_field(P, pedicel_xyz)
    t_xyz_new = new_field(P, tomato_xyz)
    end_xyz_new = new_field(P, end_xyz)
    start_xyz_new = new_field(P, start_xyz)

    coefs_xz_new = np.polyfit(p_xyz_new[:,0], p_xyz_new[:,2], deg=1)

    insert_new = np.array([1, 0, coefs_xz_new[0]]) 

    Box_new = hand_box(end_xyz_new, start_xyz_new, insert_new)
    insert = back_field(P, insert_new)
    Box = back_field(P, Box_new)
                       
    if np.dot((start_xyz-end_xyz), insert) < 0:
        insert = insert * -1
    
    vec_y = - plane_v / np.linalg.norm(plane_v)
    vec_x = np.cross( - plane_v, insert) / np.linalg.norm(np.cross( - plane_v, insert))
    vec_z = insert / np.linalg.norm(insert)
    
    ####
    #姿勢の修正←起動生成できるように
    
    vec_x_new = vec_x
    vec_y_new = vec_y
    vec_z_new = vec_z
    
    insert_h_deg = np.arcsin( vec_z_new[1] / np.linalg.norm(vec_z_new)) * 180 / math.pi
    insert_v_deg = np.arcsin( vec_z_new[2] /np.linalg.norm(np.array([vec_z_new[0], vec_z_new[2]])) ) * 180 / math.pi
    print(insert_h_deg)
    print(insert_v_deg)
    
    new_hand_upper_thick = 5
    interval = 203.01
    insert_point = end_xyz + vec_y * new_hand_upper_thick
    
    
    if abs ( vec_z_new[1] / np.linalg.norm(vec_z_new) ) > np.sin(30 * math.pi / 180):
        vec_x_new, vec_y_new, vec_z_new, R_x = twist_x(vec_x_new, vec_y_new, vec_z_new, 15)
        Box = twist_hand(Box, R_x, insert_point)
    
    if vec_z_new[2] /np.linalg.norm(np.array([vec_z_new[0], vec_z_new[2]])) < np.sin(-45 * math.pi /180):
        vec_x_new, vec_y_new, vec_z_new, R_y = twist_y(vec_x_new, vec_y_new, vec_z_new, 60)
        Box = twist_hand(Box, R_y, insert_point)

    set_point = insert_point - vec_z_new * interval
    
    
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
    hervest = "ok"
    if hervest == "ok":
        vec_x_tw, vec_y_tw, vec_z_tw, R_tw = twist_x(vec_x_new, vec_y_new, vec_z_new, 45)
        Box_tw = twist_hand(Box, R_tw, insert_point)
        
    set_point_tw  = insert_point - vec_z_tw * interval
    
#    #グラフ描写
#    if ax == False:
#        fig = plt.figure()
#       ax = Axes3D(fig)
#        
#    ax.set_xlabel("X")
#    ax.set_ylabel("Y")
#    ax.set_zlabel("Z")
#    
#    ax.plot(tomato_cut[:,0],tomato_cut[:,1],tomato_cut[:,2],marker="o",linestyle='None',color="r")
#    ax.plot(pedicel_xyz[:,0],pedicel_xyz[:,1],pedicel_xyz[:,2],marker="o",linestyle='None',color="y")
#    #ax.plot(Box[:,0],Box[:,1],Box[:,2],marker="o",color="black")
#    Box_shape(Box, ax, "cyan")
#    Box_shape(Box_tw, ax, "gray")
#    
#    ax.scatter(center[0],center[1],center[2],color="black")
#    ax.scatter(end_xyz[0],end_xyz[1],end_xyz[2],color="black")
#    ax.scatter(start_xyz[0],start_xyz[1],start_xyz[2],color="black")
#    
#    nvector(ax, vec_z, insert_point, vcolor="blue")
#    nvector(ax, vec_y, insert_point, vcolor="green")
#    nvector(ax, vec_x, insert_point, vcolor="red")
#    
#    nvector(ax, vec_z_new, insert_point, vcolor="black")
#    nvector(ax, vec_y_new, insert_point, vcolor="black")
#    nvector(ax, vec_x_new, insert_point, vcolor="black")
#    
#    nvector(ax, vec_z_tw, insert_point, vcolor="gray")
#    nvector(ax, vec_y_tw, insert_point, vcolor="gray")
#    nvector(ax, vec_x_tw, insert_point, vcolor="gray")
#    
#    nvector(ax, vec_z_new, set_point, vcolor="blue")
#    nvector(ax, vec_y_new, set_point, vcolor="green")
#    nvector(ax, vec_x_new, set_point, vcolor="red")
#    
#    nvector(ax, vec_z_tw, set_point_tw, vcolor="blue")
#    nvector(ax, vec_y_tw, set_point_tw, vcolor="green")
#    nvector(ax, vec_x_tw, set_point_tw, vcolor="red")
#    
#    
#    
#    ax.set_xlim(end_xyz[0]-75,end_xyz[0]+75)
#    ax.set_ylim(end_xyz[1]-75,end_xyz[1]+75)
#    ax.set_zlim(end_xyz[2]-75,end_xyz[2]+75)
#    #sphere(ax, center,r,color="skyblue")
#    plt.show()
    
    return end_xyz, insert

def mask_to_xyz(xyz,mask):
    return xyz[mask[:,0], mask[:,1], :] 

def index_to_xyz(i,xyz,mask):

    mask_i = mask[i].astype('int32')

    xyz_i = mask_to_xyz(xyz, mask_i)
    
    return xyz_i

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

def nvector(axes, vector, loc,
                  vcolor="black", alpha=0.5):
    # 法線ベクトルをプロット
    axes.quiver(loc[0], loc[1], loc[2],
                vector[0], vector[1], vector[2],
                color = vcolor, length = 50, arrow_length_ratio = 0.2)
    
def sphere(ax, center,r,color):
    theta1,theta2 = np.mgrid[0:2*np.pi:30j,0:2*np.pi:30j] #mgrid関数でmesh gridを定義
    x = r*np.cos(theta1)*np.sin(theta2) + center[0] #xの極座標表示 x=r*sin(theta2)cos(theta1)
    y = r*np.sin(theta1)*np.sin(theta2) + center[1] #yの極座標表示 y=r*sin(theta1)cos(theta2)
    z = r*np.cos(theta2) + center[2] #xの極座標表示 y=r*cos(theta1)

    #plot_surface(面)とplot_wireframe(グリッド線)の両方の描写を知っておくと便利。
    #ax.plot_surface(x, y, z,rstride=1, cstride=1, cmap='hsv') #shade=False
    ax.plot_wireframe(x, y, z, color=color, linewidth=0.5) #linewidthは細め
    
def hand_box(end_new, start_new, z_vec_new):
    thick = 30
    
    if np.dot((start_new-end_new), z_vec_new) < 0:
        z_vec_new = z_vec_new * -1
        
    z = z_vec_new / np.linalg.norm(z_vec_new)
    x = np.array([-z[2], 0, z[0]])
    p = end_new
    a1 = p + x * thick + z * 10
    a2 = p - x * thick + z * 10
    a3 = p - x * thick - z * 175
    a4 = p + x * thick - z * 175
    a5 = a1 + np.array([0,175,0])
    a6 = a2 + np.array([0,175,0])
    a7 = a3 + np.array([0,175,0])
    a8 = a4 + np.array([0,175,0])
    Box = np.vstack((a1,a2,a3,a4,a5,a6,a7,a8))
    return Box

def Box_shape(Z, ax, facecolor):
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
    verts = [[Z[0],Z[1],Z[2],Z[3]],
             [Z[4],Z[5],Z[6],Z[7]],
             [Z[0],Z[1],Z[5],Z[4]],
             [Z[2],Z[3],Z[7],Z[6]],
             [Z[1],Z[2],Z[6],Z[5]],
             [Z[4],Z[7],Z[3],Z[0]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=facecolor, linewidths=1, edgecolors='r', alpha=.20))

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
        point_cloud : xyzのリスト　numpy.array型
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

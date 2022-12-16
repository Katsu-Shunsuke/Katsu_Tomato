import numpy as np
import math

def calc_mask_g(mask):
    g_x = np.mean(mask[:,1].astype("int"))
    g_y = np.mean(mask[:,0].astype("int"))
    return g_x, g_y
def blank_list(mask):
    list = []
    for n in range(len(mask)):
        list.append([])
    return list

def sepal_pedicel(mask_sepal, mask_pedicel, bbox_sepal):

    sepal_pedicel_list = blank_list(mask_sepal)
    after_match_sepal = np.array([])

    for pedicel_index, this_pedicel in enumerate(mask_pedicel):

        x_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,1] # index zero since there are probs multiple
        y_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,0] # index zero since there are probs multiple

        overlapping_sepals = []#選択した小花柄がはいっているトマト
        xy_centers = []#そのトマトの中心

        for j, this_sepal in enumerate(bbox_sepal):#選択した小花柄がトマトのbboxに入っているか
            if (after_match_sepal == j).any() == False:
                x_min, y_min, x_max, y_max = this_sepal[:4]
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                if (x_end > x_min and x_end < x_max) and (y_end > y_min and y_end < y_max):#デフォルト値0.5
                    overlapping_sepals.append(j)
                    xy_centers.append([x_center, y_center])
        
        dists = []#トマトと小花柄の下端との距離
        if len(overlapping_sepals) > 1:
            for xy_center in xy_centers:
                dist = np.sqrt((xy_center[0] - x_end)**2 + (xy_center[1] - y_end)**2)
                #dist = np.abs(x_center - x_end)
                dists.append(dist)
            j_final = overlapping_sepals[dists.index(min(dists))]#一番距離が近いものが収穫するトマト
            sepal_pedicel_list[j_final].append(pedicel_index)#小花柄とのペアが決まったがくは削除
            after_match_sepal = np.append(after_match_sepal, j_final)
        elif len(overlapping_sepals) == 1:
            j_final = overlapping_sepals[0]
            sepal_pedicel_list[j_final].append(pedicel_index)
            after_match_sepal = np.append(after_match_sepal, j_final)
        else: # zero
            j_final = None

    return sepal_pedicel_list

def tomato_sepal(mask_tomato, mask_sepal, bbox_tomato):

    tomato_sepal_list = blank_list(mask_tomato)
    after_match_tomato = np.array([])
    not_match_sepal = np.array([])

    for sepal_index, this_sepal in enumerate(mask_sepal):

        x, y  = calc_mask_g(this_sepal)

        overlapping_tomatoes = []#選択した小花柄がはいっているトマト
        xy_centers = []#そのトマトの中心

        for j, this_tomato in enumerate(bbox_tomato):#選択した小花柄がトマトのbboxに入っているか
            if (after_match_tomato == j).any() == False:
                x_min, y_min, x_max, y_max = this_tomato[:4]
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                if (x > x_min and x < x_max) and (y > y_min and y < y_max):#デフォルト値0.5
                    overlapping_tomatoes.append(j)
                    xy_centers.append([x_center, y_center])
        
        dists = []#トマトと小花柄の下端との距離
        if len(overlapping_tomatoes) > 1:
            for xy_center in xy_centers:
                dist = np.sqrt((xy_center[0] - x)**2 + (xy_center[1] - y)**2)
                #dist = np.abs(x_center - x_end)
                dists.append(dist)
            j_final = overlapping_tomatoes[dists.index(min(dists))]#一番距離が近いものが収穫するトマト
            tomato_sepal_list[j_final].append(sepal_index)#小花柄とのペアが決まったがくは削除
            after_match_tomato = np.append(after_match_tomato, j_final)
        elif len(overlapping_tomatoes) == 1:
            j_final = overlapping_tomatoes[0]
            tomato_sepal_list[j_final].append(sepal_index)
            after_match_tomato = np.append(after_match_tomato, j_final)
        else: # zero
            not_match_sepal = np.append(not_match_sepal, sepal_index)

    if not_match_sepal is not None and len(after_match_tomato) < len(mask_tomato):
        for sepal_index in not_match_sepal:
            x, y = calc_mask_g(mask_sepal[sepal_index])
            dists = []
            for j, this_tomato in enumerate(bbox_tomato):
                if (after_match_tomato == j).any() == False:
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    dist = np.sqrt((xy_center[0] - x)**2 + (xy_center[1] - y)**2)
                    dists.append(dist)
            j_final = overlapping_tomatoes[dists.index(min(dists))]
            tomato_sepal_list[j_final].append(sepal_index)

    return tomato_sepal_list


def tomato_pedicel(mask_tomato, mask_pedicel, bbox_tomato):

    tomato_pedicel_list = blank_list(mask_tomato)
    after_match_tomato = np.array([])

    # if mask_pedicel_sorted is empty then this loop is skipped. 
    for pedicel_index, this_pedicel in enumerate(mask_pedicel):
        
        x_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,1] # index zero since there are probs multiple
        y_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,0] # index zero since there are probs multiple
        
        overlapping_tomatoes = []#選択した小花柄がはいっているトマト
        xy_centers = []#そのトマトの中心
        for j, this_tomato in enumerate(bbox_tomato):#選択した小花柄がトマトのbboxに入っているか
            if (after_match_tomato == j).any() == False:
                x_min, y_min, x_max, y_max = this_tomato[:4]
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                if (x_end > x_min and x_end < x_max) and (y_end > y_min and y_end < 0.5 * (y_max - y_min) + y_min):#デフォルト値0.5
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
            after_match_tomato = np.append(after_match_tomato, j_final)
        elif len(overlapping_tomatoes) == 1:
            j_final = overlapping_tomatoes[0]
            tomato_pedicel_list[j_final].append(pedicel_index)
            after_match_tomato = np.append(after_match_tomato, j_final)
        else: # zero
            j_final = None

    return tomato_pedicel_list

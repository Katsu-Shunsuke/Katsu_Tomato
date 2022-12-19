import numpy as np
import math

def calc_mask_g(mask):
    g_x = np.mean(mask[:,1].astype("int"))
    g_y = np.mean(mask[:,0].astype("int"))
    return g_x, g_y
def blank_list(n):
    list = []
    for n in range(n):
        list.append([])
    return list

def pedicel_sepal(mask_sepal, mask_pedicel, bbox_sepal):

    pedicel_sepal_list = []
    #sepal_pedicel_list = blank_list(mask_sepal)
    #after_match_sepal = np.array([])

    for pedicel_index, this_pedicel in enumerate(mask_pedicel):

        x_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,1] # index zero since there are probs multiple
        y_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,0] # index zero since there are probs multiple

        overlapping_sepals = []
        xy_centers = []

        for j, this_sepal in enumerate(bbox_sepal):#選択した小花柄がsepalのbboxに入っているか
            #if (after_match_sepal == j).any() == False:
            x_min, y_min, x_max, y_max = this_sepal[:4]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            if (x_end > x_min and x_end < x_max) and (y_end > y_min and y_end < y_max):#デフォルト値0.5
                overlapping_sepals.append(j)
                xy_centers.append([x_center, y_center])
        
        dists = []#トマトと小花柄の下端との距離
        if len(overlapping_sepals) >= 1:
            for xy_center in xy_centers:
                dist = np.sqrt((xy_center[0] - x_end)**2 + (xy_center[1] - y_end)**2)
                dists.append(dist)
            dists = np.array(dists)
            overlapping_sepals = np.array(overlapping_sepals)
            pedicel_sepal_list.append(overlapping_sepals[np.argsort(dists)].tolist())
        #    j_final = overlapping_sepals[dists.index(min(dists))]#一番距離が近いものが収穫するトマト
        #    sepal_pedicel_list[j_final].append(pedicel_index)#小花柄とのペアが決まったがくは削除
        #    after_match_sepal = np.append(after_match_sepal, j_final)
        #elif len(overlapping_sepals) == 1:
        #    j_final = overlapping_sepals[0]
        #    sepal_pedicel_list[j_final].append(pedicel_index)
        #    after_match_sepal = np.append(after_match_sepal, j_final)
        else:
            pedicel_sepal_list.append([])
            # zero
        #    j_final = None

    return pedicel_sepal_list

def sepal_tomato(mask_tomato, mask_sepal, bbox_tomato):

    sepal_tomato_list = []
    tomato_check = np.full(len(mask_tomato), True)
    log_overlap = []

    for sepal_index, this_sepal in enumerate(mask_sepal):

        x, y  = calc_mask_g(this_sepal)

        overlapping_tomatoes = []#選択した小花柄がはいっているトマト
        xy_centers = []#そのトマトの中心

        for j, this_tomato in enumerate(bbox_tomato):#選択した小花柄がトマトのbboxに入っているか
            #if (after_match_tomato == j).any() == False:
            x_min, y_min, x_max, y_max = this_tomato[:4]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            if (x > x_min and x < x_max) and (y > y_min and y < y_max):
                overlapping_tomatoes.append(j)
                xy_centers.append([x_center, y_center])
        
        log_overlap.append(overlapping_tomatoes)
        dists = []#トマトと小花柄の下端との距離
        if len(overlapping_tomatoes) >= 1:
            for xy_center in xy_centers:
                dist = np.sqrt((xy_center[0] - x)**2 + (xy_center[1] - y)**2)
                dists.append(dist)
            dists = np.array(dists)
            overlapping_tomatoes = np.array(overlapping_tomatoes)
            append_tomato = overlapping_tomatoes[np.argsort(dists)][0].tolist()
            sepal_tomato_list.append([append_tomato])
            tomato_check[append_tomato] = False
        else: # なかったら近いトマト
            if len(mask_tomato) >= len(mask_sepal):
                xy_centers2 = []
                for j, this_tomato in enumerate(bbox_tomato):
                    x_min, y_min, x_max, y_max = this_tomato[:4]
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    xy_centers2.append([x_center, y_center])
                xy_centers2 = np.array(xy_centers2)
                sepal_center = np.array([x,y])
                append_tomatoes = np.argsort(np.sqrt(np.sum((xy_centers2 - sepal_center )**2, axis=1)))
                for append_tomato in append_tomatoes:
                    if tomato_check[append_tomato]:
                        sepal_tomato_list.append([append_tomato])
                        break

    for i in range(len(sepal_tomato_list)):
        for j in range(i+1,len(sepal_tomato_list)):
            if sepal_tomato_list[i] != [] and sepal_tomato_list[j] != []:
                if sepal_tomato_list[i][0] == sepal_tomato_list[j][0]:
                    tomato_index = sepal_tomato_list[i][0]
                    mask_i_center_x, mask_i_center_y = calc_mask_g(mask_sepal[i])
                    mask_j_center_x, mask_j_center_y = calc_mask_g(mask_sepal[j])
                    x_min, y_min, x_max, y_max = bbox_tomato[tomato_index][:4]
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    dis_i = np.sqrt((mask_i_center_x - x_center)**2 + (mask_i_center_y - y_center)**2)
                    dis_j = np.sqrt((mask_j_center_x - x_center)**2 + (mask_j_center_y - y_center)**2)
                    if dis_i < dis_j:
                        if len(log_overlap[j]) > 1:
                            for log in range(1, len(log_overlap[j])):
                                if tomato_check[log_overlap[j][log]]:
                                    sepal_tomato_list[j][0] = log_overlap[j][log]

                    else:
                        if len(log_overlap[i]) > 1:
                            for log in range(1, len(log_overlap[i])):
                                if tomato_check[log_overlap[i][log]]:
                                    sepal_tomato_list[i][0] = log_overlap[i][log]


    return sepal_tomato_list


def pedicel_tomato(mask_tomato, mask_pedicel, bbox_tomato):

    pedicel_tomato_list = []
    #tomato_pedicel_list = blank_list(mask_tomato)
    #after_match_tomato = np.array([])

    # if mask_pedicel_sorted is empty then this loop is skipped. 
    for pedicel_index, this_pedicel in enumerate(mask_pedicel):
        
        x_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,1] # index zero since there are probs multiple
        y_end = this_pedicel[this_pedicel[:,0]==np.max(this_pedicel[:,0])][0,0] # index zero since there are probs multiple
        
        overlapping_tomatoes = []#選択した小花柄がはいっているトマト
        xy_centers = []#そのトマトの中心
        for j, this_tomato in enumerate(bbox_tomato):#選択した小花柄がトマトのbboxに入っているか
            #if (after_match_tomato == j).any() == False:
            x_min, y_min, x_max, y_max = this_tomato[:4]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            if (x_end > x_min and x_end < x_max) and (y_end > y_min and y_end < 0.5 * (y_max - y_min) + y_min):#デフォルト値0.5
                overlapping_tomatoes.append(j)
                xy_centers.append([x_center, y_center])
            
        dists = []#トマトと小花柄の下端との距離
        if len(overlapping_tomatoes) >= 1:
            for xy_center in xy_centers:
                dist = np.sqrt((xy_center[0] - x_end)**2 + (xy_center[1] - y_end)**2)
                dists.append(dist)
            dists = np.array(dists)
            overlapping_tomatoes = np.array(overlapping_tomatoes)
            pedicel_tomato_list.append(overlapping_tomatoes[np.argsort(dists)].tolist())
#            j_final = overlapping_tomatoes[dists.index(min(dists))]#一番距離が近いものが収穫するトマト
#            tomato_pedicel_list[j_final].append(pedicel_index)
#            after_match_tomato = np.append(after_match_tomato, j_final)
#        elif len(overlapping_tomatoes) == 1:
#            j_final = overlapping_tomatoes[0]
#            tomato_pedicel_list[j_final].append(pedicel_index)
#            after_match_tomato = np.append(after_match_tomato, j_final)
        else: # zero
            pedicel_tomato_list.append([])
            #j_final = None

    return pedicel_tomato_list

def check_relation_list(p_s, s_t, p_t, n_t, n_s, n_p):
    s_p = reverse_list(p_s, n_s)
    t_s = reverse_list(s_t, n_t)
    t_p = reverse_list(p_t, n_t)
    list_sum = []
    check_tomato = np.full(n_t,True)

    for s, t_list in enumerate(s_t):
        if t_list != []:
            t = t_list[0]
            n_sp = len(s_p[s])
            n_tp = len(t_p[t])
            
            for i in range(n_sp):
                for j in range(n_tp):
                    if s_p[s][i] == t_p[t][j]:
                        p = s_p[s][i]
                        if check_tomato[t]:
                            list_sum.append([t,s,p])
                            check_tomato[t] = False
                            break
            
    for i in range(len(check_tomato)):
        if check_tomato[i]:
            t = i
            for j in range(len(t_s[t])):
                s = t_s[t][j]
                n_sp = len(s_p[s])
                for k in range(n_sp):
                    p = s_p[s][k]
                    if check_tomato[t]:
                        list_sum.append([t,s,p])
                        check_tomato[t] = False
                        break

        if check_tomato[i]:
            t = i
            for j in range(len(t_s[t])):
                s = t_s[t][j]
                n_tp = len(t_p[t])
                for k in range(n_tp):
                    p = t_p[t][k]
                    if check_tomato[t]:
                        list_sum.append([t,s,p])
                        check_tomato[t] = False
                        break

    return np.array(list_sum)

def reverse_list(list_1_2, n_2):
    new_list = blank_list(n_2)
    for i in range(len(list_1_2)):
        if list_1_2[i] != []:
            for j in (list_1_2[i]):
                new_list[j].append(i)
    
    return new_list

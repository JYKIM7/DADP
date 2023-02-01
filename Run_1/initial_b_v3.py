"""
Step 2:
Initial the diseased regions from patients (k-means)
"""

import numpy as np
# from numpy import dot
from sklearn.cluster import KMeans

def initial_b_fun_v3(res_mat, sub_id_1, mask, landmarks, idx):
    n_obs, num_pxl = np.shape(res_mat)
    b_0 = 2*np.ones(shape=(n_obs, num_pxl))
    nclasses = 12
    mu = np.zeros(shape=(res_mat.shape[0], nclasses)) # mu.shape
    n_sub = len(np.unique(sub_id_1))

    for j in range(n_sub):
        obs_list = np.where(obs_id_1 == np.unique(sub_id_1)[j])

        if len(obs_list)==1:
            i = obs_list[0]
            kk = len(sub_id_1) // len(np.unique(sub_id_1))
            fea_mat_i = np.hstack(( np.reshape(res_mat[i, :], (-1, 1)) , \
                        np.reshape(np.tile(landmarks[0],kk), (-1, 1)), \
                        np.reshape(np.tile(landmarks[1],kk), (-1, 1)) ))
            
            kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(fea_mat_i)
            centers = kmeans.cluster_centers_
            c_sort = np.sort(centers[:, 0])
            clusters_id = np.where(centers[:, 0] <= c_sort[2])
            b_pxl = np.asarray([-1, -1])
            
            for c_id in clusters_id:
                b_pxl_l = np.where(np.isin(kmeans.labels_, c_id))[0]
                b_pxl = np.hstack((b_pxl, b_pxl_l))
            for r in i:
                b_pxl_2 = b_pxl[2:]
                tempidx = b_pxl_2 // num_pxl
                b_idx = np.where( tempidx == (i[i==r] % kk)[0].astype(int) )
                b_0[r, (b_pxl_2[b_idx] % num_pxl) ] = 5
    return b_0



"""
def initial_b_fun(res_mat, sub_id, mask, landmarks, idx):
    
    n_obs, num_pxl = np.shape(res_mat)
    b_0 = 2*np.ones(shape=(n_obs, num_pxl))
    mu = np.zeros(shape=(res_mat.shape[0], nclasses))
    
    nclasses = 12
    
    sub_id = np.unique(obs_id)
    n_sub = len(sub_id)
    
    for j in range(n_sub):
        obs_list =np.where(obs_id == sub_id[j])
        if len(obs_list)==1:
            i = obs_list[0]
            # fea_mat_j = np.hstack((np.reshape(res_mat[i, :], (-1, 1)), landmarks]))
            fea_mat_j = np.hstack(( np.reshape(res_mat[list(obs_list), :], (-1, 1)) , landmarks))
            kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(fea_mat_i)
            centers = kmeans.cluster_centers
            c_sort = np.sort(centers[:, 0])
            clusters_id = np.where(centers[:, 0] <= c_sort[2])
            b_pxl = np.asarray([-1, -1])
            for c_id in clusters_id:
                b_pxl_l = np.where(kmeans.labels_ == c_id)
                b_pxl = np.hstack((b_pxl, b_pxl_l))
            b_0[i, b_pxl[2:]] = 5
                
        elif len(obs_list)>1:
            b_pxl0 = np.arange(len(idx))
            for i in obs_list[::-1]:
                fea_mat_j = np.hstack((np.reshape(res_mat[j, b_pxl0], (-1, 1)), landmarks0[b_pxl0, :]))
                kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(fea_mat_i)
                centers = kmeans.cluster_centers
                c_sort = np.sort(centers[:, 0])
                clusters_id = np.where(centers[:, 0] <= c_sort[6])
                b_pxl = np.asarray([-1, -1])
                for c_id in clusters_id:
                    b_pxl_j = np.where(kmeans.labels_ == c_id)
                    b_pxl = np.hstack((b_pxl, b_pxl_j))
                b_pxl0 = b_pxl[2:]
                b_0[i, b_pxl0] = 5

    return b_0
    return b_0
"""

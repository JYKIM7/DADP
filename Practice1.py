import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import glob

import os
os.chdir('C:/Users/JYKIM/Desktop/Student_Kim_220915/DADP/')

input_dir = '~~~~'
# input_dir = '~~~~'
# output_dir

print(""" Step 0. load dataset """)
print("+++++++Read the info file+++++++")
# 5 variables: subject id, diagnostic status (0: NC, 1: diseased), gender, bmi, and age
info_name = input_dir + 'info.txt'
info = np.loadtxt(info_name)
obs_id = info[:, 0]
sub_id = np.unique(obs_id)

###############################################
#info = info[np.where( info[:, 0] %20==0)]
info = info[np.where(np.isin(info[:, 0] , [20,40,60,80,101,103,114]))]
obs_id = info[:, 0]
sub_id = np.unique(obs_id)
sub_id
###############################################

num_sub = len(sub_id)
print("The number of subjects is " + str(num_sub))
# The number of subjects is 160
dx = info[:, 1]
obs_id_0 = obs_id[dx == 0]
print("The number of normal observations is " + str(len(obs_id_0)))
#   The number of normal observations is 400 = 100*4
obs_id_1 = obs_id[dx > 0]
print("The number of diseased observations is " + str(len(obs_id_1)))
#   The number of diseased observations is 240 = 60*4

x_design = info[:, [0, 2, 3, 4]]
x_design[:, 0] = info[:, 0]                    # subject id
x_design[:, 1] = info[:, 2]
x_design[:, 2] = stats.zscore(info[:, 3])      # normalized bmi
x_design[:, 3] = stats.zscore(info[:, 4])      # normalized age
#x_design[:, 3] = stats.zscore( np.log(info[:, 4] )) # both fine
var_list = ['sub_id', 'gender', 'bmi', 'age']


###############################################
###############################################
print("+++++++Display one randomly selected abnormal pattern+++++++")
bmap_name = input_dir + 'bmap/b_500.txt'
bmap = np.loadtxt(bmap_name).astype(int)
plt.imshow(bmap)

###############################################
###############################################
print("+++++++Read the mask file+++++++")
mask_name = input_dir + 'mask.txt'
mask = np.loadtxt(mask_name).astype(int)
print("The image size is " + str(mask.shape))
#   The image size is (310, 310)
num_pxl = len(mask[mask == 1])
print("The number of pixels in the mask is " + str(num_pxl))
#   The number of pixels in the mask is 38009

###############################################
###############################################
print("+++++++Read the thickness maps+++++++")
img_folder_name = input_dir + "thickness_map"
img_names = glob.glob("%s/map_*.txt" % img_folder_name)
import re
img_names = sorted(img_names, key=lambda x:float(re.findall("(\d+)",x)[1]))
# img_names[0:15]
num_img = len(img_names)
img_data = np.zeros(shape=(num_img, num_pxl))
img_names[0]
640/160
for ii in range(num_img):
    img_ii = np.loadtxt(img_names[ii])
    img_data[ii, :] = img_ii[mask == 1]
print("The matrix dimension of image data is " + str(img_data.shape))
#   The matrix dimension of image data is (640, 38009)

###############################################
sub_id_list=0
for ii in range(len(sub_id)):
    temp = [sub_id[ii]*4-3, sub_id[ii]*4-2, sub_id[ii]*4-1, sub_id[ii]*4]
    sub_id_list = np.append(sub_id_list, temp )    
sub_id_list2 = sub_id_list[1:]
sub_id_list = sub_id_list2-1
sub_id_list
img_data = img_data[ sub_id_list.astype(int) ,:]
img_data.shape
###############################################

###############################################
###############################################
# lmm_fun_v2(y_design_0, x_design_0, var_list, fe_idx, re_idx)
print("+++++++Step 1: Set up linear mixed model on all voxels from normal subjects+++++++")

y_design_0 = img_data[dx == 0, :]
x_design_0 = x_design[dx == 0, :]
fe_idx = [1, 2, 3]
re_idx = [1, 2, 3]

# lmm_fun_v3(Y, X, var_list, fe_idx, re_idx)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
Y = y_design_0 
X = x_design_0

n_obs, num_pxl = np.shape(Y)
p = len(fe_idx) + 1
betas_hat = np.zeros((p, num_pxl))
pvalues = np.zeros((p, num_pxl))
s2 = np.zeros((1, num_pxl))
column_names = np.append(var_list, 'thickness') 
mdl_str = 'thickness ~ '

for j in fe_idx:
    mdl_str = mdl_str + var_list[j] + ' +'
mdl_str = mdl_str[:-2]

re_str = '~ '  
for j in re_idx:
    re_str = re_str + var_list[j] + ' +'
re_str = re_str[:-2]
    
    for k in np.arange(num_pxl): 
        y_k = np.reshape(Y[:, k], (-1, 1))  # y_k.shape     (400, 1)
        array_k = np.hstack((X, y_k))       # array_k.shape (400, 5)
        df_k = pd.DataFrame(array_k, columns=column_names)
        # Run LMER at k-th pixel
        md = smf.mixedlm(mdl_str, df_k, groups=df_k[var_list[0]], re_formula=re_str)
        try:
            mdf = md.fit(method=["lbfgs"])     
        except :
            y_k_temp = np.reshape(Y[:, k], (-1, 1))
            array_k_temp = np.hstack((X, y_k_temp))
            array_k_temp = array_k_temp \
                + 0.00001*np.random.randn(array_k_temp.shape[0], array_k_temp.shape[1])
            df_k_temp = pd.DataFrame(array_k_temp, columns=column_names)
            md = smf.mixedlm(mdl_str, df_k_temp, groups=df_k[var_list[0]], re_formula=re_str)
            mdf = md.fit(method=["lbfgs"])        
        params_hat = mdf.params.to_numpy()
        betas_hat[:, k] = params_hat[:p]
        s2[:,k] = mdf.scale
        pvalues[:, k] = mdf.pvalues.to_numpy()[:p]
        print(k)

mdf.summary()

# np.savetxt("betas_hat_5.txt", betas_hat)
# np.savetxt("s2_5.txt", betas_hat)
# np.savetxt("pvalues_5.txt", betas_hat)

# betas_hat = np.loadtxt('betas_hat_20.txt')
pvalues = np.loadtxt('pvalues_20.txt')


# betas_hat = np.loadtxt('betas_hat_20.txt')
betas_hat = np.loadtxt('betas_hat_5.txt')
betas_hat.shape
###############################################
###############################################
# initial_b_fun(res_mat, sub_id_1, mask, landmarks, idx)
print("+++++++Step 2: Initial the diseased regions from patients (k-means)+++++++")
# betas_hat = np.loadtxt('betas_hat_5.txt')
y_design_1 = img_data[dx > 0, :] # y_design_1.shape
x_design_1 = x_design[dx > 0, :] # x_design_1.shape
res_mat = y_design_1-np.dot(x_design_1, betas_hat) # res_mat.shape
sub_id_1 = x_design_1[:, 0] # sub_id_1
idx = np.nonzero(mask)      # idx.shape 
landmarks = np.asarray(idx)

res_mat.shape
res_mat[:,1]
# initial_b_fun(res_mat, sub_id_1, mask, landmarks, idx)
n_obs, num_pxl = np.shape(res_mat)
b_0 = 2*np.ones(shape=(n_obs, num_pxl))
nclasses = 20
mu = np.zeros(shape=(res_mat.shape[0], nclasses)) # mu.shape
n_sub = len(np.unique(sub_id_1))

#sub_id = np.unique(obs_id)
sub_id_1 
n_sub = len(np.unique(sub_id_1))

j=0
nclasses = 12
b_0 = 2*np.ones(shape=(n_obs, num_pxl))
from sklearn.cluster import KMeans
for j in range(n_sub):
    obs_list = np.where(obs_id_1 == np.unique(sub_id_1)[j])
    if len(obs_list)==1:
        i = obs_list[0]
        kk = len(sub_id_1) // len(np.unique(sub_id_1))
        fea_mat_i = np.hstack(( np.reshape(res_mat[i, :], (-1, 1)) , \
                    np.reshape(np.tile(landmarks[0],kk), (-1, 1)), \
                    np.reshape(np.tile(landmarks[1],kk), (-1, 1)) ))
    
        kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(fea_mat_i)
        #kmeans = KMeans(n_clusters=nclasses).fit(fea_mat_i)
        centers = kmeans.cluster_centers_
        c_sort = np.sort(centers[:, 0])
        
        #centers[:, 0].shape
        #c_sort.shape
        clusters_id = np.where(centers[:, 0] <= c_sort[2])
        b_pxl = np.asarray([-1, -1])
        for c_id in clusters_id:
            b_pxl_l = np.where(np.isin(kmeans.labels_, c_id))[0]
            b_pxl = np.hstack((b_pxl, b_pxl_l))
            
        b_pxl_2 = b_pxl[2:]
        tempidx = b_pxl_2 // num_pxl
        for r in np.unique(tempidx):
            b_idx = np.where( tempidx == (r % kk) )
            b_0[kk*j+r, (b_pxl_2[b_idx] % num_pxl) ] = 5
    
            
idx = np.nonzero(mask)
landmarks = ind2sub_v2(np.shape(mask), np.asarray(idx))  
np.bincount( sum(landmarks[1] == np.asarray(idx)) )
landmarks[1][0][8]
landmarks[2][0][8]

res_mat[i, :].shape
r=0
b_0 = 2*np.ones(shape=(n_obs, num_pxl))
0 1 2 3
4 5 6 7
8 9 10 11
b_0[0,:]
xxx = b_0[8,:].astype(int)
np.unique(xxx)
np.bincount(xxx)

asd = pd.DataFrame( b_0[0,:] )
asd.columns = ['value']
asd[asd.value==5].index[0:55]

b_pxl_2_df = pd.DataFrame( b_pxl_2, tempidx )
b_pxl_2_df
np.unique(tempidx)   # array([0, 1, 2, 3], dtype=int64)
np.bincount(tempidx) # array([8612, 8608, 8607, 8607], dtype=int64)
ii = np.nonzero(y)[0]
np.vstack((ii,y[ii])).T

nclasses = 12
b_0 = 2*np.ones(shape=(n_obs, num_pxl))
from sklearn.cluster import KMeans
for j in range(n_sub):
    obs_list = np.where(obs_id_1 == np.unique(sub_id_1)[j])

    if len(obs_list)==1:
        i = obs_list[0]
        kk = len(sub_id_1) // len(np.unique(sub_id_1))
        fea_mat_i =np.vstack(( res_mat[i, :] , landmarks[0].T, landmarks[1].T)).T

        kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(fea_mat_i)
        centers = kmeans.cluster_centers_
        c_sort = np.sort(centers[:, 0])
   
        centers[:, 0].shape
        c_sort.shape
        clusters_id = np.where(centers[:, 0] <= c_sort[1])
        b_pxl = np.asarray([-1, -1])
        for c_id in clusters_id:
            b_pxl_l = np.where(np.isin(kmeans.labels_, c_id))[0]
            b_pxl = np.hstack((b_pxl, b_pxl_l))
        for r in i:
            b_0[r, b_pxl[2:]] = 5  

b_0_initi = b_0

np.where(b_0[8,:] ==5)
np.sum(np.where(b_0[8,:] ==5))

np.where(b_0[9,:] ==5)
np.sum(np.where(b_0[9,:] ==5))
np.where(b_0[10,:] ==5)
np.sum(np.where(b_0[10,:] ==5))
###############################################
b_0.shape
ccc = b_0[0,:]
ccc = b_0[1,:]
ccc = b_0[8,:]

ccc = b_0_initi[0,:]

ccc = b_01[ii, :]/2
ccc = u2.flatten()
ccc = b0_i.flatten()
ccc = np.argmin(u, axis=1).flatten()
ccc = np.argmin(u2, axis=1).flatten()
ccc = np.argmax(u2, axis=1).flatten()


ccc = np.argmin(u1 + u2, axis=1).flatten()

xxx = ccc.astype(int)
np.unique(xxx)
np.bincount(xxx)

ccc = y_design_1[0,:]
ccc =  b_0[0,:]


ccc = np.argmin(u, axis=1).flatten()

template = mask
vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
ind_list = np.where(vec_template == 1)[0]
ddd = np.zeros(shape=(template.shape[0] * template.shape[1] , 1))
bbb = pd.DataFrame({'value': ccc}, index=ind_list)
ddd[ind_list] = np.reshape(np.array(bbb.value.astype(int)), (num_pxl, 1))
eee = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a').astype(int)
plt.imshow(eee)



plt.imshow(y_design_1)
fff = y_design_1
fff[fff > np.mean(fff)]


a = eee
a = np.random.randint(0,2,(4,4))
i=0
for i in range(a.shape[0]):
    a[[i, a.shape[0]-1-i]] = a[[a.shape[0]-1-i, i]]
a
plt.imshow(a)

a[12,:] == eee[12,:]

bbb[bbb.value==5].tail(n=55)
bbb[bbb.value==5].index[0:55]
ind2sub_v2(eee.shape, np.array(bbb[bbb.value==5].index[0]) )
#####


a = np.random.randint(0,2,(4,4))
a
cc = 2*np.ones(shape=(np.count_nonzero(a), 1))


a_template = np.reshape(a, (a.shape[0] * a.shape[1], 1))
a_list = np.where(a_template == 1)[0]

add = np.zeros(shape=(a.shape[0] * a.shape[1] , 1))

abb = pd.DataFrame(cc, index=a_list)
abb.columns = ['value']

add[a_list] = np.reshape(np.array(abb.value.astype(int)), (3, 1))
np.reshape(add, (a.shape[0],  a.shape[1]), order='c')

###############################################
###############################################

bmap_name = input_dir + 'bmap/b_403.txt'
bmap = np.loadtxt(bmap_name).astype(int)
plt.imshow(bmap)
np.where(np.reshape(bmap, (-1, 1)) > 6)
np.where(np.reshape(bmap, (-1, 1)) > 6)


bbb[bbb.value==5].index[2:55]
np.reshape(bmap, (-1, 1))
abb = pd.DataFrame(np.reshape(bmap, (-1, 1)))
abb.columns = ['value']
abb[abb.value==10].index[0:10]
8,  60,  62,  98, 102
abb.value[ bbb[bbb.value==5].index[1:100] ]=10
abb.value[ [8,  60,  62,  98, 102] ]=10

ind2sub_v2(bmap.shape, np.array(65549) )


np.where(np.reshape(bmap, (-1, 1)) > 6)
ind2sub_v2(bmap.shape, np.array(65549) )

abb.value.astype(int)[65549]
np.reshape(np.array(abb.value.astype(int)), (310, 310)) [211,139]

plt.imshow(np.reshape(np.array(abb.value.astype(int)), (310, 310)))

x = bmap.flatten()
y = np.bincount(x)
ii = np.nonzero(y)[0]
np.vstack((ii,y[ii])).T
plt.hist(x)
plt.show()

ind2sub_v2(bmap.shape, np.array(6933) )

65549

########################################################3
##########################
from tslearn.clustering import TimeSeriesKMeans
j=1
nclasses = 12
b_0 = 2*np.ones(shape=(n_obs, num_pxl))
from sklearn.cluster import KMeans
for j in range(n_sub):
    obs_list = np.where(obs_id_1 == np.unique(sub_id_1)[j])
    if len(obs_list)==1:
        i = obs_list[0]
        #i = obs_list[0][3]
        #kk=1
        kk = len(sub_id_1) // len(np.unique(sub_id_1))
        fea_mat_i = np.hstack(( np.reshape(res_mat[i, :], (-1, 1)) , \
                    np.reshape(np.tile(landmarks[0],kk), (-1, 1)), \
                    np.reshape(np.tile(landmarks[1],kk), (-1, 1)) ))
    
        #kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(fea_mat_i)
        kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw",max_iter=10).fit(fea_mat_i)
        #kmeans = KMeans(n_clusters=nclasses).fit(fea_mat_i)
        centers = kmeans.cluster_centers_
        c_sort = np.sort(centers[:, 0])
        
        centers[:, 0].shape
        c_sort.shape
        clusters_id = np.where(centers[:, 0] <= c_sort[2])
        b_pxl = np.asarray([-1, -1])
        for c_id in clusters_id:
            b_pxl_l = np.where(np.isin(kmeans.labels_, c_id))[0]
            b_pxl = np.hstack((b_pxl, b_pxl_l))
            
        b_pxl_2 = b_pxl[2:]
        tempidx = b_pxl_2 // num_pxl
        for r in np.unique(tempidx):
            b_idx = np.where( tempidx == (r % kk) )
            b_0[kk*j+r, (b_pxl_2[b_idx] % num_pxl) ] = 5
            
import time
import datetime # datetime 라이브러리 import
 
start = time.time() # 시작
kmeans = TimeSeriesKMeans(n_clusters=10, metric="dtw",max_iter=10).fit(fea_mat_i)
sec = time.time()-start # 종료 - 시작 (걸린 시간)
 
times = str(datetime.timedelta(seconds=sec)) # 걸린시간 보기좋게 바꾸기
short = times.split(".")[0] # 초 단위 까지만
print(f"{times} sec")
print(f"{short} sec")

kmeans.cluster_centers_

# np.savetxt('kmeans_20.txt', kmeans.cluster_centers_)
np.save("kmeans.npy", kmeans.cluster_centers_)
zzz = np.load("kmeans.npy")
zzz

centers = kmeans.cluster_centers_
c_sort = np.sort(centers[:, 0])

centers[:, 0].shape
c_sort.shape
clusters_id = np.where(centers[:, 0] <= c_sort[2])
b_pxl = np.asarray([-1, -1])
for c_id in clusters_id:
    b_pxl_l = np.where(np.isin(kmeans.labels_, c_id))[0]
    b_pxl = np.hstack((b_pxl, b_pxl_l))
    
b_pxl_2 = b_pxl[2:]
tempidx = b_pxl_2 // num_pxl
for r in np.unique(tempidx):
    b_idx = np.where( tempidx == (r % kk) )
    b_0[kk*j+r, (b_pxl_2[b_idx] % num_pxl) ] = 5


x = b_0.astype(int)
x = x.flatten().astype(int)
y = np.bincount(x)
ii = np.nonzero(y)[0]
np.vstack((ii,y[ii])).T

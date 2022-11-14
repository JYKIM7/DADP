import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import pandas as pd
import os
input_dir = 'C:/Users/JYKIM/Desktop/Student_Kim_220915/DADP/input/thickness_map2'
os.chdir(input_dir)

######################################################
### Load data
######################################################

# 1. Check dataframe.
df = pd.read_pickle('thickness_results.pkl')

# Check data variables
for col in df.columns:
    print(col)
    
pd.set_option('display.max_columns', 6)
df.head()    

# We have two cartilage_type. so 
df_femoral = df[df['cartilage_type']=='femoral']
sub_id_unique = np.unique(df_femoral['patient_id'])
df_femoral = df_femoral.replace('ENROLLMENT','0_ENROLLMENT')
df_femoral.sort_values(['patient_id','timepoint'],ascending=True).head()
df_femoral.head(8)

df_tibial = df[df['cartilage_type']=='tibial']
sub_id_unique = np.unique(df_tibial['patient_id'])
df_tibial = df_tibial.replace('ENROLLMENT','0_ENROLLMENT')
df_tibial.sort_values(['patient_id','timepoint'],ascending=True).head()


# 2. Load image
dt1 = np.load('thickness_results_femoral_cartilage.npz')
dt1_1 = np.nan_to_num(dt1['data'])
# dt1_1.shape
idxx = df_femoral.sort_values(['patient_id','timepoint'],ascending=True)['cartilage_type_id'].astype(int)
dt1_1 = dt1_1[idxx.argsort(),:,:]
# dt1_1 = dt1_1[idxx.argsort(),:,:].astype(int)
dt1_1_del = np.delete(dt1_1[:], -1, axis=2)

dt2 = np.load('thickness_results_tibial_cartilage.npz')
dt2_1 = np.nan_to_num(dt2['data'])
idxx = df_tibial.sort_values(['patient_id','timepoint'],ascending=True)['cartilage_type_id'].astype(int)
dt2_1 = dt2_1[idxx.argsort(),:,:]


######################################################
### Make template file (=mask)
######################################################

temp=np.ones(shape=(dt1_1_del.shape[1],dt1_1_del.shape[2]) )
temp
for i in range(dt1_1.shape[1]):
    for j in range(dt1_1.shape[1]):
        if np.sum(dt1_1[:,i,j]) < 1:
            temp[i][j] = 0
        else:
            temp[i][j] = 1
mask1 = temp.astype(int)
plt.imshow(mask1)

temp=np.ones(shape=(dt2_1.shape[1],dt2_1.shape[2]) )
temp
for i in range(dt2_1.shape[1]):
    for j in range(dt2_1.shape[1]):
        if np.sum(dt2_1[:,i,j]) < 1:
            temp[i][j] = 0
        else:
            temp[i][j] = 1
mask2 = temp.astype(int)
plt.imshow(mask2)

######################################################
### NMF
######################################################

# Construct the average disease map for each subject
df = pd.read_pickle('thickness_results.pkl')
df_femoral = df[df['cartilage_type']=='femoral']
sub_id_unique = np.unique(df_femoral['patient_id'])
m = len(sub_id_unique)
m_v = mask1.shape[0]*mask1.shape[1]
A = np.zeros( shape=(m_v, m) )

aa=0
for ll in range(m):
    time_len = len(df_femoral[df_femoral['patient_id'] == sub_id_unique[ll]]['cartilage_type_id'])
    d_map_mean = np.mean( dt2_1[range(aa, aa + time_len -1),:,:] , axis=0)
    # plt.imshow( d_map_mean )
    A[:,ll] = d_map_mean.flatten()
    aa = aa + 1

plt.imshow(np.reshape(A[:,311], (mask1.shape[0],  mask1.shape[1]) ) )

from sklearn.decomposition import NMF
K = 12  # number of clusters
model = NMF(n_components = K, init='random')
AA = A
W = model.fit_transform(AA)
H = model.components_ # W.shape, H.shape
new_img = np.dot(W,H[:,311]) # new_img.shape
plt.imshow(np.reshape(new_img, (mask1.shape[0],  mask1.shape[1]) ) )
# plt.imshow(np.reshape(new_img, (mask1.shape[0],  mask1.shape[1]) ).astype(int))


template = mask1
vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
ind_list = np.where(vec_template == 1)[0]
ddd = np.zeros(shape=(template.shape[0] * template.shape[1] , 1))
bbb = pd.DataFrame({'value': new_img.flatten()[ind_list]}, index=ind_list)

k = 5
quantiles = pd.qcut(bbb['value'], k, labels=False).astype(int) + 1
bbb = bbb.assign(quantile = quantiles.values) 
ddd[ind_list] = np.reshape(np.array(bbb['quantile']), (len(ind_list), 1))
eee = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a')
plt.imshow(eee)

quantiles = pd.qcut(bbb['value'], k, labels=False).astype(int) + 1
bbb = bbb.assign(quantile = quantiles.values)
bbb = bbb.replace(np.unique(bbb['quantile'].values)[:-1], 1)
ddd[ind_list] = np.reshape(np.array(bbb['quantile']), (len(ind_list), 1))
eee = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a')
plt.imshow(eee)

# Construct the average disease map for each subject
df = pd.read_pickle('thickness_results.pkl')
df_tibial = df[df['cartilage_type']=='tibial']
sub_id_unique = np.unique(df_tibial['patient_id'])
m = len(sub_id_unique)
m_v = mask1.shape[0]*mask1.shape[1]
A = np.zeros( shape=(m_v, m) )

aa=0
for ll in range(m):
    time_len = len(df_tibial[df_tibial['patient_id'] == sub_id_unique[ll]]['cartilage_type_id'])
    # d_map_mean = np.mean( dt2_1[range(aa, aa + time_len -1),:,:] , axis=0)
    d_map_mean = np.mean( dt2_1[range(aa, aa + time_len -1),:,:] , axis=0)
    # plt.imshow( d_map_mean )
    A[:,ll] = d_map_mean.flatten()
    aa = aa + 1

plt.imshow(np.reshape(A[:,311], (mask1.shape[0],  mask1.shape[1]) ) )

from sklearn.decomposition import NMF
K = 12  # number of clusters
model = NMF(n_components = K, init='random')
AA = A
W = model.fit_transform(AA)
H = model.components_ # W.shape, H.shape
new_img = np.dot(W,H[:,311]) # new_img.shape
plt.imshow(np.reshape(new_img, (mask2.shape[0],  mask2.shape[1]) ) )
# plt.imshow(np.reshape(new_img, (mask1.shape[0],  mask1.shape[1]) ).astype(int))


template = mask2
vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
ind_list = np.where(vec_template == 1)[0]
ddd = np.zeros(shape=(template.shape[0] * template.shape[1] , 1))
bbb = pd.DataFrame({'value': new_img.flatten()[ind_list]}, index=ind_list)

k = 6
quantiles = pd.qcut(bbb['value'], k, labels=False).astype(int) + 1
bbb = bbb.assign(quantile = quantiles.values) 
ddd[ind_list] = np.reshape(np.array(bbb['quantile']), (len(ind_list), 1))
eee = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a')
plt.imshow(eee)

quantiles = pd.qcut(bbb['value'], k, labels=False).astype(int) + 1
bbb = bbb.assign(quantile = quantiles.values)
bbb = bbb.replace(np.unique(bbb['quantile'].values)[:-1], 1)
ddd[ind_list] = np.reshape(np.array(bbb['quantile']), (len(ind_list), 1))
eee = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a')
plt.imshow(eee)

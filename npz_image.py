import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import pandas as pd
import os
input_dir = '~~~~~~'
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
        if np.sum(dt1_1[:,i,j].astype(int)) < 1:
            temp[i][j] = 0
        else:
            temp[i][j] = 1
mask1 = temp
plt.imshow(mask1)

temp=np.ones(shape=(dt2_1.shape[1],dt2_1.shape[2]) )
temp
for i in range(dt2_1.shape[1]):
    for j in range(dt2_1.shape[1]):
        if np.sum(dt2_1[:,i,j].astype(int)) < 1:
            temp[i][j] = 0
        else:
            temp[i][j] = 1
mask2 = temp
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
    # d_map_mean = np.mean( dt2_1[range(aa, aa + time_len -1),:,:] , axis=0)
    d_map_mean = np.mean( dt1_1_del[range(aa, aa + time_len -1),:,:] , axis=0)
    # plt.imshow( d_map_mean )
    A[:,ll] = d_map_mean.flatten()

# NMF
from sklearn.decomposition import NMF
K = 10  # number of clusters
model = NMF(n_components = K, init='random', random_state=1)
W = model.fit_transform(A)
H = model.components_ # W.shape, H.shape
I = np.argsort(H, axis=0) # I.shape
label = I[I.shape[0]-1,:]

import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import pandas as pd
import os
input_dir = '~~~'
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
def template_fun(data):
    temp=np.ones(shape=(data.shape[1],data.shape[2]) )
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if np.sum(data[:,i,j]) < 1:
                temp[i][j] = 0
            else:
                temp[i][j] = 1
    return temp

mask1 = template_fun(dt1_1_del)
plt.imshow(mask1)

mask2 = template_fun(dt2_1)
plt.imshow(mask2)


######################################################
######################################################

def full_to_mask(dt, template):
    vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
    ind_list = np.where(vec_template == 1)[0]
    df_maskinfo = pd.DataFrame({'value': dt.flatten()[ind_list] , 'index': ind_list} )
    return df_maskinfo

def mask_to_full(dt_value, dt_index, template):
    ddd = np.zeros(shape=(template.shape[0] * template.shape[1] , 1))
    ddd[dt_index] = np.reshape(np.array(dt_value), (len(dt_value), 1))
    full_array = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a')
    return full_array

dt = dt1_1_del[1,:,:]
template = mask1
full_to_mask(dt, template)

d_value = full_to_mask(dt, template)['value']
d_index = full_to_mask(dt, template)['index']
plt.imshow( mask_to_full(d_value, d_index, template) )

#############################################################
# Construct the average disease map for each subject
df = pd.read_pickle('thickness_results.pkl')
df = df[df['cartilage_type']=='femoral']
dt = dt1_1_del

dtt = np.zeros(shape = dt.shape) 
temp = template_fun(dt1_1_del)
np.unique(temp.flatten())
ii=0
for ii in range(dt.shape[0]):
    tem = temp
    idx = np.where(dt[ii,:,:] > np.quantile(dt[ii,:,:], 0.8))
    tem[idx] = 1
    idx2 = np.where(dt[ii,:,:] > np.quantile(dt[ii,:,:], 0.9))
    tem[idx2] = 2
    dtt[ii,:,::] = tem
    # plt.imshow(temp) 
    print(ii)

np.unique(dtt.flatten())
plt.imshow(dtt[0,:,:]) 



mask1 = template_fun(dt1_1_del)
template = mask1
sub_id_unique = np.unique(df['patient_id'])
m = len(sub_id_unique)
m_v = len(np.where(template.flatten() == 1)[0])
A = np.zeros( shape=(m_v, m) ) # A.shape

aa = 0
for ll in range(m):
    time_len = len(df[df['patient_id'] == sub_id_unique[ll]]['cartilage_type_id'])
    d_map_mean = np.mean( dtt[range(aa, aa + time_len -1),:,:] , axis=0)
    A[:, ll] = full_to_mask(d_map_mean, template)['value'] # d_map_mean.shape
    aa = aa + 1

plt.imshow( mask_to_full(A[:,1], np.where(template.flatten() == 1)[0], template) )
# plt.imshow( dt[0,:,:])


from sklearn.decomposition import NMF
K = 3 # number of clusters
model = NMF(n_components = K,max_iter=300)
AA = A - 1
W = model.fit_transform(AA) # (60795, 3)
H = model.components_       # (3, 351)

H[:,0:4]

I = np.argsort(H, axis=0) # I.shape
label = I[I.shape[0]-1,:] # same as H.argmax(axis=0) 

ii=1 # ii=1:K
for ii in range(np.size(H, 0)):
	idk = np.where(label==ii)
    # roi = np.zeros(shape=(310,310))
	globals()["roi_{}".format(ii)] = W[:,ii]*np.mean(H[ii,idk])


plt.imshow( mask_to_full(roi_0, np.where(template.flatten() == 1)[0], template) )
plt.colorbar()
plt.imshow( mask_to_full(roi_1, np.where(template.flatten() == 1)[0], template) )
plt.colorbar()
plt.imshow( mask_to_full(roi_2, np.where(template.flatten() == 1)[0], template) )
plt.colorbar()

roi = roi_0+roi_1+roi_2
plt.imshow( mask_to_full(roi, np.where(template.flatten() == 1)[0], template) )


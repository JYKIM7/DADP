y_design_1[0,:]
y_design_1[1,:]
y_design_1[2,:]
y_design_1[3,:]

ccc = y_design_1[0,:]
ccc = y_design_1[1,:]
ccc = y_design_1[2,:]
ccc = y_design_1[3,:]
ccc = y_design_1[4,:]

ccc = b_0[0,:]
ccc = b_0[1,:]
ccc = b_0[2,:]
ccc = b_0[3,:]

ccc = b0_i

ccc = b_11[0,:]

ccc= np.argmin(u, axis=1)

xxx = ccc.astype(int)
np.unique(xxx)
np.bincount(xxx)

template = mask
vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
ind_list = np.where(vec_template == 1)[0]
ddd = np.zeros(shape=(template.shape[0] * template.shape[1] , 1))
bbb = pd.DataFrame({'value': ccc}, index=ind_list)
ddd[ind_list] = np.reshape(np.array(bbb.value.astype(int)), (num_pxl, 1))
eee = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a').astype(int)
plt.imshow(eee)


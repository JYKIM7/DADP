import numpy as np

def map_fun(b_0, x_design, y_design, dx, template, beta, mu, s2, gamma, nclasses, map_iter):
    y_design1 = y_design[dx > 0, :]
    x_design1 = x_design[dx > 0, :]
    res_mat = y_design1 - dot(x_design1, beta)
    num_sub = res_mat.shape[0]
    num_vox = res_mat.shape[1]

    for ii in range(num_sub):
        r_i = res_mat[ii, :]
        mu_i = mu[ii, :]
        s2_i = s2[ii]
        b0_i = b_0[ii, :]
        u = 0
        for jj in range(map_iter):
            u1 = zeros(num_vox, nclasses)
            u2 = zeros(num_vox, nclasses)
            for ll in range(nclasses):
                temp_i = np.log( (r_i - np.repeat( np.mean(mu_i) , r_i.size ) )**2/s2_i/2 ) + np.log(s2_i)/2
                u1[:, ll] = u1[:, l]+temp_i
                # b0_i.shape
                # ind2_1.shape
                for vii in range(num_vox):
                    u3 = np.zeros(num_vox)
                    if len(template.shape) == 2:
                        vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
                    ind_list = np.where(vec_template == 1)[0]
                
                    ind = ind_list[vii]
                    sub2 = ind2sub_v2(template.shape, np.array(ind))
                    ind2_1 = sub2ind_v2(template.shape, [sub2[0] + 1, sub2[1]])
                    ind2_2 = sub2ind_v2(template.shape, [sub2[0], sub2[1] + 1])
                    ind2_3 = sub2ind_v2(template.shape, [sub2[0] - 1, sub2[1]])
                    ind2_4 = sub2ind_v2(template.shape, [sub2[0], sub2[1] - 1])
                    
                    b_0_new = pd.DataFrame(b0_i.T, index=ind_list)
                    b_0_new.columns = ['value']
                    
                    if ind2_1 in ind_list: 
                        u3_1 = delta_fun(ll, b_0_new.loc[ind2_1].value.astype(int))
                    else:   
                        u3_1 = delta_fun(ll, 0)
                    if ind2_2 in ind_list: 
                        u3_2 = delta_fun(ll, b_0_new.loc[ind2_2].value.astype(int))
                    else:   
                        u3_2 = delta_fun(ll, 0)
                    if ind2_3 in ind_list: 
                        u3_3 = delta_fun(ll, b_0_new.loc[ind2_3].value.astype(int))
                    else:   
                        u3_3 = delta_fun(ll, 0)
                    if ind2_4 in ind_list: 
                        u3_4 = delta_fun(ll, b_0_new.loc[ind2_4].value.astype(int))
                    else:   
                        u3_4 = delta_fun(ll, 0)
                    
                    u3[vii] = -(u3_1 + u3_2 + u3_3 + u3_4)
                
                #else:
                u2[vii, ll] = np.sum(u3)   
                
        u = u1 + u2*gamma
            b0_i = np.argmin(u, axis=1)-1

        if len(template.shape) == 2:
            vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
        else:
            vec_template = np.reshape(template, (template.shape[0] * template.shape[1]
                                                 * template.shape[2], 1))
        # b_0[i, vec_template == 1] = b0_i
        b_0[ii, :] = b0_i
        u_i = np.exp(-np.min(u, axis=1))
        u_i0 = u_i[b0_i == 0]
        r_i0 = r_i[b0_i == 0]
        mu[ii, 0] = np.sum(u_i0*r_i0)/np.sum(u_i0)
        u_i1 = u_i[b0_i == 1]
        r_i1 = r_i[b0_i == 1]
        mu[ii, 1] = np.sum(u_i1 * r_i1) / np.sum(u_i1)

    """return b_0"""
    return b_0

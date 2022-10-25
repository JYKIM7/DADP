"""
# Step 1: Linear Mixed Model across pixels for normal subjects. 
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `Y`: 2D array n_obs x num_pxl including thickness map.
#  - `X`: 2D array n_obs x n_cov including predictor information.
#  - `var_list`: list of predictor names.
#  - `fe_idx`: index list of predictors in fixed effect.
#  - `re_idx`: index list of predcitors in random effects.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `betas_hat`: The parameter vector (beta,sigma2,vech(D_1),...,vech(D_r))
#  - `pvalues`: Estimates of the random effects vector, b.

# Author: Chao Huang (chaohuang.stat@gmail.com)
# Last update: 2022-09-10
"""


import numpy as np
# import scipy as sp
import pandas as pd
# import statsmodels.api as sm
import statsmodels.formula.api as smf

"""
# installed all the libraries above
"""

def lmm_fun_v3(Y, X, var_list, fe_idx, re_idx):
   
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

    return(betas_hat, pvalues, s2)


"""
def lmm_fun(Y, X, var_list, fe_idx, re_idx):
    n_obs, num_pxl = np.shape(Y)
    p = len(fe_idx) + 1
    betas_hat = np.zeros((p, num_pxl))
    pvalues = np.zeros((p, num_pxl))
    column_names = var_list.append('thickness')
    mdl_str = 'thickness ~ '
    for j in fe_idx:
        mdl_str = mdl_str + var_list[j] + ' +'
    mdl_str = mdl_str[:-2]
    re_str = '~ '
    for j in re_idx:
        re_str = re_str + var_list[j] + ' +'
    re_str = re_str[:-2]
    for k in np.arange(num_pxl):
        y_k = np.reshape(Y[:, k], (-1, 1))
        array_k = np.hstack((X, y_k))
        df_k = pd.DataFrame(array_k, columns=column_names)
        # Run LMER at k-th pixel
        md = smf.mixedlm(mdl_str, df_k, groups=df_k[var_list[0]], re_formula=re_str)
        mdf = md.fit(method=["lbfgs"])
        params_hat = mdf.params.to_numpy()
        betas_hat[:, k] = params_hat[:p]
        pvalues[:, k] = mdf.params.to_numpy()
        
    return(betas_hat, pvalues)
    
"""
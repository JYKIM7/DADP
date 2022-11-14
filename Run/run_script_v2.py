"""
Run main script: Dynamic Abnormality Detection and Progression (DADP) pipeline
Usage: python ./DADP_run_script.py ./data/ ./result/

Author: Chao Huang (chaohuang.stat@gmail.com)
"""

import os
import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import glob
from utilfncs import sub2ind_v2, ind2sub_v2, delta_fun

# from lmm import lmm_fun
from lmm_v3 import lmm_fun_v3
# from initial_b import initial_b_fun
from initial_b_v3 import initial_b_fun_v3
from CMRF_MAP import map_fun

"""
installed all the libraries above
"""


def run_script(input_dir, output_dir):
    """
    Run the commandline script for DADP.

    :param
        input_dir (str): full path to the data folder
        output_dir (str): full path to the output folder
    """

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 0. load dataset """)
    print("+++++++Read the info file+++++++")
    # 5 variables: subject id, diagnostic status (0: NC, 1: diseased), gender, bmi, and age
    info_name = input_dir + 'info.txt'
    info = np.loadtxt(info_name)
    obs_id = info[:, 0]
    sub_id = np.unique(obs_id)
    num_sub = len(sub_id)
    print("The number of subjects is " + str(num_sub))
    
    dx = info[:, 1]
    obs_id_0 = obs_id[dx == 0]
    print("The number of normal observations is " + str(len(obs_id_0)))
    
    obs_id_1 = obs_id[dx > 0]
    print("The number of diseased observations is " + str(len(obs_id_1)))
    
    x_design = info[:, [0, 2, 3, 4]]
    x_design[:, 0] = info[:, 0]                    # subject id
    x_design[:, 1] = info[:, 2]                    # gender
    x_design[:, 2] = stats.zscore(info[:, 3])      # normalized bmi
    x_design[:, 3] = stats.zscore(info[:, 4])      # normalized age
    var_list = ['subject_id', 'gender', 'bmi', 'age']
    
    print("+++++++Display one randomly selected abnormal pattern+++++++")
    bmap_name = input_dir + 'bmap/b_500.txt'
    bmap = np.loadtxt(bmap_name).astype(int)
    plt.imshow(bmap)
    
    print("+++++++Read the mask file+++++++")
    mask_name = input_dir + 'mask.txt'
    mask = np.loadtxt(mask_name).astype(int)
    print("The image size is " + str(mask.shape))
    num_pxl = len(mask[mask == 1])
    print("The number of pixels in the mask is " + str(num_pxl))
    
    # Modified code - re-sort
    print("+++++++Read the thickness maps+++++++")
    img_folder_name = input_dir + "thickness_map"
    img_names = glob.glob("%s/map_*.txt" % img_folder_name)
    import re
    img_names = sorted(img_names, key=lambda x:float(re.findall("(\d+)",x)[1]))
    num_img = len(img_names)
    img_data = np.zeros(shape=(num_img, num_pxl))

    for ii in range(num_img):
        img_ii = np.loadtxt(img_names[ii])
        img_data[ii, :] = img_ii[mask == 1]
    print("The matrix dimension of image data is " + str(img_data.shape))
    
    # Modified code - lmm_fun_v3
    print("+++++++Step 1: Set up linear mixed model on all voxels from normal subjects+++++++")
    start_1 = time.time()
    y_design_0 = img_data[dx == 0, :]
    x_design_0 = x_design[dx == 0, :]
    fe_idx = [1, 2, 3]
    # re_idx = [1, 2, 3] original code.
    re_idx = [2, 3] 
    [betas_hat, pvalues] = lmm_fun_v3(y_design_0, x_design_0, var_list, fe_idx, re_idx)
    stop_1 = time.time()
    print("The cost time in Step 1 is %(t1) d" % {"t1": stop_1 - start_1})
    

    print("+++++++Step 2: Initial the diseased regions from patients (k-means)+++++++")
    start_2 = time.time()
    y_design_1 = img_data[dx > 0, :]
    x_design_1 = x_design[dx > 0, :]
    res_mat = y_design_1-np.dot(x_design_1, betas_hat)
    sub_id_1 = x_design_1[:, 0]
    idx = np.nonzero(mask)
    #landmarks = ind2sub(np.shape(mask), np.asarray(idx))
    landmarks = np.nonzero(mask)
    b_0 = initial_b_fun_v2(res_mat, sub_id_1, mask, landmarks, idx)
    #[b_0, mu] = initial_b_fun_v2(res_mat, sub_id_1, mask, landmarks, idx)
    # mu typos.
    stop_2 = time.time()
    print("The cost time in Step 2 is %(t1) d" % {"t1": stop_2 - start_2})
    

    
    """+++++++++++++++++++++++++++++++++++"""
    print("""Step 2. Diseased region detection based on HMRF and EM""")
    start_2 = time.time()
    em_iter = 10
    map_iter = 10
    gamma = 0.2  # smooth parameter
    for out_it in range(em_iter):
        # update b via MAP algorithm
        b_0, mu = map_fun(b_0, x_design, y_design, dx, template, beta, mu, s2, gamma, nclasses, map_iter)
    b = b_0
    stop_2 = time.time()
    print("The cost time in Step 2 is %(t2) d" % {"t2": stop_2 - start_2})

    """+++++++++++++++++++++++++++++++++++"""
    print("""Step 3. Save the detected regions into image""")
    template = mask
    for ii in range(b.shape[0]):
        vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
        ind_list = np.where(vec_template == 1)[0]
        ddd = np.zeros(shape=(template.shape[0] * template.shape[1] , 1))
        bbb = pd.DataFrame({'value': ccc}, index=ind_list)
        ddd[ind_list] = np.reshape(np.array(bbb.value.astype(int)), (num_pxl, 1))
        eee = np.reshape(ddd, (template.shape[0],  template.shape[1]), order='a').astype(int)
        output_file_name = output_dir + 'b_%s.img' % ii
        np.savetxt(output_file_name, eee, delimiter=',')
        
    """
    for ii in range(b.shape[0]):
        b_i = np.reshape(b[ii, :], template.shape)
        ana_img = nib.AnalyzeImage(b_i, np.eye(4))
        output_file_name = output_dir + 'b_%s.img' % ii
        nib.save(ana_img, output_file_name)

    if __name__ == '__main__':
        input_dir0 = sys.argv[1]
        output_dir0 = sys.argv[2]

        start_all = time.time()
        run_script(input_dir0, output_dir0)
        stop_all = time.time()
        delta_time_all = str(stop_all - start_all)
        print("The total elapsed time is " + delta_time_all)
      """

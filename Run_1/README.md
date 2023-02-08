# Case 1.

### 0. Read thickness maps.
When reading the thickness maps, we should check the order of the file names properly. If it's misaligned, this can lead to incorrect results when calculating array or matrix later. 


### 1. Linear Mixed Model across pixels for normal subjects. 
```python
lmm_fun(y_design_0, x_design_0, var_list, fe_idx, re_idx)
```
 + y_design_0 = Thickness map array for normal subjects.
 + x_design_0 = Information array for normal subjects.
 + var_list   = list of variables.
 + fe_idx     = index list of variables in fixed effect.
 + re_idx     = index list of variables in random effect.
 
 
This function calculates the parameters of the Linear Mixed Model based on the image data and information data of normal subjects.


### 2. Initial the diseased regions from patients (k-means).
```python
initial_b_fun(res_mat, sub_id_1, mask, landmarks, idx)
```
 + y_design_1     = Thickness map array for abnormal subjects.
 + x_design_1     = Information array for abnormal subjects.
 + res_mat        = Residual matrix about abnormal subjects by using step 1 parameters
 + sub_id_1       = Abnormal subjects ID without duplication.
 + mask           = Template data
 + landmarks, idx = Index infomation about masked region.

![image](https://user-images.githubusercontent.com/71793706/211867980-0cf9d712-cfa5-4781-b862-d1e0d3dfee4b.png)

This function calculates b_0 parameters. (Their elements is 2(normal) or 5(abnormal).)




### 3. Estimating the hidden variables, b_i(v), based on MRF-MAP.
```python
map_fun(b_0, x_design, y_design, dx, template, beta, mu, s2, gamma, nclasses, map_iter)
```

Diseased region detection
![12123](https://user-images.githubusercontent.com/71793706/217572502-d86e2ae9-ff7f-46e6-ad30-74f5c529c547.png)

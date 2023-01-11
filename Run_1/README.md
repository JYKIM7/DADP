# Case 1.

### 1. Linear Mixed Model across pixels for normal subjects. 
```python
lmm_fun(y_design_0, x_design_0, var_list, fe_idx, re_idx)
```
 + y_design_0 = Thickness map array for normal subjects.
 + x_design_0 = Information array for normal subjects.
 + var_list = list of variables.
 + fe_idx = index list of variables in fixed effect.
 + re_idx = index list of variables in random effect.
 
 
This function calculates the parameters of the Linear Mixed Model based on the image data and information data of normal subjects.


### 2. Initial the diseased regions from patients (k-means).
```python
initial_b_fun(res_mat, sub_id_1, mask, landmarks, idx)
```
 + res_mat = 
 + sub_id_1 = 
 + mask = 
 + landmarks = 
 + idx = 



### 3. Estimating the hidden variables, b_i(v), based on MRF-MAP.
```python
map_fun(b_0, x_design, y_design, dx, template, beta, mu, s2, gamma, nclasses, map_iter)
```

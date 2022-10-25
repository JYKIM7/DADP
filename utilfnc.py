def ind2sub_v2(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = np.trunc(ind.astype('int') / array_shape[1]).astype(int)
    cols = ind % array_shape[1]
    return (rows, cols)
  
  
def sub2ind_v2(array_shape, sub):
    if len(array_shape) == 2:
        ind = np.array(sub[0] * array_shape[1] + sub[1]).astype(int)
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind

def delta_fun(x1, x2):
    if x1 == x2:
        z = 1
    else:
        z = 0
    return z

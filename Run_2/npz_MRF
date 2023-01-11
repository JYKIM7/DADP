import pylab as pl
import numpy as np

def MRF(I, J, eta = 1.5, zeta=1.0):
    ind =np.arange(np.shape(I)[0])
    np.random.shuffle(ind)
    orderx = ind.copy()
    np.random.shuffle(ind)

    for i in orderx:
        for j in ind:
            oldJ = J[i,j]
            J[i,j] = 1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
            energya = -eta*np.sum(I*J) - zeta*patch
            J[i,j] = -1 # J[i,j] = -1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
            energyb = -eta*np.sum(I*J) - zeta*patch
            if energya<energyb:
                J[i,j] = 1
            else:
                J[i,j] = -1 # J[i,j] = -1
    return J

# ignore region 

I = dt[122,:,:] 

I = dt[122,:,:].astype(int) 
# np.mean(I)
# I = full_to_mask(dt[0,:,:], mask1)['value']
N = np.shape(I)[0]
# I = I[:,:,0]
# I = np.where(I<0.1,-1,1)

pl.title('Original Image')
pl.imshow(I)
# pl.imshow( mask_to_full(I, np.where(template.flatten() == 1)[0], template) )


np.median(noise)
noise = np.random.rand(N,N)
J = I.copy()
ind = np.where(noise < 1 )
J[ind] = -J[ind]
# pl.figure()
pl.title('Noisy image')
pl.imshow(J)
# pl.imshow(mask_to_full(J, np.where(template.flatten() == 1)[0], template))

import cv2
import numpy as np
from scipy import sparse

def gen_pencil_texture(J):
    H= cv2.imread('pencils/pencil0.jpg')
    H=cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)*(1/255.0)

    height = int(J.shape[0])
    width = int(J.shape[1] )
    l = 0.2
    # Adjust the input to correspond
    H_res = cv2.resize(H, [width,height])

    H_f=H_res.flatten()
    logH = np.log(H_f)

    J_f= J.flatten()
    logJ = np.log(J_f)
    
    # In order to use Conjugate Gradient method we need to prepare some sparse matrices:
    logH_sparse = sparse.spdiags(logH, 0, height*width, height*width) # 0 - from main diagonal

    e = np.ones((height * width, 1))
    ee = np.concatenate((-e,e), axis=1)

    diags_x = [0, height*width]
    diags_y = [0, 1]

    dx = sparse.spdiags(ee.T, diags_x, height*width, height*width)

    dy = sparse.spdiags(ee.T, diags_y, height*width, height*width)
    

    A =  l*((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
    b = logH_sparse.T @ logJ
    
    # Conjugate Gradient
    beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)
    
    # Adjust the result
    beta_reshaped = np.reshape(beta[0], (height, width))
    
    # The final pencil texture map T
    T = np.power(H_res, beta_reshaped)

    
    return T
    

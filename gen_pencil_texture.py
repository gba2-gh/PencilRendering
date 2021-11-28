import cv2
import numpy as np
from scipy import sparse

def gen_pencil_texture(img, H, J):
    # define the regularization parameter:
    lamda = 0.2
    height = img.shape[0]
    width = img.shape[1]
    # Adjust the input to correspond
#     H_res = transform.resize(H,(height, width))
    H_res = cv2.resize(H, (width, height), interpolation=cv2.INTER_CUBIC)
    H_res_reshaped = np.reshape(H_res, (height * width, 1))
    logH = np.log(H_res_reshaped)
    
#     J_res = transform.resize(J,(height, width))
    J_res = cv2.resize(J, (width, height), interpolation=cv2.INTER_CUBIC)
    J_res_reshaped = np.reshape(J_res, (height * width, 1))
    logJ = np.log(J_res_reshaped)
    
    # In order to use Conjugate Gradient method we need to prepare some sparse matrices:
    logH_sparse = sparse.spdiags(logH.ravel(), 0, height*width, height*width) # 0 - from main diagonal
    e = np.ones((height * width, 1))
    ee = np.concatenate((-e,e), axis=1)
    diags_x = [0, height*width]
    diags_y = [0, 1]
    dx = sparse.spdiags(ee.T, diags_x, height*width, height*width)
    dy = sparse.spdiags(ee.T, diags_y, height*width, height*width)
    
    # Compute matrix X and b: (to solve Ax = b)
    A = lamda * ((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
    b = logH_sparse.T @ logJ
    
    # Conjugate Gradient
    beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)
    
    # Adjust the result
    beta_reshaped = np.reshape(beta[0], (height, width))
    
    # The final pencil texture map T
    T = np.power(H_res, beta_reshaped)
    
    return T
    

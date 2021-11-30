import cv2
import numpy as np
from scipy import sparse

def gen_pencil_texture(H,J, l=0.2):
    height = int(J.shape[0])
    width = int(J.shape[1] )

    # Ajustar textura al tamaño de la image
    H = cv2.resize(H, [width,height])
    #flatten
    H_f=H.flatten()
    logH = np.log(H_f)

    J_f= J.flatten()
    logJ = np.log(J_f)
    
    #metriz dispersa
    logH_sparse = sparse.spdiags(logH, 0, height*width, height*width) 

    e = np.ones((height * width, 1))
    ee = np.concatenate((-e,e), axis=1)
    diags_x = [0, height*width]
    diags_y = [0, 1]

    dx = sparse.spdiags(ee.T, diags_x, height*width, height*width)
    dy = sparse.spdiags(ee.T, diags_y, height*width, height*width)
    
    #Matrices A y B
    A =  l*((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
    b = logH_sparse.T @ logJ
    
    # Método gradiente conjugado
    beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)
    
    # Redimensionar 
    beta_r = np.reshape(beta[0], (height, width))
    
    # T=H^beta
    T = np.power(H, beta_r)

    return T
    

import cv2
import math
import numpy as np
from strokes_funct import edge_funct, strokes_funct
from gen_pencil_texture import gen_pencil_texture
from hist_match import hist_match
from cal_tone_map import cal_tone_map
from matplotlib import pyplot as plt
from skimage import filters, transform, io, color, data, exposure
from scipy import signal, sparse

cv2.saliency


image = io.imread('inputs/7--129.jpg')
#ex_img_yuv = color.rgb2yuv(ex_img)
#ex_img_y_ch = ex_img_yuv[:,:,0]

##input
#image = cv2.imread('inputs/2--59.jpg')
scale_percent = 100 # percent of original size
height = int(image.shape[0] * scale_percent / 100)
width = int(image.shape[1] * scale_percent / 100)

dim = (width, height)

image= cv2.resize(image, dim) #redimensionar
image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) ##grises

gray=gray.astype(float)
gray=gray/255.0

#gray=filters.gaussian(gray,0.2, truncate=2)

##STROKE 
####0=gradiente, 1=canny, 2=sobel
edges=edge_funct(gray,method=0)

strokes=strokes_funct(edges)

#plt.imshow(strokes, cmap='gray')
#plt.show()

gray=gray*(1/255.0)

p=cal_tone_map()
J=hist_match(gray,p)

H= cv2.imread('pencils/pencil0.jpg')
H=cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)

plt.imshow(J, cmap='gray')
plt.show()
#plt.imshow(J_b, cmap='gray')
#plt.show()

#########################################################################################################

H= cv2.imread('pencils/pencil0.jpg')
H=cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)*(1/255.0)
#T=gen_pencil_texture(gray,H,J)

#Lambda
l = 0.2
# Adjust the input to correspond
H_res = cv2.resize(H, dim)
#H_res_reshaped = np.reshape(H_res, (height * width, 1))

H_f=H_res.flatten()
logH = np.log(H_f)

#test = np.power(H_res, 0.2)
#plt.imshow(test, cmap='gray')
#plt.show()

#print('H_f=\n ', H_f)    
#print('logH=\n ', logH)    
#J_res_reshaped = np.reshape(J, (height * width, 1))
J_f= J.flatten()
logJ = np.log(J_f)
    
# In order to use Conjugate Gradient method we need to prepare some sparse matrices:
#print('ravel=\n ', logH.ravel())
logH_sparse = sparse.spdiags(logH, 0, height*width, height*width) # 0 - from main diagonal
#print('logHarr=\n',logH_sparse.toarray())

e = np.ones((height * width, 1))
print('e=\n',e)

ee = np.concatenate((-e,e), axis=1)
print('ee=\n',ee)

diags_x = [0, height*width]
print('diags_x=\n',diags_x )
diags_y = [0, 1]
print('diags_y=\n',diags_y )


ddx = sparse.spdiags(e.T, 0, height*width, height*width)
#print('ddxarr=\n',ddx.toarray() )

dddx=sparse.identity(height*width)
#print('dddxarr=\n',dddx.toarray() )

dx = sparse.spdiags(ee.T, diags_x, height*width, height*width)
#print('dx=\n',dx )
#print('dx_arr=\n',dx.toarray())
dy = sparse.spdiags(ee.T, diags_y, height*width, height*width)
#print('dy=\n',dy )
#print('dy_arr=\n',dy.toarray())
    
# Compute matrix X and b: (to solve Ax = b)
#d1=dx @ dx.T
#d2=dy @ dy.T
#d3=logH_sparse.T @ logH_sparse

#print('d1=\n',d1 )
#print('d1arr=\n',d1.toarray() )
#print('d2=\n',d2 )
#print('d2array=\n',d2.toarray() )
#print('d3=\n',d3 )
#print('d3array=\n',d1.toarray() + d2.toarray() )

A =  l*((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
print('A=\n', A)
#print('Aarr =\n', A.toarray())
b = logH_sparse.T @ logJ
print('b=\n',b )
    
# Conjugate Gradient
beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)
    
# Adjust the result
beta_reshaped = np.reshape(beta[0], (height, width))
    
# The final pencil texture map T
T = np.power(H_res, beta_reshaped)
 



plt.imshow(T, cmap='gray')
plt.show()



saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")
saliencyMap2=np.sqrt(saliencyMap)
saliencyMap2=filters.gaussian(saliencyMap2,0.2, truncate=2)
plt.imshow(saliencyMap, cmap='gray')
plt.show()
plt.imshow(saliencyMap2, cmap='gray')
plt.show()




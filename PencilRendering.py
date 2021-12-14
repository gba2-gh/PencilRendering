import cv2
import math
import numpy as np
from gen_edge_strokes import gen_edge, gen_strokes
from gen_pencil_texture import *
from hist_match import *
from gen_tone_map import *
from get_gp import get_gp
from disp_hist import *

from matplotlib import pyplot as plt
from skimage import filters, transform, io, color, data, exposure
from scipy import signal, sparse
cv2.saliency


image = io.imread('inputs/uxmal.jpg')
#ex_img_yuv = color.rgb2yuv(ex_img)
#ex_img_y_ch = ex_img_yuv[:,:,0]

##input
#image = cv2.imread('inputs/2--59.jpg')
scale_percent = 100 # percent of original size
height = int(image.shape[0] * scale_percent / 100)
width = int(image.shape[1] * scale_percent / 100)

dim = (width, height)

image= cv2.resize(image, dim) #redimensionar
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) ##grises

gray=gray.astype(float)
gray=gray/255.0

#gray=filters.gaussian(gray,0.2, truncate=2)

##STROKE 
####0=gradiente, 1=canny, 2=sobel
edges=gen_edge(gray,method=2)
S=gen_strokes(edges, display=True)
plt.imshow(S, cmap='gray')
plt.show()

p=gen_tone_map()
J=hist_match(gray,p)

J_filt = filters.gaussian(J, sigma=np.sqrt(2))

plt.imshow(J, cmap='gray')
plt.title('J')
plt.show()

disp_hist(J_filt,'J_filt')


H= cv2.imread('pencils/pencil0.jpg')
H=cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)
H=H/255

T=gen_pencil_texture(H,J)
plt.imshow(T, cmap='gray')
plt.title('T')
plt.show()

F=T*S
plt.imshow(F, cmap='gray')
plt.title('F')
plt.show()
#########################################################################################################

#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
#(success, saliencyMap) = saliency.computeSaliency(image)
#saliencyMap = (saliencyMap * 255).astype("uint8")
#saliencyMap2=np.sqrt(saliencyMap)
#saliencyMap2=filters.gaussian(saliencyMap2,0.2, truncate=2)
#plt.imshow(saliencyMap, cmap='gray')
#plt.show()
#plt.imshow(saliencyMap2, cmap='gray')
#plt.show()




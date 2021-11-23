import cv2
import math
import numpy as np
from strokes_funct import strokes_funct
from matplotlib import pyplot as plt
from skimage import filters, transform
from scipy import signal



##input
image = cv2.imread('inputs/2--59.jpg')
scale_percent = 50 # percent of original size
height = int(image.shape[0] * scale_percent / 100)
width = int(image.shape[1] * scale_percent / 100)

dim = (width, height)

image= cv2.resize(image, dim) #redimensionar
image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) ##grises

gray=gray.astype(float)
gray=filters.gaussian(gray,0.2, truncate=2)

##STROKE 
####0=gradiente, 1=canny, 2=sobel
edges=strokes_funct(gray,method=2)

#Line segments
kernel_size= 7#int(height*(1/30))
kernel=np.zeros((kernel_size,kernel_size))

####Primera direccion
kernel[math.floor(kernel_size/2),:] = 1

## Convolucion en 8 direcciones
G=np.zeros((height, width, 8))
L=np.zeros((kernel_size,kernel_size,8))
for i in range(8):
    L[:,:,i]=transform.rotate(kernel,i*45)
    G[:,:,i] = signal.convolve2d(edges, L[:,:,i], mode='same')

 # Indices de elementos maximos
max_idx = np.argmax(G, axis=2)
##Formar clasificacion (C) con los elementos maximos de G
C=np.zeros((height, width, 8))
for i in range(8):
    #idx = max_idx==i #and
    C[:,:,i] = edges*(max_idx==i)


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax[0].imshow(C[:,:,0], cmap='gray')
ax[0].set_title('0', fontsize=20)
ax[1].imshow(C[:,:,1], cmap='gray')
ax[1].set_title('1', fontsize=20)
ax[2].imshow(C[:,:,2], cmap='gray')
ax[2].set_title('2', fontsize=20)
plt.show()

##Line shaping
out= np.zeros((height, width))
for i in range(8):
    out =out + signal.convolve2d(C[:,:,i], L[:,:,i], mode='same')

out=1-out

plt.imshow(out, cmap='gray')
plt.show()



plt.imshow(edges, cmap='gray')
plt.show()



cv2.waitKey(0)



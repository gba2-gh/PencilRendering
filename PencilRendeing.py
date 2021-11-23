import cv2
import math
import numpy as np
from strokes_funct import strokes_funct
from matplotlib import pyplot as plt
from skimage import filters
from skimage import feature


##input
image = cv2.imread('inputs/7--88.jpg')
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
edges=strokes_funct(gray)

#Line segments

plt.imshow(edges, cmap='gray')
plt.show()

cv2.waitKey(0)



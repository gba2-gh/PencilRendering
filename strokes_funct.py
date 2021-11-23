import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
from skimage import feature

def strokes_funct(gray, method=0):
    height = gray.shape[0]
    width = gray.shape[1]
    ##GRADIENT
    G= np.zeros([height, width], dtype='float')
    gx= np.zeros((height,width))
    gy= np.zeros((height,width))
    #for i in range(height):
    #    for j in range(width):
    #        G[i][j]=gray[i][j]

    dx = np.absolute(gray[: , 0:width - 1] - gray[: , 1:width])  ##Toda la columna, de 0 a width -1 
    dy=  np.absolute(gray[1:height-1,:] - gray[2:height,:])

    gx[:,0:width-1] = dx
    gy[1:height-1,:] =dy

    G= gx+gy


    #CANNY
    cannyE = feature.canny(gray, sigma=1,low_threshold=0, high_threshold=100)

    #sobel
    sobelE= filters.sobel(gray)


    #Display
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

    ax[0].imshow(G, cmap='gray')
    ax[0].set_title('Gradiente', fontsize=20)
    ax[1].imshow(cannyE, cmap='gray')
    ax[1].set_title('Canny', fontsize=20)
    ax[2].imshow(sobelE, cmap='gray')
    ax[2].set_title('Sobel', fontsize=20)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()

    if method==0:
        out=G
    elif method==1:
        out=cannyE
    else:
        out=sobel

    return(out)
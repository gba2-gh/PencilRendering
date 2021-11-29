import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters, feature, transform
from scipy import signal


def edge_funct(gray, method=0, display=False):
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
    dy=  np.absolute(gray[0:height-1,:] - gray[1:height,:])

    gx[:,0:width-1] = dx
    gy[0:height-1,:] =dy

    G= gx+gy


    #CANNY
    cannyE = feature.canny(gray, sigma=1,low_threshold=0, high_threshold=100)

    #sobel
    sobelE= filters.sobel(gray)

    
    #Display
    if display:
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
        out=sobelE

    return(out)

def strokes_funct(edges, lineSize=7, display=False):
    height = edges.shape[0]
    width = edges.shape[1]
    #Line segments
    kernel_size= lineSize#21#int(height*(1/30))
    kernel=np.zeros((kernel_size,kernel_size))

    ####Primera direccion
    kernel[math.floor(kernel_size/2),:] = 1

    ## Convolucion en 8 direcciones
    G=np.zeros((height, width, 8))
    L=np.zeros((kernel_size,kernel_size,8))
    for i in range(8):
        ker=transform.rotate(kernel,i*22.5)
        L[:,:,i]=ker
        aux=signal.convolve2d(edges, ker, mode='same')
        G[:,:,i] = aux
    print(G[:,:,1])
    # Indices de elementos maximos
    max_idx = np.argmax(G, axis=2)
    ##Formar clasificacion (C) con los elementos maximos de G
    C=np.zeros((height, width, 8))
    for i in range(8):
        #idx = max_idx==i #and
        aux = edges*(max_idx==i)
        C[:,:,i] = aux

    #Display c maps
    if display:
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

    out=1-out/255

    basic_ker = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1,:] = 1 # ------- (horizontal line)

    stroke_width=1
    for w in range(1, stroke_width):
        if (math.floor(kernel_size/2) - w) >= 0:
            basic_ker[math.floor(kernel_size/2) - w, :] = 1
        if (math.floor(kernel_size/2) + w) < (kernel_size):
            basic_ker[kernel_size + 1 + w, :] = 1
    # It's time to compute S':
    S_tag_sep = np.zeros_like(C)
    for d in range(8):
        ker = transform.rotate(basic_ker, (d * 180) / 8)
        S_tag_sep[:,:,d] = signal.convolve2d(C[:,:,d], ker, mode='same')
    S_tag = np.sum(S_tag_sep, axis=2)
    # Remember that S shpuld be an image, thus we need to make sure the values are in [0,1]
    S_tag_normalized = (S_tag - np.min(S_tag.ravel())) / (np.max(S_tag.ravel()) - np.min(S_tag.ravel()))
    # The last step is to invert it (b->w, w->b)
    S = 1 - S_tag_normalized





    return(S)
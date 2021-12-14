import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters, feature, transform
from scipy import signal


def gen_edge(gray, method=0, display=False):
    height = gray.shape[0]
    width = gray.shape[1]
    ##GRADIENTE
    G= np.zeros([height, width], dtype='float')
    gx= np.zeros((height,width))
    gy= np.zeros((height,width))

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


def gen_strokes(edges, lineSize=7, numDir=8, lineWidth=1, display=False):
    height = edges.shape[0]
    width = edges.shape[1]

    kernelSize= lineSize
    #kernel impar
    if kernelSize %2==0:
        kernelSize+=1
    kernel=np.zeros((kernelSize,kernelSize))

    ####Primera direccion horizontal
    kernel[math.floor(kernelSize/2),:] = 1

    ## Convolucion en n direcciones
    G=np.zeros((height, width, numDir))
    L=np.zeros((kernelSize,kernelSize,numDir))
    theta=180/numDir
    for i in range(numDir):
        ker=transform.rotate(kernel,i*theta) ##Rotar kernel \theta grados
        L[:,:,i]=ker
        aux=signal.convolve2d(edges, ker, mode='same')
        G[:,:,i] = aux

    # Indices de elementos maximos
    max_idx = np.argmax(G, axis=2)

    ##Formar clasificacion (C) con los elementos maximos de G
    C=np.zeros((height, width, 8))
    for i in range(numDir):
        aux = edges*(max_idx==i)
        C[:,:,i] = aux

    #Display c maps
    if display:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))

        ax[0].imshow(C[:,:,0], cmap='gray')
        ax[0].set_title('0', fontsize=20)
        ax[0].axis('off')
        ax[1].imshow(C[:,:,1], cmap='gray')
        ax[1].set_title('1', fontsize=20)
        ax[1].axis('off')
        ax[2].imshow(C[:,:,2], cmap='gray')
        ax[2].set_title('3', fontsize=20)
        ax[2].axis('off')
        ax[3].imshow(C[:,:,3], cmap='gray')
        ax[3].set_title('3', fontsize=20)
        ax[3].axis('off')
        plt.show()


    ##Line shaping
    #out= np.zeros((height, width))
    #for i in range(8):
    #    out =out + signal.convolve2d(C[:,:,i], L[:,:,i], mode='same')

    #out=1-out/255

    ker_ref = np.zeros((kernelSize , kernelSize ))
    ker_ref[math.floor(kernelSize/2),:] = 1 # ------- (horizontal line)


    ##Tamaño de la linea del trazo
    for w in range(0, lineWidth):
        if (math.floor(kernelSize/2)-1 - w) >= 0:
            ker_ref[math.floor(kernelSize/2) -1 - w, :] = 1
        if (math.floor(kernelSize/2) + w) < (kernelSize):
            ker_ref[math.floor(kernelSize/2) + 1 + w, :] = 1


    Spn = np.zeros_like(C)
    for d in range(numDir):
        ker = transform.rotate(ker_ref, d * theta)
        Spn[:,:,d] = signal.convolve2d(C[:,:,d], ker, mode='same')
    Sp = np.sum(Spn, axis=2)
    # normalizar en [0,1]
    S = (Sp - np.min(Sp.ravel())) / (np.max(Sp.ravel()) - np.min(Sp.ravel()))
    # invertir
    #S = 1 - S

    return(S)
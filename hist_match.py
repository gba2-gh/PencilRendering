import numpy as np
from skimage import filters, exposure

def hist_match(gray, p):
    h = exposure.histogram(gray, nbins=256)

    #Equalizar histogramas
    P = np.cumsum(p)
    H = np.cumsum(h / np.sum(h))

    # Histogram matching:
    #Group mapping Law
    matched = np.zeros_like(p)
    for i in range(256):
        # Valor igual o mas cercano:
        dist = np.abs(P - H[i])
        argmin_dist = np.argmin(dist)##indice del  minimo
        matched[i] = argmin_dist
    matched= matched / 256

    J = matched[(255 * gray).astype(int)]
    
    return J

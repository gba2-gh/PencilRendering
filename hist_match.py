import numpy as np
from skimage import filters, exposure

def hist_match(gray, p):
    P = np.cumsum(p)
    #Equalize
    # histograma original:
    h = exposure.histogram(gray, nbins=256)
    # CDF :
    H = np.cumsum(h / np.sum(h))

    # Histogram matching:
    matched = np.zeros_like(p)
    for v in range(256):
        # find the closest value:
        dist = np.abs(P - H[v])
        argmin_dist = np.argmin(dist)##indice del  minimo
        matched[v] = argmin_dist
    matched= matched / 256

    J = matched[(255 * gray).astype(int)]
    # smooth:
    J_b = filters.gaussian(J, sigma=np.sqrt(2))


    return J_b

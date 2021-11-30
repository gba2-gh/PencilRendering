import numpy as np
from matplotlib import pyplot as plt

def disp_hist(hist, title):
    histogram, bin_edges = np.histogram(hist, bins=256, range=(0, 1))
    plt.figure()
    plt.title(title)
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0.0, 1.0])  
    plt.plot(bin_edges[0:-1], histogram)  
    plt.show()
    return

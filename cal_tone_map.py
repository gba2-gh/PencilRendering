import numpy as np
import math
def cal_tone_map(w1=11,w2=37,w3=52):
        ##TONE MAP
    #####Parametros aprendidos paper
    desv_b=9
    u_a=105
    u_b=225
    media_d=90
    desv_d=11
    lap_peak=255
    p = np.zeros(256)
    Z=0
    for i in range(256):
        if i <= lap_peak:
            p1 = w1 * (1 / desv_b) * np.exp(-(255 - i) / desv_b)
        else:
            p1=0

        if (u_a <= i <= u_b):
            p2 = w2* 1 / (u_b - u_a)
        else:
            p2 = 0

        p3 = w3* (1/np.sqrt(2*math.pi*desv_d))*(np.exp(-(i-media_d)**2/(2*desv_d**2)))*0.01
        p[i] = p1 + p2 + p3
        Z=Z+p[i]

    p=p/Z


    return p
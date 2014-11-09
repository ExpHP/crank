
from colorsys import hls_to_rgb
import numpy as np
import math

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + math.pi)  / (2 * math.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    l = l**2.
    l /= l.max()
    l *= 0.60

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1,2,0)
    return c

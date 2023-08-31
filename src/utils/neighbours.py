import numpy as np
from utils.params import *

def get_neighbours(x):
    l = []
    # up
    if x - M >= 0:
        l.append(x - M)
    # down
    if x + M <= 63:
        l.append(x + M)
    # right
    if (x - 1) % M != M-1:
        l.append(x - 1)
    # left
    if (x + 1) % M != 0:
        l.append(x + 1)

    l.append(x)

    return np.asarray(l)

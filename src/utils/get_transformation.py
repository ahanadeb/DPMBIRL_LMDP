import numpy as np


def get_f(F, X):
    # get matrix which needs to be multiplied to r in order to get 64X1 reward vector
    f = np.zeros((X, F))
    n = np.ones((2,))
    h = np.zeros((int(F / 2), int(F / 4)))
    i = 0
    j = 0
    while i < int(F / 2):
        h[i:i + 2, j] = n
        j = j + 1
        i = i + 2
    i = 0
    j = 0
    while i < X:
        f[i:i + 8, j:j + 4] = h
        f[i + 8:i + 16, j:j + 4] = h
        i = i + 16
        j = j + 4

    return f  # np.matmul(f,r).reshape((X,1))

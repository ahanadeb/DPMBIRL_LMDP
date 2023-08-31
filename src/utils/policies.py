import numpy as np


def pi_lazy(X, A):
    p = np.zeros((X, A))
    for i in range(0, X):
        p[i, 0] = 1

    return p


def pi_aggressive(X, A):
    p = np.zeros((X, A))
    for i in range(0, X):
        if i <= 50:
            p[i, 0] = 1
        else:
            p[i, 1] = 1
    return p

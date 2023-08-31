import numpy as np
from utils.params import *
from utils.util_functions import *
from utils.neighbours import *


def grad(r, P, z):
    # I am getting dz from here
    dz = np.zeros((F, X))
    #  print("z",z)
    for n in range(0, 30):
        for i in range(0, F):
            for j in range(0, X):
                ll = np.multiply(P[j, :].reshape((X,)), np.power(z, gamma - 1).reshape((X,)))
                #dz[i, j] = ((z[j] / gamma) + np.exp(r[i] / gamma) * gamma * np.dot(ll, dz[i, :]))
                dz[i, j] = ((z[j] / gamma) + np.exp(r[i] / gamma) * gamma * np.dot(ll, dz[i, :]))

    # for i in range(0, F):
    #     for j in range(0, X):
    #         a = (dz[i, j] / z[j]) * gamma
    #         dz[i, j] = a
    #

    return dz


def gradLLH(s1, s2, z, dz):
    llhgrad = np.zeros((F, 1))

    z_ex = np.exp(eta_new * z)
    for i in range(0, F):
        l = get_neighbours(s1)
        s = 0
        p = 0
        for k in l:
            s = s + eta_new * dz[i, int(k)] * z_ex[int(k)] * z[int(k)] / gamma
            p = p + z_ex[int(k)]
        dzdr = dz[i, int(s2)] *z[int(s2)] / gamma
        llhgrad[i, 0] = eta_new * dzdr - s / p

    return llhgrad

import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.lmdp import *
from utils.neighbours import *


def acc_ratio(traj, r1, z1, r2, z2, P_un):
    a = 0
    b = 0
    for i in range(0, len(traj)):
        tr = traj[i]
        states = tr[:, 0]
        next_s = tr[:, 1]
        for j in range(0, tr.shape[0]):
            l = get_neighbours(states[j])
            p1 = np.zeros((X, 1))
            p2 = np.zeros((X, 1))
            assert next_s[j] in l, "state not in possible states"
            for k in l:
                p1[int(k)] = np.exp(eta_new * np.log(z1[int(k)]))
                p2[int(k)] = np.exp(eta_new * np.log(z2[int(k)]))
            p1 = p1 / np.sum(p1)
            p2 = p2 / np.sum(p2)
            a = a + np.log(p1[int(next_s[j])])
            b = b + np.log(p2[int(next_s[j])])

    if a-b >0:
        ratio =1
    else:
        ratio = np.exp(a-b)

    return ratio

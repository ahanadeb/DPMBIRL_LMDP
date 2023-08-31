import numpy as np
from utils.util_functions import *
from utils.acc_ratio import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.cluster_assignment import *
from utils.lmdp import *
from utils.neighbours import *
import torch


def calLogLLH_torch(r, traj, P_un):
    torch.autograd.set_detect_anomaly(True)
    # converting r tensor from (F,1) to (X,1) matrix
    r = torch.tensor(r, dtype=torch.float64, requires_grad=True)
    F_new = torch.tensor(F_matrix, dtype=torch.float64)
    new_r = torch.reshape(torch.matmul(F_new, r), (X,))

    # calculating Z
    k= torch.exp(new_r / gamma)
    #get diagonal matrix
    m=torch.eye(X,dtype=torch.float64 ) * torch.outer(torch.ones(X,dtype=torch.float64), k )

    #G = torch.diagflat(torch.exp(new_r / gamma))
    #G = torch.tensor(G, dtype=torch.float64)

    z = torch.ones((X,), dtype=torch.float64 )
    P_un = torch.tensor(P_un,dtype=torch.float64)
    # c = torch.tensor(c, dtype=torch.float64)
   # z = torch.matmul(torch.matmul(m, P_un), c)
    for i in range(0, 70):
        c = torch.pow(z, gamma)
        c = torch.tensor(c, dtype=torch.float64)
        z = torch.matmul(torch.matmul(m, P_un), c)




    z = z.reshape((X,))


    dot_val = np.zeros((X, 1))
    dot_mat2 = np.zeros((X, X))
    llh2 = 0
    for i in range(0, len(traj)):
        tr = traj[i]
        states = tr[:, 0]
        next_s = tr[:, 1]
        for j in range(0, tr.shape[0]):
            # create a vector for dot product with val
            dot_val[int(next_s[j])] = dot_val[int(next_s[j])] + 1
            l = get_neighbours(states[j])
            for k in l:
                dot_mat2[j, int(k)] = 1
        dot_val = dot_val * eta_new
        dot_val = torch.reshape(torch.tensor(dot_val), (X,))

        f = torch.dot(dot_val, z)

        dot_mat2 = dot_mat2 * eta_new
        dot_mat = torch.tensor(dot_mat2)

        x = torch.matmul(dot_mat, z)

        llh2 = llh2 + (f - x).sum()
    #print("llh2", llh2)
    llh2.backward()
    #print("r.grad", r.grad)
    return r.grad

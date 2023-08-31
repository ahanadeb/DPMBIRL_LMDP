import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.acc_ratio import *
from utils.lmdp import *
from utils.cluster_assignment import sample_reward
from utils.log_post import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils.evd import *


def emirl():
    P_un = uncontrolled_lmdp()
    r = get_reward_lmdp(F, RF)
    tn=9

    ch, traj_set_prev, rewards_prev, y_prev= 0,0,0,0
    traj_data, rewards_gt, y = lmdp_trajectories2(F, P_un, r, ch, traj_set_prev, rewards_prev, y_prev, RF, tn, tl)

    np.set_printoptions(threshold=np.inf)

    EVD = []

    data = Data()
    data.L = -np.inf
    data.rho=np.ones((cl_no,1))/cl_no
    data.weight = np.zeros((F,cl_no))
    for k in range(cl_no):
        data.weight[:,k] = sample_reward(F, mu, sigma, lb, ub)
    z, L = estep(data,traj_data,P_un)
    data.z = z
    data.L = L
    print("L", L)

    for iter in range(50):
        rho, weight= mstep(data, traj_data,P_un)
        assignment = np.amax(data.z, axis=1)

        evd=evd_emirl(rewards_gt,assignment, data, P_un,tn)
        EVD.append(evd)
        print("EVD: " ,evd)
        print("assignment: ", assignment )

        newdata =Data()
        newdata.rho = rho
        newdata.weight = weight
        z, L = estep(newdata, traj_data,P_un)
        newdata.z = z
        newdata.L=L
        print("newdata L", newdata.L)
        delta = newdata.L-data.L
        if delta > 0:
            data=newdata

    print("cluster assignment: ", np.amax(data.z, axis=1))
    plt.plot(np.asarray(EVD))
    plt.xlabel("iterations")
    plt.ylabel("average EVD")
    plt.show()






class Data:
    L = []
    rho= []
    weight = []
    z= []

def estep(data, traj_set,P_un):
    ntraj = traj_set.shape[2]
    logllh = np.zeros((ntraj, cl_no))
    logllh2 = np.zeros((ntraj, cl_no))
    traj=[]
    for i in range(traj_set.shape[2]):
        traj.append(traj_set[:,:,i])

    for k in range(cl_no):
        w = data.weight[:,k]
        for m in range(ntraj):
            #llh, gradL = calLogLLH(w, traj, P_un)
            llh2, gradL2=calLogLLH_sing(w, traj_set[:,:,m], P_un)
            logllh[m, k] = llh2

    z = np.zeros((ntraj, cl_no))
    for m in range(ntraj):
        for k in range(cl_no):

            z[m,k]= data.rho[k] *np.exp(logllh[m,k])

        if np.sum(z[m,:])>0:
            z[m,:] = z[m,:]/np.sum(z[m,:])

    L=0

    for m in range(ntraj):
        for k in range(cl_no):
            if z[m,k]>0:
                L = L + (np.log(data.rho[k]) + logllh[m, k]) * z[m, k]

    return z,L


def mstep(data, traj_set,P_un):
    ntraj = traj_set.shape[2]
    traj = []
    for i in range(traj_set.shape[2]):
        traj.append(traj_set[:, :, i])
    bn = (lb, ub)
    bnds = ((bn,) * F)
    rho = np.zeros((cl_no,1))
    for k in range(cl_no):
        rho[k] = np.sum(data.z[:,k])
    rho = rho/ntraj

    weight = np.zeros((F, cl_no))
    for k in range(cl_no): #cl_no
        if rho[k]>0:
            print("started 1")
            func = lambda x: calEMLogLLH(data.z[:, k], traj,P_un,x)
            print("ended 1")
            w_test = []
            v_test=-np.inf

            for i in range(1):
                w0 = data.weight[:, k]
                print("started")
               # w,val = minimize(func, w0, bounds=bnds)

                res = minimize(func, w0,
                               method='SLSQP',
                               bounds=bnds)
                w = res.x
                val = res.fun
                print(w , val, res.success, res.status)
                if v_test <-val:
                    v_test=-val
                    w_test = w
            print("ended")
            weight[:,k]=w_test
        else:
            weight[:, k] = data.weight[:,k]

    return rho, weight
#
# def savehist(data,hst,i):
#    b =  np.amax(data.z, axis=1)
#


import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.log_post import *
from utils.acc_ratio import *
from utils.torch_grad import *
from utils.evd import *


def update_weight(k, traj_set, C, P_un):
    accepted = 0
    for i in range(0, 10):
        t = []
        traj = []
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                t = np.append(t, l)
        for y in t:
            traj.append(traj_set[:, :, int(y)])

        r1 = C.reward[:, k]
        p1 = C.policy[:, :, k]
        z1 = C.value[:, k]

        # new added
        if np.all(C.gradL[:, k] == 0):  # figure out shape of llh
            llh, gradL = calLogLLH(r1, traj, P_un)
            prior, gradP = calLogPrior(r1)
            C.llh[k, 0] = llh
            C.prior[k, 0] = prior
            C.gradL[:, k] = np.squeeze(gradL)
            C.gradP[:, k] = np.squeeze(gradP)

        logP = C.llh[k]  # + C.prior[k]
        grad = C.gradL[:, k]  # + C.gradP[:, k]

        eps=np.random.randn(F)
       # print(grad)
     #   print("reward before", r1.reshape((1,F)))
        r2 = r1.reshape((r1.shape[0], 1)) + np.power(.1,2) * grad.reshape((r1.shape[0], 1))  # + (
        #   sigma * eps)

        # r2 = r1.reshape((r1.shape[0], 1)) + (np.power(1, 2) * grad / 2).reshape((r1.shape[0], 1)) #+ (sigma * eps)
        r2 = np.maximum(lb, np.minimum(ub, r2))
     #   print("reward after after", r2.reshape((1,F)))
        #     print("after", r2)
        #      print("After, ", r2)
        #  print("r2 after", r2)
    llh2, gradL2 = calLogLLH(r2, traj, P_un)
    prior2, gradP2 = calLogPrior(r2)
    logP2 = llh2  # + prior2
    grad2 = gradL2  # + gradP2
    eps=eps*1e-4
    a = eps + (sigma / 2) * (grad + grad2)

   # print(grad+grad2)
   # print(logP2)
    a = np.exp(-.5 * np.sum(np.power(a,2))) * np.exp(logP2)
    b = np.exp(-.5 * np.sum(np.power(eps,2))) * np.exp(logP)
    ratio = a / b
  #  ratio = np.exp(logP2) / np.exp(logP)
    rand_n = random.uniform(0, 1)
    #print(a, b)
    #print(np.exp(logP2), np.exp(logP))
    llhp=logP
    if rand_n < ratio:
    #if logP2> logP:
        print("Accepted")
        accepted = 1
        llhp=logP2
        #print("evd: ",evd_single(r1, r2, P_un))
        C.reward[:, k] = np.squeeze(r2)
        # C.policy[:, :, k] = p2
        # C.value[:, k] = v2
        C.llh[k] = llh2
        C.prior[k] = prior2
        C.gradL[:, k] = np.squeeze(gradL2)
        C.gradP[:, k] = np.squeeze(gradP2)

        # old stuff (randomly sampling r)
        # r2=sample_reward(F, mu, sigma, lb, ub)
        # rx = reward_feature(M, N,r2).reshape(X, 1)
        # z2, r3 = get_z(rx, P_un)
        # #p2 = optimal_policy(P_un, z2)
        #
        #
        # ratio = acc_ratio(traj, r2, z2, r1, z1,P_un)
        # rand_n = random.uniform(0, 1)
        # if rand_n < ratio:
        #     C.reward[:, k] = np.squeeze(r2)
        #    # C.policy[:,:, k] = p2
        #     C.value[:, k] = np.squeeze(z2)

    return C, accepted,llhp




def update_weight2(k, traj_set, C, P_un):
    accepted = 0
    for i in range(0, 10):
        t = []
        traj = []
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                t = np.append(t, l)
        for y in t:
            traj.append(traj_set[:, :, int(y)])

        r1 = C.reward[:, k]
        p1 = C.policy[:, :, k]
        z1 = C.value[:, k]

        # new added



        # old stuff (randomly sampling r)
        r2=sample_reward(F, mu, sigma, lb, ub)
        rx = reward_feature(M, N,r2).reshape(X, 1)
        z2, r3 = get_z(rx, P_un)
        #p2 = optimal_policy(P_un, z2)


        ratio = acc_ratio(traj, r2, z2, r1, z1,P_un)
        rand_n = random.uniform(0, 1)
        if rand_n < ratio:
            print("Accepted")
            C.reward[:, k] = np.squeeze(r2)
           # C.policy[:,:, k] = p2
            C.value[:, k] = np.squeeze(z2)

    return C, accepted
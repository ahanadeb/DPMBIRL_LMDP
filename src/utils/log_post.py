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
from utils.grad import grad
from utils.grad import gradLLH

np.set_printoptions(threshold=np.inf)


def calLogLLH(r, traj, P_un):
    r3 = reward_feature(M, N, r).reshape(X, 1)

    z, r2 = get_z(r3, P_un)
    # P = optimal_policy(P_un, z)

    llh = 0

    dz = grad(r, P_un, z)
    f_grad = np.zeros((F, 1))

    for i in range(0, len(traj)):
        tr = traj[i]
        states = tr[:, 0]
        next_s = tr[:, 1]

        for j in range(0, tr.shape[0]):
            l = get_neighbours(states[j])
            p = np.zeros((X, 1))

            assert next_s[j] in l, "state not in possible states"
            for k in l:
                p[int(k)] = np.exp(eta_new * np.log(z[int(k)]))
            p = p / np.sum(p)
            llh = llh + np.log(p[int(next_s[j])])

        for q in range(0, tr.shape[0]):
            k = gradLLH(states[q], next_s[q], z, dz)
            # k=(d_logpi[:, int(states[q])]).reshape((F,1))
            f_grad = f_grad + k
    #print(f_grad/llh)
    return llh, f_grad


def calLogPrior(r):
    x = r - mu
    prior = np.sum(-1 * (x * np.transpose(x)) / (2 * np.power(sigma, 2)))

    grad = -x / np.power(sigma, 2)
    return prior, grad


def stateFeature(X, F, M, N):
    F = np.zeros((X, F))
    for x in range(1, M + 1):
        for y in range(1, N + 1):
            s = loc2s(x, y, M)
            i = np.ceil(x / B)
            j = np.ceil(y / B)
            f = loc2s(i, j, M / B)
            F[int(s) - 1, int(f) - 1] = 1
    return F


def loc2s(x, y, M):
    x = max(1, min(M, x))
    y = max(1, min(M, y))
    s = (y - 1) * M + x

    return s


def calDPMLogPost(traj_set, C, P_un):
    # print(C.assignment)
    prob = assignment_prob(C.assignment, alpha)
    logDPprior = np.log(prob)
    logLLH = 0
    logPrior = 0
    NC = int(np.max(C.assignment))
    for k in range(0, NC + 1):
        r1 = C.reward[:, k]
        if not C.policy_empty:
            # from reward get best policy and optimal value
            r = reward_feature(M, N, r1).reshape(X, 1)
            z, r2 = get_z(r, P_un)
            # P = get_transitions(M, N, A, p, q, obstacles)
            # V, V_hist, policy, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
        #  policy = optimal_policy(P_un, z)
        else:
            policy = C.policy[:, :, k]

        t = []
        traj = []
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                t = np.append(t, l)
        for y in t:
            traj.append(traj_set[:, :, int(y)])

        llh, gradL = calLogLLH(r1, traj, P_un)

        prior, gradP = calLogPrior(r1)
        logLLH = logLLH + llh
        logPrior = logPrior + prior
    print("logpost ", logDPprior, " ", logLLH, " ", logPrior)
    logPost = logDPprior + logLLH + logPrior

    return logPost


def state_count(states):
    states = states.T
    # get count of how many times each state was reached
    count = np.zeros((X, 1))
    for i in range(0, states.shape[0]):
        count[int(states[i])] = count[int(states[i])] + 1
    return count


def calEMLogLLH(z, traj, P_un, w):
    llh = 0
    # grad=0
    for m in range(len(traj)):
        if z[m] > 0:
            x = calLogLLH2(w, traj, P_un)
            llh = llh + z[m] * x
            # grad = grad + z[m] * y

    llh = -llh
    # grad=-grad
    return llh


def calLogLLH2(r, traj, P_un):
    r3 = reward_feature(M, N, r).reshape(X, 1)

    z, r2 = get_z(r3, P_un)
    # P = optimal_policy(P_un, z)

    llh = 0

    for i in range(0, len(traj)):
        tr = traj[i]
        states = tr[:, 0]
        next_s = tr[:, 1]

        for j in range(0, tr.shape[0]):
            l = get_neighbours(states[j])
            p = np.zeros((X, 1))

            assert next_s[j] in l, "state not in possible states"
            for k in l:
                p[int(k)] = np.exp(eta_new * np.log(z[int(k)]))
            p = p / np.sum(p)
            llh = llh + np.log(p[int(next_s[j])])

    # print(f_grad)
    return llh


def calLogLLH_sing(r, traj, P_un):
    r3 = reward_feature(M, N, r).reshape(X, 1)

    z, r2 = get_z(r3, P_un)
    # P = optimal_policy(P_un, z)

    llh = 0

    dz = grad(r, P_un, z)
    f_grad = np.zeros((F, 1))

    states = traj[:, 0]
    next_s = traj[:, 1]

    for j in range(0, traj.shape[0]):
        l = get_neighbours(states[j])
        p = np.zeros((X, 1))

        assert next_s[j] in l, "state not in possible states"
        for k in l:
            p[int(k)] = np.exp(eta_new * np.log(z[int(k)]))
        p = p / np.sum(p)
        llh = llh + np.log(p[int(next_s[j])])

    for q in range(0, traj.shape[0]):
        k = gradLLH(states[q], next_s[q], z, dz)
        # k=(d_logpi[:, int(states[q])]).reshape((F,1))
        f_grad = f_grad + k
    # print(f_grad)
    return llh, f_grad

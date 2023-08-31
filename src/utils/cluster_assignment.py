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


def initialise_c(tl):
    c = np.zeros((tl, 1))
    for i in range(0, tl):
        c[i] = int(random.randint(0, tl - 1))

    return c


def sample_reward(F, mu, sigma, lb, ub):
    r = np.zeros((F, 1))
    r = mu + np.random.randn(F) * sigma
    r = np.maximum(lb, np.minimum(ub, r));
    r = r.reshape((1, F))
    # print("sample reward", r)
    return r


def assignment_prob(c, alpha):
    z = count(c).astype(int)
    ix = np.arange(1, len(z) + 1).astype(int)
    Z = np.sum(z*ix)
    k = np.power(ix, z) * factorial(z)


    #alpha is one so dont need the first part as of now
    #pr = np.math.factorial(Z) / np.prod(np.arange(alpha, alpha + Z)) * np.power(alpha, sum(z)) / np.prod(k)
    pr = np.power(alpha, sum(z)) / np.prod(k)

    assert pr >0, "PR IS NEGATIVE"


    return np.float128(pr)


def count(c):
    N = len(c)
    nrCl = int(np.max(c))
    szCl = np.zeros((1, nrCl))
    z = np.zeros((1, N))
    for k in range(0, nrCl):
        for j in range(0, len(c)):
            if c[j] == k + 1:
                szCl[0, k] = szCl[0, k] + 1
    for i in range(0, nrCl):
        if szCl[0, i] > 0:
            z[0, int(szCl[0, i]) - 1] = z[0, int(szCl[0, i]) - 1] + 1
    return z[0]


def log_post(c):
    log_cluster_ass = np.log(assignment_prob(c, alpha))

    return log_post


class Cluster:
    assignment = []
    reward = []
    policy = []
    values = []
    llh = []
    prior = []
    gradL = []
    gradP = []


def init_cluster(C, tn, F, X, P_un):
    C.assignment = np.random.randint(0, tn, size=(1, tn))[0]
    C.reward = np.zeros((tn, F))
    C.policy = np.zeros((X, X, tn))
    C.value = np.zeros((X, tn))
    NC = np.max(C.assignment)
    for i in range(0, NC + 1):
        C.reward[i, :] = sample_reward(F, mu, sigma, lb, ub)
        r = reward_feature(M, N, C.reward[i, :]).reshape(X, 1)
        z, r2 = get_z(r, P_un)
        C.value[:, i] = np.squeeze(z)
    C.reward = np.transpose(C.reward)
    C.llh = np.zeros((NC, 1))
    C.prior = np.zeros((NC, 1))
    C.gradL = np.zeros((C.reward.shape[0], C.reward.shape[1]))
    C.gradP = np.zeros((C.reward.shape[0], C.reward.shape[1]))
    C.policy_empty = True
    return C


def newZ(P_un):
    r = sample_reward(F, mu, sigma, lb, ub)
    r = r.reshape((F, 1))
    r2 = reward_feature(M, N, r).reshape(X, 1)
    z, r = get_z(r2, P_un)
    return r, P, z


def update_cluster(C, m, traj_set, P_un, iter=2):
    for it in range(0, iter):
        c = int(C.assignment[m])
        r1 = C.reward[:, c]
        p1 = C.policy[:, :, c]
        z1 = C.value[:, c]
        NC = int(np.max(C.assignment))
        prior = np.zeros((NC + 1, 1))
        for k in range(0, NC + 1):
            for l in range(0, len(C.assignment)):
                if C.assignment[l] == k:
                    prior[k] = prior[k] + 1

        prior[c] = prior[c] - 1
        prior[NC] = alpha
        c2 = int(sampleMult(prior))
        if c2 > NC:  # if its a new cluster, accept it with prob
            r2, p2, z2 = newZ(P_un)
        else:
            r2 = C.reward[:, c2]
            p2 = C.policy[:, :, c2]
            z2 = C.value[:, c2]
        # if its an existing cluster, acceptance ratio
        traj = []
        traj.append(traj_set[:, :, m])
        ratio = acc_ratio(traj, r2, z2, r1, z1, P_un)
        # print("ratio",ratio)
        rand_n = random.uniform(0, 1)
        if rand_n < ratio:
            C.assignment[m] = c2
            if c2 > NC:
                # print("reward shape",C.reward.shape)
                C.reward[:, c2] = r2
                C.policy[:, :, c2] = p2
                C.value[:, c2] = v2
        C.policy_empty = False
    return C


def sampleMult(p):
    s = sum(p)
    if s != 1:
        p = p / s
    q = np.cumsum(p)
    c2 = 0
    r = random.uniform(0, 1)
    for i in range(0, len(q)):
        if q[i] > r:
            c2 = i
            break
    return c2


def relabel_cluster(C, tn):
    R = np.zeros((1, F))
    P = np.zeros((X, X))
    V = np.zeros((X, 1))
    tmpId = np.zeros((1, 2))
    relabel = 0
    for k in range(0, int(np.max(C.assignment)) + 1):
        sum = 0
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                sum = sum + 1
        if sum > 0:
            R = np.append(R, C.reward[:, k].reshape((1, C.reward[:, k].shape[0])), axis=0)
            P_temp = C.policy[:, :, k].reshape((C.policy[:, :, k].shape[0], C.policy[:, :, k].shape[1]))
            P = np.dstack((P, P_temp))
            V = np.append(V, C.value[:, k].reshape((C.value[:, k].shape[0], 1)), axis=1)
            tmpId = np.append(tmpId, np.asarray([k, R.shape[0] - 2]).reshape((1, 2)), axis=0)
            relabel = 1
    R = R[1:, :]
    V = V[:, 1:]
    P = P[:, :, 1:]
    tmpId = tmpId[1:, :]
    if relabel == 1:
        B = np.zeros((len(C.assignment), 1))
        # print(len(tmpId))
        for i in range(0, len(tmpId)):
            for l in range(0, len(C.assignment)):
                if C.assignment[l] == tmpId[i, 0]:
                    B[l] = int(tmpId[i, 1])
        C.assignment = B
        C.reward = np.transpose(R)
        C.value = V
        C.policy = P

    return C

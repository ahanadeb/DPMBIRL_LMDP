import numpy as np
from utils.reward import *
from utils.params import *
import scipy
import scipy.sparse
from utils.transition import *
from utils.neighbours import *
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
# import numpy.linalg as linalg


# P_un = uncontrolled transition dynamics
def get_z(r, P_un):
    np.set_printoptions(threshold=np.inf)
    r = matrix_to_vector(r)
    z = np.ones((X, 1))
    G = np.diagflat(np.exp(r / gamma), )

    for i in range(0,2500):
        z = np.matmul(np.matmul(G, P_un), np.power(z, gamma))
    return z, r


def optimal_policy(P_un, z):
    a = np.zeros((X, X))
    for i in range(0, X):
        for j in range(0, X):
            a[i, j] = P_un[i, j] * z[j]
        a[i, :] = a[i, :] / np.sum(a[i, :])
    return a



def lmdp_gen_traj(X, P, z, tl):
    traj = np.zeros((tl, 2))
    #s0=(random.choice([7,15, 23,31,39,47,55,63]))
    s0 = random.randint(0, X - 1)  # initial state
    print("Initial states", s0)
    for i in range(0, tl):
        states = (np.arange(X)).reshape((X,))
        traj[i, 0] = s0
        # get all neighbours of the state

        l = get_neighbours(s0)
        l2=np.zeros((len(l),1))
        neighbours_prob = np.zeros((X, 1))
        for j in range(0, len(l)):
            l2[j,0]=z[l[j]]
            q = np.log(z[l[j]])
            q = np.exp(eta_new * q)
            neighbours_prob[l[j]] = q
        neighbours_prob = neighbours_prob / np.sum(neighbours_prob)

       # next_s = random.choices(states, weights=neighbours_prob.reshape((X,)), k=1)[0]
        #added

        next_s = l[np.argmax(l2)]


        traj[i, 1] = next_s#[0]
        s0 = next_s#[0]

    # print("traj", traj)
    return traj


def lmdp_trajectories(F, P_un, r, RF, tn, tl):
    # shape RF*F 3*16
    traj_data = np.zeros((tl, 2, tn))  # 6 trajectories
    seq = []
    traj_per_agent = int(tn / RF)
    y = np.arange(RF)
    y = np.repeat(y, traj_per_agent)
    np.random.shuffle(y)
    # y[0]=0

    for i in range(0, tn):
        # if i ==0:
        # traj_data[:, :, i] = traj_test
        # else:
        j = int(y[i])
        seq.append(j)
        reward = r[j, :]
        reward = reward_feature(M, N, reward).reshape(X, 1)
        z, r2 = get_z(reward, P_un)
        P = optimal_policy(P_un, z)
        traj = lmdp_gen_traj(X, P, z, tl)
        traj_data[:, :, i] = traj
    rewards_gt = np.zeros((F, 1))
    for k in seq:
        rewards_gt = np.append(rewards_gt, np.transpose(r[k, :]).reshape((F, 1)), axis=1)
    rewards_gt = rewards_gt[:, 1:]
    # print("now", traj_data.shape)
    print("original assignment", y)
    return traj_data, rewards_gt, y


def lmdp_trajectories2(F, P_un, r, ch, traj_set_prev, rewards_prev, y_prev, RF, tn, tl):
    # shape RF*F 3*16
    traj_data = np.zeros((tl, 2, tn))  # 6 trajectories
    seq = []
    traj_per_agent = int(tn / RF)
    y = np.arange(RF)
    y = np.repeat(y, traj_per_agent)
    np.random.shuffle(y)
    # y[0]=0
    x = 0
    if ch != 0:

        for p in range(0, traj_set_prev.shape[2]):
            traj_data[:, :, p] = traj_set_prev[:, :, p]
            x = x + 1
    for i in range(x, tn):

        j = int(y[i])
        seq.append(j)
        reward = r[j, :]
        reward = reward_feature(M, N, reward).reshape(X, 1)
        z, r2 = get_z(reward, P_un)
        P = optimal_policy(P_un, z)

        traj = lmdp_gen_traj(X, P, z, tl)

        traj_data[:, :, i] = traj
    rewards_gt = np.zeros((F, 1))

    for k in seq:
        rewards_gt = np.append(rewards_gt, np.transpose(r[k, :]).reshape((F, 1)), axis=1)
    rewards_gt = rewards_gt[:, 1:]
    if ch != 0:
        rewards_gt = np.concatenate((rewards_prev, rewards_gt), axis=1)
        for q in range(0, len(y_prev)):
            y[q] = y_prev[q]

    # print("now", traj_data.shape)
    print("original assignment", y)
    return traj_data, rewards_gt, y





def lmdp_trajectories3(F, P_un, r, RF, tn, tl):
    # shape RF*F 3*16
    traj_data = np.zeros((tl, 2, tn))  # 6 trajectories
    seq = []
    traj_per_agent = int(tn / RF)
    y = np.arange(RF)
    y = np.repeat(y, traj_per_agent)
    np.random.shuffle(y)
    # y[0]=0
    for i in range(0, tn):
        j = int(y[i])
        seq.append(j)
        reward = r[j, :]
        reward = reward_feature(M, N, reward).reshape(X, 1)
        z, r2 = get_z(reward, P_un)
        P = optimal_policy(P_un, z)
        traj = lmdp_gen_traj(X, P, z, tl)
        traj_data[:, :, i] = traj
    rewards_gt = np.zeros((F, 1))

    for k in seq:
        rewards_gt = np.append(rewards_gt, np.transpose(r[k, :]).reshape((F, 1)), axis=1)
    rewards_gt = rewards_gt[:, 1:]


    # print("now", traj_data.shape)
    print("original assignment", y)
    return traj_data, rewards_gt, y



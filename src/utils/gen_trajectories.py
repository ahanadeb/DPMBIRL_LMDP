import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.gen_trajectories import *
from utils.params import *
import random


# def gen_trajectories(X, P, pi, tl):
#     P_pi = trans(P, pi)  # transition matrix of optimal policy
#     traj = np.zeros((tl, 2))
#     next_s = random.randint(0, X - 1)  # initial state
#
#     for i in range(0, tl):
#         states = (np.arange(X)).reshape((X,))
#         next_s = random.choices(states, weights=P_pi[next_s, :].reshape((X,)), k=1)
#         traj[i, 0] = next_s[0]
#         traj[i, 1] = np.argmax(pi[next_s, :])
#     return traj


def trajectory_set(F, RF, tn, tl):
    r = get_reward(F, RF)  # shape RF*F 3*16
    traj_data = np.zeros((tl, 2, tn))  # 6 trajectories
    seq = []
    traj_per_agent = int(tn / RF)
    y = np.arange(RF)
    y = np.repeat(y, traj_per_agent)
    np.random.shuffle(y)

    for i in range(0, tn):
        j = int(y[i])
        seq.append(j)
        reward = reward_feature(M, N, r[j, :]).reshape(X, 1)
        P = get_transitions(M, N, A, p, q, obstacles)
        V, V_hist, policy, time = policy_iteration(X, P, reward, A, gamma, max_iter=100)
        traj = gen_trajectories(X, P, policy, tl)
        traj_data[:, :, i] = traj
    rewards_gt = np.zeros((F, 1))
    for k in range(len(seq)):
        rewards_gt = np.append(rewards_gt, np.transpose(r[j, :]).reshape((F, 1)), axis=1)
    rewards_gt = rewards_gt[:, 1:]
    return traj_data, rewards_gt


def traj_form(traj):
    traj_states = traj[:, 0]
    states = np.unique(traj_states)
    traj_new = np.zeros((len(states), 3))
    traj_new[:, 0] = states
    for i in range(0, len(states)):
        for j in range(0, len(traj[:, 0])):
            if traj[j, 0] == states[i]:
                traj_new[i, 2] = traj_new[i, 2] + 1  # count of visit
                traj_new[i, 1] = traj[j, 1]  # policy at that state
    return traj_new


def traj_merj(traj_set, t):
    new = np.zeros((1, 3))

    for s in range(0, X):
        new_arr = np.zeros((1, 3))
        found = 0
        for i in range(len(t)):
            a = traj_set[:, :, int(t[i])]
            a = traj_form(a)
            if np.any(a[:, 0] == s):
                j = np.where(a[:, 0] == s)[0][0]
                new_arr[0, 0] = s
                new_arr[0, 1] = a[j, 1]
                new_arr[0, 2] = new_arr[0, 2] + a[j, 2]
                found = 1
        if found == 1:
            new = np.append(new, new_arr, axis=0)

    return new[1:, :]


def traj_form_lmdp(traj):
    traj_states = traj[:, 0]
    states = np.unique(traj_states)
    traj_new = np.zeros((len(states), 2))
    traj_new[:, 0] = states
    for i in range(0, len(states)):
        for j in range(0, len(traj[:, 0])):
            if traj[j, 0] == states[i]:
                traj_new[i, 1] = traj_new[i, 1] + 1  # count of visit
    return traj_new


def traj_merj_lmdp(traj_set, t):
    new = np.zeros((1, 2))
    for s in range(0, X):
        new_arr = np.zeros((1, 2))
        found = 0
        for i in range(len(t)):
            a = traj_set[:, :, int(t[i])]
            a = traj_form_lmdp(a)
            if np.any(a[:, 0] == s):
                j = np.where(a[:, 0] == s)[0][0]
                new_arr[0, 0] = s
                new_arr[0, 1] = new_arr[0, 1] + a[j, 1]
                found = 1
        if found == 1:
            new = np.append(new, new_arr, axis=0)

    return new[1:, :]

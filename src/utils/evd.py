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
from utils.log_post import *
from utils.saveHist import *
from utils.lmdp import *


def evd(hist, reward_gt, tn, P_un, y):
    # print("hist length", len(hist.policy))
    e = np.zeros((tn, 1))
    new_e = 0

    for i in range(0, tn):
        k = int(hist.assignment[i])

        # changed k to i here in the next line
        r1 = reward_feature(M, N, reward_gt[:, i]).reshape(X, 1)
    #    print("association", k, y[i])
        r2 = hist.reward[:, k]

    #    print("Rewards orig", reward_gt[:, i])
    #    print("reward est", r2)
        r2 = reward_feature(M, N, r2).reshape(X, 1)

        z_eval, r_n = get_z(r2, P_un)

        p = optimal_policy(P_un, z_eval)
        q1 = KL_matrix(p, P_un)
        v_eval = evaluate_analytical(p, r1 - q1, gamma)

        z_true, r_m = get_z(r1, P_un)
        p = optimal_policy(P_un, z_true)
        q1 = KL_matrix(p, P_un)

        v_true = evaluate_analytical(p, r1 - q1, gamma)
        v_true = np.dot(v_true.T, start)
        v_eval = np.dot(v_eval.T, start)
        e[i] = v_true - v_eval
    # e = e / tn
    ev = np.abs(np.mean(e))
    # print("e",e)
    new_e = np.abs(np.mean(new_e))
    print('v_true', v_true)
    print('v_eval', v_eval)
    return ev


def evd_emirl(rewards_gt,assignment, data,P_un,tn):
    e = np.zeros((tn, 1))
    for i in range(0, tn):
        j=int(assignment[i])

        # changed k to i here in the next line
        r1 = reward_feature(M, N, rewards_gt[:, i]).reshape(X, 1)
        r2 = reward_feature(M, N, data.weight[:, j]).reshape(X, 1)


        z_eval, r_n = get_z(r2, P_un)

        p = optimal_policy(P_un, z_eval)
        q1 = KL_matrix(p, P_un)
        v_eval = evaluate_analytical(p, r1 - q1, gamma)

        z_true, r_m = get_z(r1, P_un)
        p = optimal_policy(P_un, z_true)
        q1 = KL_matrix(p, P_un)

        v_true = evaluate_analytical(p, r1 - q1, gamma)
        v_true = np.dot(v_true.T, start)
        v_eval = np.dot(v_eval.T, start)
        e[i] = v_true - v_eval
    ev = np.abs(np.mean(e))
    return ev





def evd_single(r1, r2, P_un):
    r1 = reward_feature(M, N, r1).reshape(X, 1)
    r2 = reward_feature(M, N, r2).reshape(X, 1)

    z_eval, r_n = get_z(r2, P_un)

    p = optimal_policy(P_un, z_eval)
    q1 = KL_matrix(p, P_un)
    v_eval = evaluate_analytical(p, r1 - q1, gamma)

    z_true, r_m = get_z(r1, P_un)
    p = optimal_policy(P_un, z_true)
    q1 = KL_matrix(p, P_un)

    v_true = evaluate_analytical(p, r1 - q1, gamma)
    v_true = np.dot(v_true.T, start)
    v_eval = np.dot(v_eval.T, start)
    evd_v = np.mean(v_true - v_eval)

    return evd_v

import numpy as np
from utils.lmdp import *
from utils.reward import *
from utils.transition import *
import matplotlib.pyplot as plt
from utils.DPMHL import *
from utils.evd import *


def exp_lmdp():
    print("hey")
    P_un = uncontrolled_lmdp()
    np.set_printoptions(threshold=np.inf)

    EVD = []
    y = []
    i = 2
    r = get_reward_lmdp(F, RF)
    reward = reward_feature(M, N, r[0, :]).reshape(X, 1)
    z, r2 = get_z(reward, P_un)
    # P = optimal_policy(P_un, z)
    traj_test = lmdp_gen_traj(X, P_un, z, tl)
    while i < 7:
        total_maxiter
        tn = RF * i
        y.append(int(i))
        traj_set, rewards_gt, y2 = lmdp_trajectories(F, P_un, r,traj_test,RF, tn, tl)
        maxC = dpmhl(traj_set, total_maxiter, tn, P_un)
        e = evd(maxC, rewards_gt, total_maxiter, tn, P_un, y2)
        EVD.append(e)
        print("EVD = ", EVD)
        i = i + 1
    print("Completed. EVD = ", EVD)
    plt.plot(y, np.asarray(EVD))
    plt.xlabel('no. of trajectories per agent')
    plt.ylabel('EVD for the new trajectory')
    plt.savefig('figure.png')
    plt.show()


def exp_lmdp2(drive):
    print("hey")
    P_un = uncontrolled_lmdp()
    np.set_printoptions(threshold=np.inf)

    EVD = []
    y = []
    i = 2
    r = get_reward_lmdp(F, RF, drive)
    print("reward", r)

    traj_set_prev=0
    rewards_prev=0
    y_prev=0
    while i < 8:
        e_avg=0
        tn = RF * i
        y.append(int(i))
        traj_set, rewards_gt, y2 = lmdp_trajectories2(F, P_un, r,i-2,traj_set_prev,rewards_prev,y_prev,RF, tn, tl)

       # print("trajectories", traj_set.T)
        sample_nos=1
        for k in range(0,sample_nos):
            maxC = dpmhl(traj_set, total_maxiter, tn, P_un, rewards_gt,y2)
            e = evd(maxC, rewards_gt, tn, P_un, y2)
            #print("EVD = ", e)

            e_avg = e_avg + e/sample_nos

        EVD.append(e_avg)
        print("EVD averages = ", EVD)
        i = i + 1
        traj_set_prev = traj_set
        rewards_prev=rewards_gt
        y_prev=y2
    print("Completed. EVD = ", EVD)
    plt.plot(y, np.asarray(EVD))
    plt.xlabel('# of trajectories per agent')
    plt.ylabel('Average EVD')
    plt.savefig('figure.png')
    plt.show()
    #return y,EVD

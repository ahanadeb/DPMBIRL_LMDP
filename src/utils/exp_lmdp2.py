import numpy as np
from utils.lmdp import *
from utils.reward import *
from utils.transition import *
import matplotlib.pyplot as plt
from utils.DPMHL import *
from utils.evd import *



def exp_lmdp22():
    print("hey")
    P_un = uncontrolled_lmdp()
    np.set_printoptions(threshold=np.inf)


    r = get_reward_lmdp(F, RF)
    reward = reward_feature(M, N, r[0, :]).reshape(X, 1)
    z, r2 = get_z(reward, P_un)

    EVD = []
    y = []
    i = 2
    traj_set_prev=0
    rewards_prev=0
    y_prev=0
    while i < 9:
        tn = RF * i
        y.append(int(i))
        traj_set, rewards_gt, y2 = lmdp_trajectories3(F, P_un, r, RF, tn, tl)
        maxC = dpmhl(traj_set, total_maxiter, tn, P_un)
        e = evd(maxC, rewards_gt, total_maxiter, tn, P_un, y2)
        EVD.append(e)
        print("EVD = ", EVD)
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

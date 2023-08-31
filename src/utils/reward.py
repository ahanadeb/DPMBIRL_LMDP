import numpy as np
import random
import pandas as pd

def get_reward(F, RF):
    R = np.zeros((RF, F))-1
    for j in range(0, RF):
        for i in range(0, F):
            if random.random() < .2:
                R[j, i] = np.random.uniform(low=-1, high=+1)
    return R

def get_reward_lmdp(F, RF, drive):
    R = np.zeros((RF, F))
    p=0
    while p<RF:
        r = np.zeros((F,1))-1
        l = np.random.permutation(F)
        k = int(np.ceil(.1*F))
        idx = l[0:k]

        m= np.random.rand(k,1)-1


        j=0
        for i in idx:
            r[i,0] = m[j]
            j=j+1
        R[p,:] = np.transpose(r)
        if np.all(r==0):
            p=p-1
        p=p+1
    #print("orig reward", R)
    # path = './rewards_mod.csv'
    # if drive == 1:
    #     path = "/content/LMDP/rewards_mod.csv"
    # df = pd.read_csv(path, sep=',', header=None)
    # u=df.values
    u = R

    # pd.DataFrame(R).to_csv("rewards_mod2.csv", header=None, index=None)

    return u





def reward_feature(M, N, r):
    if r.shape[0] != 16:
        r = np.transpose(r)
    #r = np.transpose(r)
    reward = np.zeros((M, N))
    i = 1
    k = 0
    while i < M:
        j = 1
        while j < N:
            reward[i, j] = r[k]
            reward[i - 1, j] = r[k]
            reward[i, j - 1] = r[k]
            reward[i - 1, j - 1] = r[k]
            j = j + 2
            k = k + 1
        i = i + 2

    return reward

import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.cluster_assignment import *
from utils.update_weight import *
from tqdm import tqdm
from utils.saveHist import *
from utils.evd import *
import matplotlib.pyplot as plt



def dpmhl(traj_set, maxiter,tn, P_un,rewards_gt,y2):
    # initialisations
    C = Cluster()
    C = init_cluster(C, tn, F, X, P_un)
    C = relabel_cluster(C,tn)
    pr = calDPMLogPost(traj_set, C, P_un)
    maxC = MaxC()
    hist = Hist()
    hist = init_h(hist)
    maxC.logpost = -np.inf
    maxC, hist, bUpdate, h = saveHist(C, pr, maxC, hist)
    print('init pr = ', pr)

    evd_Arr =[]
    llh_arr=[]
    for i in range(0,maxiter):
        # first cluster update state
        x = np.random.randint(0, tn - 1, size=(1, tn))[0]

        for m in x:
            C = update_cluster(C, m, traj_set, P_un)
        C = relabel_cluster(C,tn)
        x = np.random.randint(0, int(np.max(C.assignment)) + 1, size=(1, int(np.max(C.assignment))))[0]
        Acc = 0
        for k in x:
            C, accepted,llhp = update_weight(k, traj_set, C, P_un)
            if accepted:
                Acc=1
           # print("here",k,C.assignment)

        if Acc==1:
            e = evd(C, rewards_gt, tn, P_un, y2)
           # print("EVD here", e)
            evd_Arr.append(np.abs(e))
            llh_arr.append(llhp)

            #if accepted:



        pr = calDPMLogPost(traj_set, C, P_un)
        maxC, hist, bUpdate, h = saveHist(C, pr, maxC, hist)
        #if bUpdate:
        print(i, 'th iteration, pr = ', pr, " ", maxC.logpost, " ", np.transpose(maxC.assignment))
    plt.plot(np.asarray(evd_Arr))
    plt.xlabel("iterations")
    plt.ylabel("average EVD")
    plt.show()
    plt.plot(np.asarray(llh_arr))
    plt.xlabel("iterations")
    plt.ylabel("log likelihood")
    plt.show()
    print("evd", evd_Arr)
    print("liklihood", llh_arr)

    return maxC

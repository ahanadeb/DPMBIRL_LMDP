import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.gen_trajectories import *
from utils.cluster_assignment import *
from utils.params import *
from utils.acc_ratio import *
from utils.DPMHL import *
import random
import sys
from utils.lmdp import *
from utils.reward import *
from utils.exp_lmdp import *
from utils.neighbours import *
from utils.torch_grad import *

def lmdp_plot():
    E=[]
    iter=10
    for i in range(0,iter):
        y,evd = exp_lmdp2()
        evd=np.asarray(evd)
        plt.plot(y, evd, color='gray',alpha=0.3)
        E.append(evd)
    E=np.asarray(E).reshape((evd.shape[0],iter))
    E=np.sum(E, axis=1) / iter
    plt.plot(y, E, color='blue')
    plt.xlabel('# of trajectories per agent')
    plt.ylabel('Average EVD')
    plt.savefig('figure.png')
    plt.show()
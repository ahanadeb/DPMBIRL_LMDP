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

def saveHist(C, pr, maxC, hist):
    bUpdate = False
    if maxC.logpost<pr:
        maxC.assignment = C.assignment
        maxC.reward = C.reward
        maxC.policy = C.policy
        maxC.values = C.values
        maxC.logpost = pr
        bUpdate = True
    h=update_h(hist, C, maxC, pr)


    return maxC, hist, bUpdate,h

class MaxC:
    assignment = []
    reward = []
    policy = []
    values = []
    logpost =[]

class Hist:
    assignment = []
    reward = []
    policy = []
    values = []
    llh = []
    prior = []
    gradL = []
    gradP = []
    maxLogPost = []
    logpost = []

def init_h(hist):
    hist.assignment = []
    hist.reward = []
    hist.policy = []
    hist.values = []
    hist.llh = []
    hist.prior = []
    hist.gradL = []
    hist.gradP = []
    hist.maxLogPost = []
    hist.logpost = []
    return hist

def update_h(hist,C, maxC, pr):
    hist.assignment.append(C.assignment)
    hist.reward.append(C.reward)
    hist.policy.append(C.policy)
    hist.values.append(C.values)
    hist.llh.append(C.llh)
    hist.prior.append(C.prior)
    hist.gradL.append(C.gradL)
    hist.gradP.append(C.gradP)
    hist.maxLogPost.append(maxC.logpost)
    hist.logpost.append(pr)
    return hist



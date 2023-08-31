import numpy as np
from tqdm import tqdm
import time
from utils.policies import *


def trans(P, pi):  # constructs X by X transition matrix for policy pi
    X = np.size(P, 0)
    P_pi = np.zeros((X, X))
    for m in range(0, X):
        for n in range(0, X):
            P_pi[m][n] = np.dot(P[m][n], pi[m])
    return P_pi


def scalarize(m, n, M, N):  # turns a pair of (m,n) coordinates into a flat 1-dimensional representation x
    x = N * m + n
    return x


def vectorize(x, M, N):  # turns a flat 1-dimensional representation x into a pair of (m,n) coordinates
    n = np.mod(x, N)
    m = (x - n) // N
    return (m, n)


def move(m, n, delta_m, delta_n, M, N,
         obstacles):  # from a given grid position (m,n), move in direction (delta_m,delta_n) if possible
    mtr = np.maximum(np.minimum(m + delta_m, M - 1), 0)
    ntr = np.maximum(np.minimum(n + delta_n, N - 1), 0)
    if obstacles[mtr, ntr] == 1:
        return scalarize(m, n, M, N)
    else:
        return scalarize(mtr, ntr, M, N)


def vector_to_matrix(value_vector, M,
                     N):  # turns a function on the state space represented as a flat X-dimensional vector to an M by N matrix
    X = N * M
    value_matrix = np.zeros((M, N))
    for x in range(0, X):
        [m, n] = vectorize(x, M, N)
        value_matrix[m, n] = value_vector[x]
        if obstacles[m, n] == 1:
            value_matrix[m, n] = np.nan
    return value_matrix


def matrix_to_vector(
        value_matrix):  # turns a function on the state space represented as an M by N matrix into a flat X-dimensional vector
    [M, N] = [np.size(value_matrix, 0), np.size(value_matrix, 1)]
    X = N * M
    value_vector = np.zeros((X))
    for x in range(0, X):
        [m, n] = vectorize(x, M, N)
        value_vector[x] = value_matrix[m, n]
    return value_vector


def evaluate_analytical(P_pi, r, gamma):  ### policy evaluation subroutine (analytical solution)
    X = np.size(P_pi, 0)
    I = np.eye(X)
    A = (I - gamma * P_pi)
    A_inverse = np.linalg.inv(A)
    # r = np.sum(r * pi, axis=1)
    # print("R ", r)
    value = A_inverse.dot(r)
    return value


def value_iteration(P, P_un, r, gamma, X):
    V = np.zeros(X)
    for i in range(0, 100):
        V_new = np.zeros((V.shape[0]))
        for m in range(0, X):
            V_new[m] = r[m] + gamma * (np.dot(P[m, :], V))
        V = V_new
    return V


def conv_policy(policy):
    # converts policy from one hot encoding to Xx1 array
    p = np.zeros((policy.shape[0], 1))
    brek
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            if policy[i, j] == 1:
                p[i, 0] = j

    return p


def policy_iteration(X, P, r, A, gamma, max_iter):
    # start = time.time()
    V_hist = np.zeros((X, max_iter))
    policy = np.zeros((X, A))  # + .5
    V_new = np.zeros((X))
    V = np.zeros((X))
    # print("Running policy iteration for ", max_iter, " iterations-")
    for i in range(0, max_iter):
        V = evaluate_analytical(P, policy, r, gamma)
        for m in range(0, X):
            v = np.zeros((A))
            for a in range(0, A):
                v[a] = r[m] + gamma * (np.dot(P[m, :, a], V))
            V_new[m] = np.max(v)

            policy[m] = np.zeros((A))
            policy[m, np.argmax(v)] = 1
        V = V_new
        V_hist[:, i] = V
    # print("Elapsed time: ", (time.time() - start), " seconds.\n")
    time = 0
    return V, V_hist, policy, time


def KL(P, Q):
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


def KL_matrix(p1, p2):
    assert p1.shape[0] == p2.shape[0], "unequal rows"
    kl = np.zeros((p1.shape[0], 1))
    for i in range(0, p1.shape[0]):
        kl[i] = KL(p1[i, :], p2[i, :])

    return kl


def det_p(p):
    X = p.shape[0]
    pol = np.zeros((X, X))
    for i in range(0, X):
        l = np.argmax(p[i, :])
        pol[i, l] = 1

    return pol

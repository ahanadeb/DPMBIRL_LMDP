import numpy as np
from utils.util_functions import *
from utils.params import *


def get_transitions(M, N, A, p, q , obstacles):
    X = M * N
    q = q/3
    P = np.zeros((X, X, A))
    for m in range(0, M):
        for n in range(0, N):
            x = scalarize(m, n, M, N)
            # go down: m++ if possible
            y = move(m, n, 1, 0, M, N, obstacles)
            P[x, y, 0] = P[x, y, 0] + p
            P[x, y, 1] = P[x, y, 1] + q
            P[x, y, 2] = P[x, y, 2] + q
            P[x, y, 3] = P[x, y, 3] + q
            # go right: n++ if possible
            y = move(m, n, 0, 1, M, N, obstacles)
            P[x, y, 0] = P[x, y, 0] + q
            P[x, y, 1] = P[x, y, 1] + p
            P[x, y, 2] = P[x, y, 2] + q
            P[x, y, 3] = P[x, y, 3] + q
            # go up: m-- if possible
            y = move(m, n, -1, 0, M, N, obstacles)
            P[x, y, 0] = P[x, y, 0] + q
            P[x, y, 1] = P[x, y, 1] + q
            P[x, y, 2] = P[x, y, 2] + p
            P[x, y, 3] = P[x, y, 3] + q
            # go left: n-- if possible
            y = move(m, n, 0, -1, M, N, obstacles)
            P[x, y, 0] = P[x, y, 0] + q
            P[x, y, 1] = P[x, y, 1] + q
            P[x, y, 2] = P[x, y, 2] + q
            P[x, y, 3] = P[x, y, 3] + p

    return P

def uncontrolled_lmdp():
    print("uncontrolled")
    P = get_transitions(M, N, A, p, q, obstacles)
    pi = np.zeros((X, A))
    J = np.random.choice(A, X)

    # assign with advanced indexing
    pi[np.arange(X), J] = 1
    P = trans(P, pi)
    #creating uncontrolled probab equal in all directions
    for i in range(0,X):
        s=0
        for j in range(0,X):
            if P[i,j] !=0:
                s = s +1
        for j in range(0,X):
            if P[i, j] != 0:
                P[i,j] = 1/s
    return P
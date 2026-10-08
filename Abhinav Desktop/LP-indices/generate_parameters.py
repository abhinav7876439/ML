"""
This document implements functions that generate parameters like random uniform, sparse, tri-diagonal matrices, or random initial conditions or reward vectors etc.


"""

import numpy as np
import random

def give_init(n):
    init = np.array([np.random.exponential() for i in range(n)])
    init /= sum(init)
    return init

def give_states(n,N,init):
    states = np.zeros(n,dtype=int)
    for i in range(n):
        ss = round(N*init[i])
        if ss + sum(states) <= N:
            states[i] = ss
        else:
            states[i] = ss-1
            break
    states[-1] = N - sum(states[:n-1])
    return states

# generate a tri-diagonal n dimensional transition matrix
def tri_matrix(n):
    P = np.zeros((n,n))
    P[0,0] = np.random.exponential()
    P[0,1] = np.random.exponential()
    P[n-1,n-2] = np.random.exponential()
    P[n-1,n-1] = np.random.exponential()
    for i in range(1,n-1):
        P[i,i-1] = np.random.exponential()
        P[i,i] = np.random.exponential()
        P[i,i+1] = np.random.exponential()
    for i in range(n):
        ss = np.sum(P[i])
        for j in range(n):
            P[i,j] /= ss
    return P

def tri_para(n):
    P0 = tri_matrix(n)
    P1 = tri_matrix(n)
    R0 = np.array([np.random.uniform() for i in range(n)])
    R1 = np.array([np.random.uniform() for i in range(n)])
    return P0,P1,R0,R1

# generate a full rank dense n-dimensional transition matrix
def dense_matrix(n):
    P = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            P[i,j] = np.random.exponential()
    for i in range(n):
        s = np.sum(P[i])
        for j in range(n):
            P[i,j] = P[i,j]/s
    return P

def dense_para(n):
    P0 = dense_matrix(n)
    P1 = dense_matrix(n)
    R0 = np.array([np.random.exponential() for i in range(n)])
    R1 = np.array([np.random.exponential() for i in range(n)])
    return P0,P1,R0,R1

def rested_para(n):
    P0 = np.identity(n)
    P1 = dense_matrix(n)
    R0 = np.array([np.random.exponential() for i in range(n)])
    R1 = np.array([np.random.exponential() for i in range(n)])
    return P0,P1,R0,R1

def sort_para(P0,P1,R0,R1,order):
    n = len(order)
    new_P0 = np.zeros((n,n))
    new_P1 = np.zeros((n,n))
    new_R0 = np.zeros(n)
    new_R1 = np.zeros(n)
    for i in range(n):
        for j in range(n):
            new_P0[i,j] = P0[order[i], order[j]]
            new_P1[i,j] = P1[order[i], order[j]]
        new_R0[i] = R0[order[i]]
        new_R1[i] = R1[order[i]]
    return new_P0, new_P1, new_R0, new_R1

def stable_para(n,alpha,tri=True):
    while True:
        if tri:
            P0,P1,R0,R1 = tri_para(n)
        else:
            P0,P1,R0,R1 = dense_para(n)
        plus = infinite_lp_thres(P0,P1,R0,R1,alpha)
        thres = plus[-1]
        P = np.copy(P0)
        for i in plus:
             P[i] = P1[i] - P1[thres] + P0[thres]
        eigens = np.linalg.eig(P)[0]
        norms = np.array([abs(eigens[i]) for i in range(n)])
        if max(norms) < 1+1e-8:
            order = infinite_lp_index(P0,P1,R0,R1,alpha)
            P0,P1,R0,R1 = sort_para(P0,P1,R0,R1,order)
            return P0,P1,R0,R1
        
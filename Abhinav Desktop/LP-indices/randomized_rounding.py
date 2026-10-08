"""
This document is the implementation of a randomized rounding algorithm that
can be found in Section 5.2.3 of the following paper:

https://dl.acm.org/doi/pdf/10.1145/2964791.2901467

The main function is the "randomized_rounding" function
"""
import numpy as np

def randomized_rounding(probas, states):
    """
    This function will return a list of possible activation vectors encoded in "possible_acts",
    and a probability vector "prob" encoding the probability of sampling each possible activation
    vector in "possible_acts".

    Args:
    - probas: a list of numbers between 0 and 1
    - states: a list of integer numbers encoding the number of bandits being in each state in set B
    """
    base = np.zeros(len(probas), dtype=int)
    reduced = []
    c = 0
    for i in range(len(probas)):
        s = probas[i][0]
        nb_s = states[s]
        p_s = probas[i][1]
        prod = nb_s*p_s
        base[i] = int(prod)
        reduced.append([prod-base[i],1])
        c += reduced[i][0]
    cc = c - reduced[-1][0]    
    c = int(round(c))
    reduced[-1][0] = abs(c - cc)
    acts,prob = rounding(c,reduced)
    possible_acts = []
    for item in acts:
        possible_acts.append(list(base+item))
    return possible_acts,prob


def transform(ans1, y):
    card = len(y)
    action = np.zeros(card,dtype=int)
    left = 0
    right = 0
    for i in range(card):
        left = right
        right += y[i][1]
        action[i] = sum(ans1[left:right])
    return list(action)

def find_ind(thres, s):
    N = len(s)
    if thres > s[N-1] - 1e-8:
        return N-1
    left = 0
    right = N-1
    current = int((left+right)/2)
    while right-left > 1:
        if s[current] > thres:
            right = current
            current = int((left+right)/2)
        else:
            left = current
            current = int((left+right)/2)
    return current

def placement(c, y):
    budget = sum(y)
    if c != round(budget) or abs(budget-round(budget)) > 1e-7:
        c = int(round(budget))
        #print("Error!")
        #return False
    N = len(y)
    s = np.zeros(N)
    t = np.zeros(N)
    tau = np.zeros(N)
    Tau = np.ones(N+1)
    sums = 0.
    for i in range(N):
        s[i] = sums
        t[i] = sums + y[i]
        if abs(t[i]-round(t[i])) < 1e-8:
            tau[i] = 0
        else:
            tau[i] = t[i] - int(t[i])
        sums = t[i]
    tau_sort = sorted(set(tau))
    K = len(tau_sort)
    Tau[:K] = tau_sort
    nu = []
    for k in range(N):
        x = np.zeros(N,dtype=int)
        for l in range(c):
            thres = l + Tau[k]
            ind = find_ind(thres,s)
            x[ind] = 1
        nu.append([x,Tau[k+1]-Tau[k]])
    return nu

def rounding(c,y):
    card = len(y)
    z = []
    for i in range(card):
        z.extend([y[i][0]]*y[i][1])
    ans = placement(c,z)
    possible_acts = []
    probas = []
    for item in ans:
        act = transform(item[0],y)
        if act in possible_acts:
            ind = possible_acts.index(act)
            probas[ind] += item[1]
        else:
            possible_acts.append(act)
            probas.append(item[1])
    return possible_acts,probas

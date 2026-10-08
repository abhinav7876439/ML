"""
This document assembles various utility funtions that are used in simulations.


"""
import numpy as np
import random


def sampling(proba_line):
    n = len(proba_line)
    seed = np.random.uniform(0,1)
    position = 0
    while sum(proba_line[0:position+1]) < seed:
        position += 1
    return position

def actives2(s,M):
    """
    return (i, a) where:
    - states 0 to i-1 are actives
    - state i is active with number of bandits a
    
    """
    i = 0
    while sum(s[:i]) < M:
        i += 1
    return i-1, M - sum(s[:i-1])

def simulate_one_step(states,M,P0,P1):
    i, a = actives2(states, M)
    n = len(states)
    data = []
    for j in range(i):
        data.append(np.random.multinomial(states[j], P1[j]))
    data.append(np.random.multinomial(a, P1[i]))
    data.append(np.random.multinomial(states[i]-a, P0[i]))
    for j in range(i+1,n):
        data.append(np.random.multinomial(states[j], P0[j]))

    print("Data : ",data)
    return sum(data)

def reward_one_step(states,M,R0,R1):
    i, a = actives2(states,M)
    if i < len(R0):
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    
def sort_data(P0,P1,R0,R1,states,order):
    n = len(order)
    new_P0 = np.zeros((n,n))
    new_P1 = np.zeros((n,n))
    new_R0 = np.zeros(n)
    new_R1 = np.zeros(n)
    new_states = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            new_P0[i,j] = P0[order[i], order[j]]
            new_P1[i,j] = P1[order[i], order[j]]
        new_R0[i] = R0[order[i]]
        new_R1[i] = R1[order[i]]
        new_states[i] = states[order[i]]
    return new_P0, new_P1, new_R0, new_R1, new_states

def simulate_one_step_order(states,M,P0,P1,R0,R1,order):
    n = len(order)
    P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,order)
    next_states = simulate_one_step(states,M,P0,P1)
    reward = reward_one_step(states,M,R0,R1)
    s = np.argsort(order)
    real_next = np.zeros(n, dtype=int)
    for i in range(n):
        real_next[i] = next_states[s[i]]
    return real_next,reward

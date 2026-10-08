"""
This document is the implementation of the update policy for singular problems. It solves the linear program and apply priority policy obtained from the solution upon reaching the first singular time. It then solves a new linear program at that moment. For singular of type 2 moments, it applies a randomized rounding.

Apart from that, there is also a window parameter "t" that we can tune, which is the number of time step that we impose the policy to apply an update, regardless of whether it is a singular time or not.

"""

from __future__ import print_function
import sys
import threading
from time import sleep
try:
    import thread
except ImportError:
    import _thread as thread

import numpy as np
from randomized_rounding import *
from simulation import *
import pulp as pulp
from pulp import *
from generate_parameters import sort_para

def quit_function(fn_name):
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt

def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

# The function "give_filling" solves the linear program and for every time 0 <= t <= T-1, returns a priority order if it is non-singular, and returns sets A,B,C as well as the capacity for all states in B if it is singular of type 2, i.e. Cardinal(B(t)) > 1. 
@exit_after(300)
def give_filling(P0,P1,R0,R1,T,init,alpha):
    n = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,n)
    horizon = range(0,T)
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y",(horizon,action,state),lowBound=0., upBound=1.)
    
    for t in horizon:
        prob += lpSum([variables[t][1][s] for s in state]) == alpha
    for t in range(0,T-1):
        for s in state:
            prob += variables[t+1][0][s] + variables[t+1][1][s] == lpSum([variables[t][a][ss]*P[a][ss][s] for a in action for ss                                                                          in state])
    for s in state:
        prob += variables[0][0][s] + variables[0][1][s] == init[s]
    
    prob += lpSum([variables[t][a][s]*R[a][s] for t in horizon for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))
    lambdas = []
    for name,c in list(prob.constraints.items())[:T]:
        lambdas.append(c.pi)
        
    V = np.zeros((T+1,n))
    Q = np.zeros((T,2,n))
    I = np.zeros((T,n))
    for t in horizon:
        t = T-t-1
        for a in action:
            for s in state:
                Q[t][a][s] = R[a][s] - a*lambdas[t] + sum(V[t+1][ss]*P[a][s][ss] for ss in state)
        for s in state:
            V[t][s] = max(Q[t][0][s], Q[t][1][s])
            I[t][s] = Q[t][1][s] - Q[t][0][s]
    ind = []
    for t in horizon:
        order = list(np.argsort(-I[t]))        
        A = []
        B = []
        C = []
        capa = []
        for i in order:
            if I[t][i] > 5*1e-7:
                A.append(i)
            elif abs(I[t][i]) < 5*1e-7:
                B.append(i)
                v = variables[t][1][i]
                V = v.varValue
                capa.append(V)
            else:
                C.append(i)
        if len(B) < 2:
            ind.append(order)
        else:
            ind.append([A,B,capa,C])
    return ind

# For a singular time of type 2, using A,B,C,capa obtained from function "give_filling", the function "fill" will return the water filling result for a states vector.
def fill(states,alpha,A,B,capa,C):
    N = sum(states)
    init = states/N
    active = 0.
    probas = []
    AA = []
    CC = list(np.copy(A))
    CC.extend(B)
    CC.extend(C)
    # By default, if init[s]==0, we put the state s in CC. This is also to avoid division by 0 error.
    total_A = 0.
    for s in A:
        total_A += init[s]
    #case 1
    if total_A >= alpha:
        for s in A:
            if init[s] > 1e-7:     
                if active + init[s] < alpha:
                    AA.append(s)
                    CC.remove(s)
                    active += init[s]
                else:
                    probas.append([s,(alpha-active)/init[s]])    
                    CC.remove(s)
                    return probas,AA,CC
        return probas,AA,CC
    else:
        for s in A:
            AA.append(s)
            CC.remove(s)
        active += total_A
        total_B = 0.
        for s in B:
            total_B += init[s]
        # modify the capacity to simplify coding    
        for i,s in enumerate(B):
            capa[i] = min(capa[i],init[s])
        #case 2, only need a first passage for the filling  
        if active + sum(capa) >= alpha:
            for i,s in enumerate(B):
                if init[s] > 1e-7:
                    if active + capa[i] < alpha:
                        probas.append([s,capa[i]/init[s]])
                        CC.remove(s)
                        active += capa[i]
                    else:
                        probas.append([s,(alpha-active)/init[s]])
                        CC.remove(s)
                        return probas,AA,CC
            return probas,AA,CC
        #case 3, a second passage of filling is necessary      
        elif active + total_B > alpha:
            active += sum(capa)
            for i,s in enumerate(B):
                if init[s] > 1e-7:
                    if active + init[s] - capa[i] < alpha:
                        AA.append(s)
                        CC.remove(s)
                        active += init[s] - capa[i]
                    elif active < alpha and active + init[s] - capa[i] >= alpha:
                        probas.append([s,(capa[i]+alpha-active)/init[s]])
                        CC.remove(s)
                        active = alpha
                    else:
                        probas.append([s,capa[i]/init[s]])
                        CC.remove(s)
            return probas,AA,CC
        #case 4
        else:
            for s in B:
                AA.append(s)
                CC.remove(s)
            active += total_B
            for s in C:
                if init[s] > 1e-7:
                    if active + init[s] < alpha:
                        AA.append(s)
                        CC.remove(s)
                        active += init[s]
                    else:
                        probas.append([s,(alpha-active)/init[s]])
                        CC.remove(s)
                        return probas,AA,CC
            return probas,AA,CC

def singular_type1(P0,P1,R0,R1,states,probas,A,C):
    d = len(R0)
    r = 0.
    data = []
    for s in A:
        data.append(np.random.multinomial(states[s], P1[s]))
        r += R1[s]*states[s]
    for s in C:
        data.append(np.random.multinomial(states[s], P0[s]))
        r += R0[s]*states[s]
    
    possible_acts,acts_prob = randomized_rounding(probas,states)   
    pp = sampling(acts_prob)
    action_choice = possible_acts[pp]
    for i,nb_active in enumerate(action_choice):
        s = probas[i][0]
        r += R1[s]*nb_active + R0[s]*(states[s]-nb_active)
        data.append(np.random.multinomial(nb_active, P1[s]))
        data.append(np.random.multinomial(states[s]-nb_active, P0[s]))
    return sum(data),r

# This function solves the linear program and returns the priorities before the first singular time of type 2, as well as the set A,B,C and the probabilities of activation of each state in set B 
@exit_after(300)
def singular_and_ind(P0,P1,R0,R1,T,init,alpha):
    n = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,n)
    horizon = range(0,T)
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y",(horizon,action,state),lowBound=0., upBound=1.)
    for t in horizon:
        prob += lpSum([variables[t][1][s] for s in state]) == alpha
    for t in range(0,T-1):
        for s in state:
            prob += variables[t+1][0][s] + variables[t+1][1][s] == lpSum([variables[t][a][ss]*P[a][ss][s] for a in action for ss                                                                          in state])
    for s in state:
        prob += variables[0][0][s] + variables[0][1][s] == init[s]
    prob += lpSum([variables[t][a][s]*R[a][s] for t in horizon for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))
    singular = 0
    while singular < T:
        cc = 0
        probas = []
        A = []
        C = []
        for i in range(n):
            v1 = variables[singular][0][i]
            v2 = variables[singular][1][i]
            if v1.varValue > 5*1e-7 and v2.varValue > 5*1e-7:
                cc+=1
                probas.append([i,v2.varValue/(v1.varValue+v2.varValue)])
            elif v1.varValue > 5*1e-7 and v2.varValue < 5*1e-7:
                C.append(i)
            else:
                A.append(i)
        if cc > 1:
            break
        singular += 1  
    lambdas = []
    for name,c in list(prob.constraints.items())[:T]:
        lambdas.append(c.pi)  
    V = np.zeros((T+1,n))
    Q = np.zeros((T,2,n))
    I = np.zeros((T,n))
    for t in horizon:
        t = T-t-1
        for a in action:
            for s in state:
                Q[t][a][s] = R[a][s] - a*lambdas[t] + sum(V[t+1][ss]*P[a][s][ss] for ss in state)
        for s in state:
            V[t][s] = max(Q[t][0][s], Q[t][1][s])
            I[t][s] = Q[t][1][s] - Q[t][0][s]
    ind = []
    for t in horizon[:singular]:
        ind.append(list(np.argsort(-I[t])))
    return singular,ind,probas,A,C

# simulate the update policy, "t" is a window parameter.
def update_sim(P0,P1,R0,R1,T,states,M,t=100000,printing=False):
    N = sum(states)
    alpha = M/N
    horizon = T
    reward = 0.
    while horizon > 0:
        init = np.array(states)/N
        singular,ind,probas,A,C = singular_and_ind(P0,P1,R0,R1,horizon,init,alpha)
        if printing:
            print("update happens at time-to-go " + str(horizon))
        if t > singular > 0:
            for order in ind:
                states,r = simulate_one_step_order(states,M,P0,P1,R0,R1,order)
                reward += r
            horizon -= singular
        elif singular == 0:
            states,r = singular_type1(P0,P1,R0,R1,states,probas,A,C)
            reward += r
            horizon -= 1   
        else:
            for order in ind[:t]:
                states,r = simulate_one_step_order(states,M,P0,P1,R0,R1,order)
                reward += r
            horizon -= t          
    return reward/(N*T)

# If window = 1, then updates at every decision epoch; if window > T, then this is just water-filling
def update_window_sim(P0,P1,R0,R1,T,states,M,window,printing=False):
    N = sum(states)
    n = len(R0)
    alpha = M/N
    init = states/N
    reward = 0. 
    horizon = T
    current_m = init
    while horizon > 0:
        ind = give_filling(P0,P1,R0,R1,horizon,current_m,alpha)
        true_window = min(horizon,window)
        for t,order in enumerate(ind[:true_window]):
            if len(order) != 4:
                states,r = simulate_one_step_order(states,M,P0,P1,R0,R1,order)
                reward += r
            else:
                A,B,capa,C = order
                probas,AA,CC = fill(states,alpha,A,B,capa,C)
                if len(probas) < 2:
                    ss = set(np.arange(n))
                    aa = set(AA)
                    cc = set(CC)
                    b = ss - (aa|cc)
                    BB = list(b)
                    order = list(np.copy(AA))
                    order.extend(BB)
                    order.extend(CC)
                    states,r = simulate_one_step_order(states,M,P0,P1,R0,R1,order)
                    reward += r  
                else:
                    states,r = singular_type1(P0,P1,R0,R1,states,probas,AA,CC)
                    reward += r   
        current_m = states/N
        horizon -= true_window
        if printing:
            print(horizon)
    return reward/(T*N)

#states are already sorted according to their LP indices
def infinite_sim(P0,P1,R0,R1,T,states,M):
    N = sum(states)
    reward = 0.
    for _ in range(T):
        r = reward_one_step(states,M,R0,R1)
        states = simulate_one_step(states,M,P0,P1)
        reward += r
    return reward/(T*N)

def greedy_sim(P0,P1,R0,R1,T,states,M):
    N = sum(states)
    n = len(R0)
    reward_diff = np.zeros(n)
    for i in range(n):
        reward_diff[i] = R1[i] - R0[i]
    order = list(np.argsort(-reward_diff))
    P0,P1,R0,R1 = sort_para(P0,P1,R0,R1,order)
    reward = 0.
    for _ in range(T):
        r = reward_one_step(states,M,R0,R1)
        states = simulate_one_step(states,M,P0,P1)
        reward += r
    return reward/(T*N)

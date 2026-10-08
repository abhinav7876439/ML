

import numpy as np
import pulp as pulp
from pulp import *
from simulation import *
from lp_solver import *
from update_policy import *

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
        
# # For a finite horizon restless bandit model, "filling_sim" will simulate the filling policy. It returns the time average bandit average reward as a performance measure. 
# def filling_sim(P0,P1,R0,R1,T,states,M):
#     N = sum(states)
#     n = len(R0)
#     alpha = M/N
#     init = states/N
#     ind = give_filling(P0,P1,R0,R1,T,init,alpha)
#     reward = 0.
#     for t,order in enumerate(ind):
#         if len(order) != 4:
#             states,r = simulate_one_step_order(states,M,P0,P1,R0,R1,order)
#             reward += r
#         else:
#             A,B,capa,C = order
#             probas,AA,CC = fill(states,alpha,A,B,capa,C)
#             if len(probas) < 2:
#                 ss = set(np.arange(n))
#                 aa = set(AA)
#                 cc = set(CC)
#                 b = ss - (aa|cc)
#                 BB = list(b)
#                 order = list(np.copy(AA))
#                 order.extend(BB)
#                 order.extend(CC)
#                 states,r = simulate_one_step_order(states,M,P0,P1,R0,R1,order)
#                 reward += r  
#             else:
#                 states,r = singular_type1(P0,P1,R0,R1,states,probas,AA,CC)
#                 reward += r   
#     return reward/(T*N)    

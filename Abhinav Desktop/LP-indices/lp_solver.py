"""
This document contains codes to solve the linear program, as well as calculating the relax upper (lower) bounds and the lp indices.

"""

import numpy as np
import pulp as pulp
from pulp import *
from update_policy import *

def solve_lp(P0,P1,R0,R1,T,init,alpha):
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
    return prob

def compute_indices(P0,P1,R0,R1,T,init,alpha):
    n = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,n)
    horizon = range(0,T)
    prob = solve_lp(P0,P1,R0,R1,T,init,alpha)
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
    indices = []
    for t in horizon:
        indices.append(I[t])
    return indices

def relax_upperbound(P0,P1,R0,R1,T,init,alpha):
    prob = solve_lp(P0,P1,R0,R1,T,init,alpha)
    rel = value(prob.objective)/T
    return rel

def relax_lowerbound(P0,P1,R0,R1,T,init,alpha):
    d = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,d)
    horizon = range(0,T)
    prob = LpProblem("LP1", LpMinimize)
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
    rel_lower = value(prob.objective)/T
    return rel_lower

def infinite_upper(P0,P1,R0,R1,alpha):
    d = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,d)       
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y",(action,state),lowBound=0, upBound=1.)
    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss]*P[a][ss][s] for a in action for ss in state])
    for s in state:
        prob += lpSum([variables[a][s] for a in action for s in state]) == 1.
    prob += lpSum([variables[a][s]*R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))
    return value(prob.objective)

def infinite_lower(P0,P1,R0,R1,alpha):
    d = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,d)       
    prob = LpProblem("LP1", LpMinimize)
    variables = LpVariable.dicts("Y",(action,state),lowBound=0, upBound=1.)
    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss]*P[a][ss][s] for a in action for ss in state])
    for s in state:
        prob += lpSum([variables[a][s] for a in action for s in state]) == 1.
    prob += lpSum([variables[a][s]*R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))
    return value(prob.objective)

def infinite_lp_index(P0,P1,R0,R1,alpha):
    n = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,n) 
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y",(action,state),lowBound=0., upBound=1.)
    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss]*P[a][ss][s] for a in action for ss in state])
    for s in state:
        prob += lpSum([variables[a][s] for a in action for s in state]) == 1.
    prob += lpSum([variables[a][s]*R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))    
    name,c = list(prob.constraints.items())[0]
    gamma = c.pi
    T = 1000
    V = np.zeros((T+1,n))
    Q = np.zeros((T,2,n))
    I = np.zeros(n)
    for t in range(T):
        t = T-t-1
        for a in action:
            for s in state:
                Q[t][a][s] = R[a][s] - a*gamma + sum(V[t+1][ss]*P[a][s][ss] for ss in state)
        for s in state:
            V[t][s] = max(Q[t][0][s], Q[t][1][s])
    for s in state:
        I[s] = Q[0][1][s] - Q[0][0][s]
        if abs(I[s]) < 5*1e-7:
            I[s] = 0
    return list(np.argsort(-I))

def infinite_fix_point(P0,P1,R0,R1,alpha):
    d = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,d)       
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y",(action,state),lowBound=0, upBound=1.)
    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss]*P[a][ss][s] for a in action for ss in state])
    for s in state:
        prob += lpSum([variables[a][s] for a in action for s in state]) == 1.
    prob += lpSum([variables[a][s]*R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))
    m_star = np.zeros(d)
    for i in range(d):
        V1 = variables[0][i]
        V2 = variables[1][i]
        v1 = V1.varValue
        v2 = V2.varValue
        m_star[i] = v1 + v2
    return m_star

def test_regularity(P0,P1,R0,R1,T,init,alpha):
    # solve the LP
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
    # Test regularity
    for t in horizon:
        count = 0
        for s in state:
            V1 = variables[t][1][s]
            V2 = variables[t][0][s]
            v1 = V1.varValue
            v2 = V2.varValue
            if abs(v1) > 5*1e-7 and abs(v2) > 5*1e-7:
                count += 1
        if count == 0:
            return False
    return True

def test_stability(P0,P1,R0,R1,alpha):
    n = len(R0)
    plus = infinite_lp_thres(P0,P1,R0,R1,alpha)
    thres = plus[-1]
    P = np.copy(P0)
    for i in plus:
        P[i] = P1[i] - P1[thres] + P0[thres]
    eigens = np.linalg.eig(P)[0]
    norms = np.array([abs(eigens[i]) for i in range(n)])
    if max(norms) > 1+1e-8:
        return False
    return True

def infinite_lp_thres(P0,P1,R0,R1,alpha):
    n = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,n) 
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y",(action,state),lowBound=0., upBound=1.)
    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss]*P[a][ss][s] for a in action for ss in state])
    for s in state:
        prob += lpSum([variables[a][s] for a in action for s in state]) == 1.
    prob += lpSum([variables[a][s]*R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))
    plus = []
    neutral = 0
    for i in range(n):
        V1 = variables[0][i]
        V2 = variables[1][i]
        v1 = V1.varValue
        v2 = V2.varValue
        if v2 > 1e-7 and v1 < 1e-7:
            plus.append(i)
        elif v2 > 1e-7 and v1 > 1e-7:
            neutral = i
    plus.append(neutral)
    return plus

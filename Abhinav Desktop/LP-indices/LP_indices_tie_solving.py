import numpy as np
import pulp as pulp
from pulp import *
from simulation import *
from lp_solver import *
from update_policy import *
from generate_parameters import *
from filling_policy import *
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed


def solve_ties_with_fix_priority(P0,P1,R0,R1,T,init,alpha,priority):
    n = len(R0)
    P = [P0,P1]
    R = [R0,R1]
    action = range(0,2)
    state = range(0,n)
    horizon = range(0,T)
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y",(horizon,action,state),lowBound=0.)
    for t in horizon:
        prob += lpSum([variables[t][1][s] for s in state]) == alpha
    for t in range(0,T-1):
        for s in state:
            prob += variables[t+1][0][s] + variables[t+1][1][s] == lpSum([variables[t][a][ss]*P[a][ss][s] for a in action for ss                                                                          in state])
    for s in state:
        prob += variables[0][0][s] + variables[0][1][s] == init[s]
    prob += lpSum([variables[t][a][s]*R[a][s] for t in horizon for a in action for s in state])
    prob.solve()
    ind = []
    for t in horizon: 
        A = []
        B = []
        C = []
        capa = []
        for i in priority:
            v1 = variables[t][1][i]
            v0 = variables[t][0][i]
            V1 = v1.varValue
            V0 = v0.varValue
            if V1 > 5*1e-7 and V0 < 5*1e-7:
                A.append(i)
            elif V1 < 5*1e-7 and V0 > 5*1e-7:
                C.append(i)
            else:
                B.append(i)
                capa.append(V1)
        if len(B) < 2:
            A.extend(B)
            A.extend(C)
            ind.append(A)
        else:
            ind.append([A,B,capa,C])
    return ind

def fix_order_sim(P0,P1,R0,R1,T,states,M,priority):
    N = sum(states)
    n = len(R0)
    alpha = M/N
    init = states/N
    ind = solve_ties_with_fix_priority(P0,P1,R0,R1,T,init,alpha,priority)
    reward = 0.
    for t,order in enumerate(ind):        
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
    return reward/(T*N) 

def test_with_one_parameter_set(T,n,myN,alpha,priorities,paras,para_nb,repeat):
    Filling_data = []
    Greedy_data = []
    Fix_order_data = []
    P0,P1,R0,R1,init,alpha = paras[para_nb]
    for N in myN:
        states = give_states(n,N,init)
        true_init = states/N
        M = int(round(alpha*N))
        upper = relax_upperbound(P0,P1,R0,R1,T,true_init,alpha)
        lower = relax_lowerbound(P0,P1,R0,R1,T,true_init,alpha)
        
        #fills = []
        #for _ in range(repeat):
        #    fill = filling_sim(P0,P1,R0,R1,T,states,M)
        #    fills.append(fill)
        fills = Parallel(n_jobs=-1,batch_size=5)(delayed(filling_sim)(P0,P1,R0,R1,T,states,M) for _ in range(repeat))    
        filling_perf = np.mean(fills)
        filling_score = 100*(filling_perf - lower)/(upper - lower)
        Filling_data.append(filling_score)
        
        reward_diff = np.zeros(n)
        for i in range(n):
            reward_diff[i] = R1[i] - R0[i]
        priority = list(np.argsort(-reward_diff))
        #greedys = []
        #for _ in range(repeat):
        #    greedy = fix_order_sim(P0,P1,R0,R1,T,states,M,priority)
        #    greedys.append(greedy)
        greedys = Parallel(n_jobs=-1,batch_size=5)(delayed(fix_order_sim)(P0,P1,R0,R1,T,states,M,priority) for _ in range(repeat))
        greedy_perf = np.mean(greedys)
        greedy_score = 100*(greedy_perf - lower)/(upper - lower)
        Greedy_data.append(greedy_score)
    
        mini_data = []
        for priority in priorities:
            #fix_orders = []
            #for _ in range(repeat):
            #    fix_order = fix_order_sim(P0,P1,R0,R1,T,states,M,priority)
            #    fix_orders.append(fix_order)
            fix_orders = Parallel(n_jobs=-1,batch_size=5)(delayed(fix_order_sim)(P0,P1,R0,R1,T,states,M,priority) for _ in range(repeat))
            fix_order_perf = np.mean(fix_orders)
            fix_order_score = 100*(fix_order_perf - lower)/(upper - lower)
            mini_data.append(fix_order_score)
        Fix_order_data.append(mini_data)
    
    temp = np.array(Fix_order_data)
    Fix_order_data = temp.transpose()
    
    return Filling_data,Greedy_data,Fix_order_data

def test_with_many_parameter_sets(T,n,myN,alpha,priorities,paras,repeat):
    full_data = []
    FILL = []
    GREEDY = []
    PRIO = []
    samples = len(paras)
    #compute for each para
    for i in tqdm(range(samples)):
        Filling_data,Greedy_data,Fix_order_data = test_with_one_parameter_set(T,n,myN,alpha,priorities,paras,i,repeat)
        FILL.append(Filling_data)   
        GREEDY.append(Greedy_data)
        PRIO.append(Fix_order_data)
    #process the data    
    temp = np.array(FILL)
    FILL = temp.transpose()
    filling_score_mean = []
    filling_score_var = []
    for list_of_scores_with_fix_N in FILL:
        score_mean = np.mean(list_of_scores_with_fix_N)
        score_var = 2*np.std(list_of_scores_with_fix_N)/np.sqrt(samples-1)
        filling_score_mean.append(score_mean)
        filling_score_var.append(score_var)
    full_data.append([filling_score_mean,filling_score_var])
    
    temp = np.array(GREEDY)
    GREEDY = temp.transpose()
    greedy_score_mean = []
    greedy_score_var = []
    for list_of_scores_with_fix_N in GREEDY:
        score_mean = np.mean(list_of_scores_with_fix_N)
        score_var = 2*np.std(list_of_scores_with_fix_N)/np.sqrt(samples-1)
        greedy_score_mean.append(score_mean)
        greedy_score_var.append(score_var)
    full_data.append([greedy_score_mean,greedy_score_var])

    for i in range(len(priorities)):
        priority_score = []
        for j in range(samples):
            priority_score.append(PRIO[j][i])
        temp = np.array(priority_score)
        priority_score = temp.transpose()
        priority_score_mean = []
        priority_score_var = []
        for list_of_scores_with_fix_N in priority_score:
            score_mean = np.mean(list_of_scores_with_fix_N)
            score_var = 2*np.std(list_of_scores_with_fix_N)/np.sqrt(samples-1)
            priority_score_mean.append(score_mean)
            priority_score_var.append(score_var)
        full_data.append([priority_score_mean,priority_score_var]) 
    return full_data

def para_and_init(n,alpha):
    P0,P1,R0,R1 = dense_para(n)
    init = give_init(n)
    return P0,P1,R0,R1,init,alpha

def test_one_para_with_one_prio(T,n,myN,alpha,prio,paras,para_nb,repeat):
    P0,P1,R0,R1,init,alpha = paras[para_nb]
    score = []
    for N in myN:
        states = give_states(n,N,init)
        true_init = states/N
        M = int(round(alpha*N))
        upper = relax_upperbound(P0,P1,R0,R1,T,true_init,alpha)
        lower = relax_lowerbound(P0,P1,R0,R1,T,true_init,alpha)
        fix_orders = Parallel(n_jobs=-1,batch_size=5)(delayed(fix_order_sim)(P0,P1,R0,R1,T,states,M,prio) for _ in range(repeat))
        fix_order_perf = np.mean(fix_orders)
        fix_order_score = 100*(fix_order_perf - lower)/(upper - lower)
        score.append(fix_order_score)
    return score

def test_all_paras_with_one_prio(T,n,myN,alpha,prio,paras,repeat):
    PRIO = []
    samples = len(paras)
    for i in range(samples):
        score_one_para = test_one_para_with_one_prio(T,n,myN,alpha,prio,paras,i,repeat)
        PRIO.append(score_one_para)
    temp = np.array(PRIO)
    PRIO = temp.transpose()
    prio_score_mean = []
    prio_score_var = []
    for list_of_scores_with_fix_N in PRIO:
        score_mean = np.mean(list_of_scores_with_fix_N)
        score_var = 2*np.std(list_of_scores_with_fix_N)/np.sqrt(samples-1)
        prio_score_mean.append(score_mean)
        prio_score_var.append(score_var)
    return prio_score_mean,prio_score_var

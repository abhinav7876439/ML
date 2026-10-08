import numpy as np
import csv
import datetime
import markovianbandit as bandit
import os
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD


M = 20    ## M = alpha*N   where N = sum(states)
#P0 = [[0.5,0,0,0.5], [0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5]]
#P1 = [[0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5], [0.5,0,0,0.5]]
#R0 = [-1,0,0,1]
#R1 = [-1,0,0,1] 
states = [33, 33, 34]     ## (representing the number of bandits in each state)
#order = [2, 1, 0, 3]      ## order = Activation States
T = 10000
N = sum(states)


# Circulant Input and Output file paths
input_file = "Non_whittle_input_parameters_1.csv"
output_files = "Circulant_output_results.txt"

 
# Log file path
log_file = output_files

# Input Parameters
num_states = 3  # Representing the number of bandits in each state

def read_parameters_from_csv(file_path):
    """Reads parameters from a CSV file and ensures data consistency."""
    P0, P1 = [], []
    R0, R1 = None, None
    alpha = None
    default_alpha = 0.2  # Default alpha value if not provided

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip the header row

        for row in reader:
            # Validate row length
            if len(row) < 4:
                raise ValueError(f"Incomplete row in CSV: {row}")

            try:
                # Parse RP0 and RP1
                P0.append([float(eval(value)) for value in row[0].split(",")])
                P1.append([float(eval(value)) for value in row[1].split(",")])

                # Parse Rr0 and Rr1 (ensure they are scalar values for each state)
                if R0 is None:
                    R0 = [float(eval(value)) for value in row[2].split(",")]
                if R1 is None:
                    R1 = [float(eval(value)) for value in row[3].split(",")]

                # Parse alpha (last column), if present
                if len(row) > 4 and row[4].strip():
                    alpha = float(eval(row[4]))
                else:
                    print(f"Warning: Missing alpha value in row {row}, using default alpha {default_alpha}")
                    alpha = default_alpha
            except Exception as e:
                raise ValueError(f"Error processing row {row}: {e}")

    # Convert RP0 and RP1 to NumPy arrays
    P0 = np.array(P0)
    P1 = np.array(P1)

    # Convert Rr0 and Rr1 to 1D NumPy arrays
    R0 = np.array(R0)
    R1 = np.array(R1)

    return P0, P1, R0, R1, alpha

P0, P1, R0, R1, alpha = read_parameters_from_csv(input_file)
print(f"CP0 = {P0}")
print(f"CP1 = {P1}")
print(f"CR0 = {R0}")
print(f"CR1 = {R1}")
print(f"Alpha = {alpha}")

model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
whittle_indices = model.whittle_indices()
print(f"Whittle Indices: {whittle_indices}")
whittle_order = np.argsort(-whittle_indices)
print("Whittle Order:", whittle_order)

def infinite_lp_index(P0, P1, R0, R1, alpha):
    """Calculate Whittle Indices using an LP relaxation approach."""
    n = len(R0)
    P = [P0, P1]
    R = [R0, R1]
    action = range(2)
    state = range(n)
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y", (action, state), lowBound=0.0, upBound=1.0)

    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss] * P[a][ss][s] for a in action for ss in state])
    prob += lpSum([variables[a][s] for a in action for s in state]) == 1.0
    prob += lpSum([variables[a][s] * R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))

    gamma = list(prob.constraints.values())[0].pi
    T = 1000
    V = np.zeros((T + 1, n))
    Q = np.zeros((T, 2, n))
    I = np.zeros(n)

    for t in range(T):
        t = T - t - 1
        for a in action:
            for s in state:
                Q[t][a][s] = R[a][s] - a * gamma + sum(V[t + 1][ss] * P[a][s][ss] for ss in state)
        for s in state:
            V[t][s] = max(Q[t][0][s], Q[t][1][s])
    for s in state:
        I[s] = Q[0][1][s] - Q[0][0][s]
        if abs(I[s]) < 5e-7:
            I[s] = 0

    return list(np.argsort(-I)), I


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

#D = actives2(states,M)
#print(f"Active States Asli wala = {D}")


def simulate_one_step(states,M,P0,P1):
    i, a = actives2(states, M)
    n = len(states)

    print("***********value of i : ",i)
    print("***********value of a : ",a)
    print("***********value of M : ",M)
    print("***********value of n : ",n)


    data = []
    for j in range(i):
        data.append(np.random.multinomial(states[j], P1[j]))
    data.append(np.random.multinomial(a, P1[i]))
    data.append(np.random.multinomial(states[i]-a, P0[i]))
    for j in range(i+1,n):
        data.append(np.random.multinomial(states[j], P0[j]))
    return sum(data)

#A = simulate_one_step(states, M, P0, P1)
#print("Data : ",A)

#new_states, M, new_R0, new_R1

def reward_one_step(states,M,R0,R1):
    i, a = actives2(states,M)
    if i < len(R0):                               # If total number of active states is less than the total number of states
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))      # average reward received
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    
#B = reward_one_step(states,M,R0,R1)
    
# It is necessary to reorder the states, transitions and rewards according to a specific policy or sorting criterion
def sort_data(P0,P1,R0,R1,states,order):

    # Convert input matrices to NumPy arrays if they are not already
    P0 = np.array(P0)
    P1 = np.array(P1)
    R0 = np.array(R0)
    R1 = np.array(R1)
    states = np.array(states)

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

#C = sort_data(P0,P1,R0,R1,states,order)

lp_order, lp_indices = infinite_lp_index(P0, P1, R0, R1, alpha)
print(f"lp order = {lp_order}")
print(f"lp indices = {lp_indices}")

#states are already sorted according to their LP indices
def infinite_sim(P0,P1,R0,R1,T,states,M):
    N = sum(states)
    total_reward = []
    #transition_states = states
    reward = 0

    new_P0, new_P1, new_R0, new_R1, transition_states = sort_data( P0, P1, R0, R1, states, lp_order )
    
    for _ in range(1,T+1):
        
        r = reward_one_step(transition_states, M, new_R0, new_R1)
        print(r)
        transition_states = simulate_one_step(transition_states, M, new_P0, new_P1)
        print("Next states :", transition_states)
        reward += r
        print("Reward :", reward)
        #print(reward / (T * 5))
        total_reward.append(r)
    return total_reward, reward/(T)

#total_avg_reward, avg_rew = infinite_sim(new_P0, new_P1, new_R0, new_R1, T, new_states,M)
#print("Avg Reward per arm : ",P)
#print("Total average reward ", total_avg_reward)


total_avg_reward, avg_rew = infinite_sim(P0, P1, R0, R1, T, states, M)

def plt_graph( rewards, total_time):

    cum_rewards = np.cumsum(rewards)
    time_steps = np.arange(1, len(rewards) + 1)
    running_time_avg_reward = cum_rewards / time_steps

    print("*************************** Expected Reward : ", np.mean(running_time_avg_reward))
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(total_time), running_time_avg_reward, label="Running time cumulative rewards", color="blue")
    #plt.plot(range(len(whittle_moving_avg_rewards)), whittle_moving_avg_rewards, label="Whittle Moving Avg Rewards", color="red")
    plt.xlabel("Time Steps")
    plt.ylabel("running time average Reward")
    plt.title(f"Average Rewards for Non -whittle (M={M}, N={sum(states)})")
    plt.grid()
    plt.legend()
    plt.show()

plt_graph(total_avg_reward, T)


def simulate_one_step_order(states,M,P0,P1,R0,R1,order):
    n = len(order)
    sorted_P0,sorted_P1,sorted_R0,sorted_R1,sorted_states = sort_data(P0,P1,R0,R1,states,order)
    print("Sorted states :", sorted_states)
    next_states = simulate_one_step(sorted_states,M,sorted_P0,sorted_P1)
    print("Next states :", next_states)
    reward = reward_one_step(sorted_states,M,sorted_R0,sorted_R1)
    print("Reward :", reward)
    s = np.argsort(order)
    print("s :", s)
    real_next = np.zeros(n, dtype=int)
    for i in range(n):
        real_next[i] = next_states[s[i]]
        print("Real Next states :", real_next[i])
    return real_next,reward
#Q = simulate_one_step_order(states,M,P0,P1,R0,R1,order)
#print("One step Order: ",Q)


# def simulate_and_plot_details(T, states, M, P0, P1, R0, R1, order):
#     """
#     Simulates the system for T time steps, prints states and rewards before and after sorting,
#     and plots the number of arms in each state before and after sorting once.

#     Parameters:
#         T (int): Total time steps.
#         initial_states (list): Initial state distribution.
#         M (int): Total active resources.
#         P0, P1: Transition probability matrices.
#         R0, R1: Reward vectors.
#         order (list): Sorting order based on Whittle index or custom criterion.

#     Returns:
#         rewards (list): Rewards at each time step.
#     """
#     rewards = []
#     current_states = np.array(states)

#     # Initialize lists to store states for plotting
#     states_before_sorting = []
#     states_after_sorting = []

#     for t in range(T):
#         # Print current step
#         print(f"\nTime Step {t + 1}:")

#         # Before sorting
#         print(f"States before sorting: {current_states}")
#         states_before_sorting.append(current_states.copy())

#         # Sort and transition states
#         P0_sorted, P1_sorted, R0_sorted, R1_sorted, sorted_states = sort_data(P0, P1, R0, R1, current_states, order)
#         print(f"States after sorting: {sorted_states}")
#         states_after_sorting.append(sorted_states.copy())

#         # Calculate rewards
#         reward = reward_one_step(sorted_states, M, R0_sorted, R1_sorted)
#         print(f"Reward: {reward}")
#         rewards.append(reward)

#         # Reverse sorting to realign states
#         reverse_order = np.argsort(order)
#         real_next_states = np.zeros(len(order), dtype=int)
#         for i in range(len(order)):
#             real_next_states[i] = simulate_one_step(sorted_states, M, P0_sorted, P1_sorted)[reverse_order[i]]

#         # Update current states
#         current_states = real_next_states

#     # Convert states to arrays for plotting
#     states_before_sorting = np.array(states_before_sorting)
#     states_after_sorting = np.array(states_after_sorting)

#     return rewards

# # Simulate and plot details

# rewards = simulate_and_plot_details(T, C, M, P0, P1, R0, R1, order)
# print(f"Total rewards over {T} steps: {np.sum(rewards):.4f}")
# print(f"Average reward per step: {np.mean(rewards):.4f}")

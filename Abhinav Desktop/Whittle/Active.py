
# import csv
# import datetime
# import markovianbandit as bandit
# import os
# import numpy as np

# """Code to update the policy and generate trajectories."""

M = 16    ## M = alpha*N   where N = sum(states)
# CP0 = [[0.5,0,0,0.5], [0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5]]
# CP1 = [[0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5], [0.5,0,0,0.5]]
# CR0 = [-1,0,0,1]
# CR1 = [-1,0,0,1] 
states = [5,3,7,4]     ## (representing the number of bandits in each state)
order = [2, 3, 1, 0]      ## order = Activation States
T = 10
N = sum(states)
# epochs = 100



import csv
import datetime
import markovianbandit as bandit
import os
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD

# Circulant Input and Output file paths
input_file = "Circulant_input_parameters.csv"
output_files = "Transitions.txt"

 
# Log file path
log_file = output_files

# # Input Parameters
# N = 100
# num_states = 4  # Representing the number of bandits in each state
# T = 10
# epochs = 1
# Num_arms = [10, 20, 30, 40, 50, 60, 70, 100, 200, 400, 600, 800, 1000, 1350, 1750, 2000]
# # order = []  # Activation states, will be determined dynamically


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

CP0, CP1, CR0, CR1, alpha = read_parameters_from_csv(input_file)
print(f"CP0 = {CP0}")
print(f"CP1 = {CP1}")
print(f"CR0 = {CR0}")
print(f"CR1 = {CR1}")
print(f"Alpha = {alpha}")

# M = int(alpha * N)  # Total active resources
# print(f"Total active resources = {M}")







# def actives2(states, M):   ##  How active resources (M) are allocated among state ## states = [5, 3, 7, 4] 
#     """
#     Returns (i, a) where:
#     - states 0 to i-1 are active
#     - state i is active with number of bandits a
#     """
#     i = 0
#     while i < len(states) and sum(states[:i]) < M:
#         i += 1
#     if i == 0:
#         return 0, M
#     elif i >= len(states):
#         return len(states) - 1, states[-1]
#     else:
#         return i - 1, M - sum(states[:i - 1])    ## i: Index of the last state that receives active resources,   a: Number of active resources allocated to state i
    


# # states = [5, 3, 7, 4] # (initial bandit distribution across states).
# # M = 16 # (total active resources)
# P = actives2(states,M)
# print(f"Active States = {P}")



# def actives2(states, M):   ##  How active resources (M) are allocated among state ## states = [5, 3, 7, 4]
#     """
#     Returns (i, a) where:
#     - states 0 to i-1 are fully active
#     - state i is active with number of bandits a
#     """
#     i = 0
#     while i < len(states) and sum(states[:i]) < M:
#         i += 1
#     if i == 0:  # If no states are fully active, allocate all resources to state 0
#         return 0, M
#     elif sum(states[:i]) >= M:  # M is partially allocated to state i
#         return i - 1, M - sum(states[:i - 1])
#     else:  # If M exceeds total resources, allocate the remainder to the last state
#         return len(states) - 1, M - sum(states[:-1])    ## i: Index of the last state that receives active resources,   a: Number of active resources allocated to state i

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

D = actives2(states,M)
print(f"Active States Asli wala = {D}")




# def simulate_one_step(states,M,P0,P1):    ## It simulates how the states of bandits evolve after one decision step, given the allocation of active resources and the state-transition probabilities
#     i, a = actives2(states, M)              ## It returns the new number of bandits in each state after transitions
#     n = len(states)
#     data = []
#     for j in range(i):
#         data.append(np.random.multinomial(states[j], P1[j]))
#     data.append(np.random.multinomial(a, P1[i]))
#     data.append(np.random.multinomial(states[i]-a, P0[i]))
#     for j in range(i+1,n):
#         data.append(np.random.multinomial(states[j], P0[j]))
#     return sum(data)    # Compute the total transition outcomes using sum(data), returning the resulting "next state".


# B = simulate_one_step(states,M,CP0,CP1)
# print(f"Transition = {B}")



def fix_simulate_one_step(states, M, P0, P1):
    """
    Simulates one step of state transitions for bandits.

    Args:
        states (list): The number of bandits in each state.
        M (int): The total number of active resources available.
        P0 (list): Transition probabilities for inactive bandits.
        P1 (list): Transition probabilities for active bandits.

    Returns:
        list: The new states after one step.
    """
    i, a = actives2(states, M)  # Select the active state `i` and the number of active resources `a`
    print("****************** States is : ",states)
    print("********************* i is : ",i)
    print("********************* M is : ",M)
    print("***************a is : ",a)
    n = len(states)
    data = []

    # Handle transitions for states
    for j in range(i):
        data.append(np.random.multinomial(states[j], P1[j]))  # Fully active states before `i`
    if a > 0:
        # If there are active resources for state `i`
        active_bandits = np.random.multinomial(min(a, states[i]), P1[i])  # Active portion
        inactive_bandits = np.random.multinomial(max(0, states[i] - a), P0[i])  # Inactive portion
        data.append(active_bandits + inactive_bandits)
    else:
        # If no active resources for state `i`, treat it as inactive
        data.append(np.random.multinomial(states[i], P0[i]))
    for j in range(i + 1, n):
        data.append(np.random.multinomial(states[j], P0[j]))  # Fully inactive states after `i`

    # Compute the total transition outcomes
    return np.sum(data, axis=0).tolist()


Z = fix_simulate_one_step(states,M,CP0,CP1)
print(f"Transition = {Z}")


def reward_one_step(states,M,R0,R1):
    i, a = actives2(states,M)
    if i < len(R0):                               # If total number of active states is less than the total number of states
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))      # average reward received
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    

# CR0 = [-1,0,0,1]
# CR1 = [-1,0,0,1]   ## Figure out the changes when it is not symmetric
C = reward_one_step(states,M,CR0,CR1)
print(f"Reward: {C}")


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


# CP0 = [[0.5,0,0,0.5], [0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5]]
# CP1 = [[0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5], [0.5,0,0,0.5]]
# CR0 = [-1,0,0,1]
# CR1 = [-1,0,0,1] 
# states = [5, 3, 7, 4]     ## (representing the number of bandits in each state)
# order = [2, 1, 3, 0]
D = sort_data(CP0,CP1,CR0,CR1,states,order)

print(D)
# print(CP0)
# print(CP1)
# print(CR0)
# print(CR1)
# print(states)
# print(order)


def simulate_one_step_order(states,M,P0,P1,R0,R1,order):
    n = len(order)
    P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,order)
    next_states = fix_simulate_one_step(states,M,P0,P1)
    print("Next States:", next_states)
    reward = reward_one_step(states,M,R0,R1)
    s = np.argsort(order)
    print("S:", s)
    real_next = np.zeros(n, dtype=int)
    for i in range(n):
        real_next[i] = next_states[s[i]]
        print(f"Real Ram: {real_next[i]}")
    return real_next,reward

E = simulate_one_step_order(states,M,CP0,CP1,CR0,CR1,order)
real_next, reward = E
# Print real_next and reward
print("Real Next States:", real_next)
print("Reward:", reward)
print(E)




# def simulate_one_step_1(states, M, P0, P1):
#     """
#     Simulate one step of the process with normalized probabilities.
#     Handles cases where row sums are zero.
#     """
#     i, a = actives2(states, M)
#     n = len(states)
#     data = []

#     # Normalize P0 and P1 safely, replacing zero rows with uniform distribution
#     def normalize_matrix(matrix):
#         normalized_matrix = []
#         for row in matrix:
#             row_sum = row.sum()
#             if row_sum == 0:
#                 # Replace zero rows with a uniform distribution
#                 normalized_matrix.append(np.ones_like(row) / len(row))
#             else:
#                 normalized_matrix.append(row / row_sum)
#         return np.array(normalized_matrix)

#     P0 = normalize_matrix(P0)
#     P1 = normalize_matrix(P1)

#     for j in range(i):
#         data.append(np.random.multinomial(states[j], P1[j]))
#     data.append(np.random.multinomial(a, P1[i]))
#     data.append(np.random.multinomial(states[i] - a, P0[i]))
#     for j in range(i + 1, n):
#         data.append(np.random.multinomial(states[j], P0[j]))

#     return sum(data)


# def validate_probability_matrix(matrix):
#     """Ensure no rows in the matrix sum to zero."""
#     for row in matrix:
#         if row.sum() == 0:
#             raise ValueError("Probability matrix contains a row that sums to zero.")


# P0, P1, R0, R1 = read_parameters_from_csv(circulant_input_file)
# validate_probability_matrix(P0)
# validate_probability_matrix(P1)

# if row.sum() == 0:
#     print(f"Zero-sum row detected in matrix: {row}")




# # Function to simulate infinite horizon process for one epoch
# def infinite_sim(P0, P1, R0, R1, T, states, M):
#     """Simulate the infinite horizon process for one epoch."""
#     N = sum(states)
#     reward = 0.0
#     for _ in range(T):
#         P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,order)
#         r = reward_one_step(states, M, R0, R1)
#         states = fix_simulate_one_step(states, M, P0, P1)
#         reward += r
#     return reward / (T * N)

#states are already sorted according to their LP indices
def infinite_sim(P0,P1,R0,R1,T,states,M):
    N = sum(states)
    reward = 0.
    for _ in range(T):
        r = reward_one_step(states,M,R0,R1)
        states = fix_simulate_one_step(states,M,P0,P1)
        reward += r
    return reward/(T*N)


F = infinite_sim(CP0,CP1,CR0,CR1,T,states,M)
print("Cum_reward:", F)



# def simulate_and_plot_details(T, initial_states, M, P0, P1, R0, R1, order):
#     rewards = []
#     N = sum(states)
#     current_states = np.array(initial_states)

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
#         print(f"Reverse Order: {reverse_order}")
#         real_next_states = np.zeros(len(order), dtype=int)
#         print("Real Next States:", real_next_states)
#         for i in range(len(order)):
#             real_next_states[i] = fix_simulate_one_step(sorted_states, M, P0_sorted, P1_sorted)[reverse_order[i]]
#             print("Ram:", real_next_states[i])

#         # Update current states
#         current_states = real_next_states
#         print("Current States:", current_states)

#     # Convert states to arrays for plotting
#     states_before_sorting = np.array(states_before_sorting)
#     states_after_sorting = np.array(states_after_sorting)

#     return rewards

# # Simulate and plot details

# rewards = simulate_and_plot_details(T, states, M, CP0, CP1, CR0, CR1, order)
# print(f"Total rewards over {T} steps: {np.sum(rewards):.4f}")
# print(f"Average reward per step: {np.mean(rewards):.4f}")





# # Function to run 100 epochs and calculate the average reward with plot
# def run_simulation_for_epochs(P0, P1, R0, R1, T, states, M, epochs):
#     """Run the simulation for the specified number of epochs and log the results, also plot average reward."""
#     total_reward = 0.0
#     epoch_rewards = []  # List to store rewards for each epoch
    
#     for epoch in range(epochs):
#         # Reset states for each epoch
#         current_states = states[:]
#         epoch_reward = infinite_sim(P0, P1, R0, R1, T, current_states, M)
#         total_reward += epoch_reward
#         epoch_rewards.append(epoch_reward)  # Store the reward for this epoch

    
#     average_reward = total_reward / epochs
#     return average_reward

# G = run_simulation_for_epochs(CP0, CP1, CR0, CR1, T, states, M, epochs)
# print("Avg_reward:", G)



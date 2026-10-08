import csv
import datetime
import markovianbandit as bandit
import os
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD

# Circulant Input and Output file paths
input_file = "Transition.csv"
output_files = "Circulant_output_results.txt"

 
# Log file path
log_file = output_files

# Input Parameters
N = 100
num_states = 4  # Representing the number of bandits in each state
T = 500
epochs = 2
Num_arms = [10, 20, 30, 40, 50, 60, 70, 100, 200, 400, 600, 800, 1000, 1350, 1750, 2000]
# order = []  # Activation states, will be determined dynamically


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

M = int(alpha * N)  # Total active resources
print(f"Total active resources = {M}")



def log_output(output, file_path):
    """Log output with timestamp to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as logfile:
        if output:  # Only add a timestamp if output is not empty
            logfile.write(f"[{timestamp}] {output}\n")
        else:  # Add a blank line without a timestamp
            logfile.write("\n")


def write_output_to_text_file(file_path, whittle_indices, alpha):
    """Write output to a text file."""
    with open(file_path, "a") as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Date and Time: {current_time}\n")
        file.write(f"Alpha: {alpha}\n")
        file.write(f"Whittle Indices: {whittle_indices}\n\n")
    print(f"Output successfully written to text file: {file_path}")



def generate_uniform_states(num_arms, num_states):
    """
    Distributes the total number of bandits (N) uniformly across the given number of states (num_states).
    
    Parameters:
        N (int): Total number of bandits.
        num_states (int): The number of states to distribute the bandits across.
        
    Returns:
        list: A list representing the number of bandits in each state, with total sum equal to N.
    """
    base_bandits = num_arms // num_states
    remaining_bandits = num_arms % num_states

    states = [base_bandits] * num_states
    for i in range(remaining_bandits):
        states[i] += 1

    return states

C = generate_uniform_states(N, num_states)
print(f"States = {C}")

# C = [0,0,0,100]
# print(f"States = {C}")



# def distribute_arms_exponential(total_arms, num_states, scale=1.0):
#     """
#     Distribute a total number of arms among states based on an exponential distribution.

#     Parameters:
#     - total_arms (int): The total number of arms to distribute.
#     - num_states (int): The number of states.
#     - scale (float): The scale parameter for the exponential distribution (default is 1.0).

#     Reurns:
#     - arms (list): A list where each element represents the number of arms in a state.
#     """
#     # Generate raw exponential values for each state
#     raw_values = np.random.exponential(scale, num_states)
#     # Normalize to ensure they sum to 1
#     normalized_values = raw_values / np.sum(raw_values)
#     # Scale by the total number of arms and round to integers
#     arms = np.round(normalized_values * total_arms).astype(int)
    
#     # Adjust to ensure the total number of arms is exactly `total_arms`
#     while sum(arms) < total_arms:
#         arms[np.argmax(normalized_values)] += 1  # Add to the state with the largest normalized value
#     while sum(arms) > total_arms:
#         arms[np.argmax(arms)] -= 1  # Subtract from the state with the largest number of arms

#     return arms

# C = distribute_arms_exponential(100, 4)
# print(f"States = {C}")


def pre_rounding(states):
    rounded_states = np.floor(states).astype(int)  # Start with the floor of each value
    fractional_parts = states - rounded_states
    deficit = int(round(sum(states))) - sum(rounded_states)  # Adjust to maintain the total

    # Randomly round up based on fractional parts
    if deficit > 0:
        indices = np.argsort(-fractional_parts)[:deficit]
        rounded_states[indices] += 1

    return rounded_states

B = pre_rounding(C)
print(f"Rounded States = {B}")


# Initialize model and calculate Whittle indices
model_circulant = bandit.restless_bandit_from_P0P1_R0R1(CP0, CP1, CR0, CR1)
whittle_indices = model_circulant.whittle_indices()
print(f"Whittle Indices: {whittle_indices}")
whittle_order = np.argsort(-whittle_indices)
print("Whittle Order:", whittle_order)



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

G = sort_data(CP0,CP1,CR0,CR1,C,whittle_order)
print(f"Sorted States according to Whittle Priority Policy = {G}")
new_CP0, new_CP1, new_CR0, new_CR1, new_states = G
print(f"Number of arms after sorting (Sorted States) according to Whittle Priority Policy = {new_states}")


def actives2(states, M):
    """
    Returns (i, a) where:
    - states 0 to i-1 are fully active
    - state i is active with number of bandits a

    Raises an error if M exceeds the total number of bandits (sum(states)).
    """
    # Check if the budget exceeds the total number of arms
    """if M > sum(states):
        raise ValueError("The budget M exceeds the total number of bandits (sum(states)).")"""

    i = 0
    while i < len(states) and sum(states[:i]) < M:
        i += 1

    if i == 0:  # If no states are fully active, allocate all resources to state 0
        return 0, M, -1
    elif sum(states[:i]) >= M:  # M is partially allocated to state i
        return i - 1, M - sum(states[:i - 1]), 0
    else:  # If M exceeds total resources (unlikely with the above check)
        return len(states) - 1, M - sum(states[:-1]), 1  ## i: Index of the last state that receives active resources,   a: Number of active resources allocated to state i
    
D = actives2(new_states,M)
print(f"Active States = {D}")

def simulate_one_step(states,M,P0,P1):    ## It simulates how the states of bandits evolve after one decision step, given the allocation of active resources and the state-transition probabilities
    i, a, check_res = actives2(states, M)              ## It returns the new number of bandits in each state after transitions
    n = len(states)
    data = []
    if check_res == 0:
        for j in range(i):
            data.append(np.random.multinomial(states[j], P1[j]))
        data.append(np.random.multinomial(a, P1[i]))
        data.append(np.random.multinomial(states[i]-a, P0[i]))
        for j in range(i+1,n):
            data.append(np.random.multinomial(states[j], P0[j]))
        return sum(data)    # Compute the total transition outcomes using sum(data), returning the resulting "next state".
    elif check_res == -1:
        data.append(np.random.multinomial(states[0], P1[0]))
        for j in range(1, n):
            data.append(np.random.multinomial(states[j], P0[j]))
    elif check_res == 1:
        raise IOError("Less resources")


E = simulate_one_step(new_states,M,new_CP0,new_CP1)
print(f"After Transition = {E}")


def reward_one_step(states,M,R0,R1):
    i, a, res = actives2(states,M)
    if i < len(R0):                               # If total number of active states is less than the total number of states
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))      # average reward received
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    
F = reward_one_step(new_states,M,new_CR0,new_CR1)
print(f"One step Reward = {F}")




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

J = simulate_one_step_order(C,M,CP0,CP1,CR0,CR1,whittle_order)
real_next, reward = J
# Print real_next and reward
print("Real Next States:", real_next)
print("Reward:", reward)



def simulate_and_plot_details(T, initial_states, M, P0, P1, R0, R1, order):
    """
    Simulates the system for T time steps, prints states and rewards before and after sorting,
    and plots the number of arms in each state before and after sorting once.

    Parameters:
        T (int): Total time steps.
        initial_states (list): Initial state distribution.
        M (int): Total active resources.
        P0, P1: Transition probability matrices.
        R0, R1: Reward vectors.
        order (list): Sorting order based on Whittle index or custom criterion.

    Returns:
        rewards (list): Rewards at each time step.
    """
    rewards = []
    current_states = np.array(initial_states)

    # Initialize lists to store states for plotting
    states_before_sorting = []
    states_after_sorting = []

    for t in range(T):
        # Print current step
        print(f"\nTime Step {t + 1}:")

        # Before sorting
        print(f"States before sorting: {current_states}")
        states_before_sorting.append(current_states.copy())

        # Sort and transition states
        P0_sorted, P1_sorted, R0_sorted, R1_sorted, sorted_states = sort_data(P0, P1, R0, R1, current_states, order)
        print(f"States after sorting: {sorted_states}")
        states_after_sorting.append(sorted_states.copy())

        # Calculate rewards
        reward = reward_one_step(sorted_states, M, R0_sorted, R1_sorted)
        print(f"Reward: {reward}")
        rewards.append(reward)

        # Reverse sorting to realign states
        reverse_order = np.argsort(order)
        real_next_states = np.zeros(len(order), dtype=int)
        for i in range(len(order)):
            real_next_states[i] = simulate_one_step(sorted_states, M, P0_sorted, P1_sorted)[reverse_order[i]]

        # Update current states
        current_states = real_next_states

    # Convert states to arrays for plotting
    states_before_sorting = np.array(states_before_sorting)
    states_after_sorting = np.array(states_after_sorting)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, T + 1), rewards, label="Reward at each step", color="blue")
    plt.xlabel("Time Steps")
    plt.ylabel("Reward")
    plt.title("Reward at Each Time Step")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot number of arms in each state before and after sorting
    num_states = len(initial_states)
    plt.figure(figsize=(10, 5))
    for state_idx in range(num_states):
        plt.plot(range(1, T + 1), states_before_sorting[:, state_idx],
                 label=f"State {state_idx} (Before Sorting)", linestyle="--", alpha=0.7)
        plt.plot(range(1, T + 1), states_after_sorting[:, state_idx],
                 label=f"State {state_idx} (After Sorting)", alpha=0.7)
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Arms")
    plt.title("Number of Arms in Each State Before and After Sorting")
    plt.grid()
    plt.legend()
    plt.show()



    return rewards

# Simulate and plot details

rewards = simulate_and_plot_details(T, C, M, CP0, CP1, CR0, CR1, whittle_order)
print(f"Total rewards over {T} steps: {np.sum(rewards):.4f}")
print(f"Average reward per step: {np.mean(rewards):.4f}")


    # # Plot the average number of arms in each state over time
    # plt.figure(figsize=(12, 6))
    # time_steps = np.arange(T + 1)
    # for state_idx in range(num_states):
    #     avg_state_counts = np.mean(state_counts_over_time[:, :, state_idx], axis=0)
    #     plt.plot(time_steps, avg_state_counts, label=f"State {state_idx + 1}")

    # plt.xlabel("Time Steps")
    # plt.ylabel("Average Number of Arms")
    # plt.title("Average Number of Arms in Each State Over Time")
    # plt.legend()
    # plt.grid()
    # plt.show()

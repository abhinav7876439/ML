import csv
import datetime
import markovianbandit as bandit
import os
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD


# Circulant Input and Output file paths
input_file = "Restart_input_parameters.csv"
output_files = "Restart_output_results.txt"
 
# Log file path
log_file = output_files
plot_output_folder = "Whittle LP Restart Plots"  # New folder for saving plot images

# Input Parameters
#N = 100
num_states = 5  # Representing the number of bandits in each state
T = 300
epochs = 1
Num_arms = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000]
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

#M = int(alpha * N)  # Total active resources
#print(f"Total active resources = {M}")



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

#CD = generate_uniform_states(N, num_states)
#print(f"States = {CD}")

#CE = [0,0,0,100]
#print(f"States = {CE}")



def distribute_arms_exponential(total_arms, num_states, scale=1.0):
    """
    Distribute a total number of arms among states based on an exponential distribution.

    Parameters:
    - total_arms (int): The total number of arms to distribute.
    - num_states (int): The number of states.
    - scale (float): The scale parameter for the exponential distribution (default is 1.0).

    Reurns:
    - arms (list): A list where each element represents the number of arms in a state.
    """
    # Generate raw exponential values for each state
    raw_values = np.random.exponential(scale, num_states)
    # Normalize to ensure they sum to 1
    normalized_values = raw_values / np.sum(raw_values)
    # Scale by the total number of arms and round to integers
    arms = np.round(normalized_values * total_arms).astype(int)
    
    # Adjust to ensure the total number of arms is exactly `total_arms`
    while sum(arms) < total_arms:
        arms[np.argmax(normalized_values)] += 1  # Add to the state with the largest normalized value
    while sum(arms) > total_arms:
        arms[np.argmax(arms)] -= 1  # Subtract from the state with the largest number of arms

    return arms

#C = distribute_arms_exponential(100, num_states)
#print(f"States = {C}")


def pre_rounding(states):
    rounded_states = np.floor(states).astype(int)  # Start with the floor of each value
    fractional_parts = states - rounded_states
    deficit = int(round(sum(states))) - sum(rounded_states)  # Adjust to maintain the total

    # Randomly round up based on fractional parts
    if deficit > 0:
        indices = np.argsort(-fractional_parts)[:deficit]
        rounded_states[indices] += 1

    return rounded_states

#B = pre_rounding(C)
#print(f"Rounded States = {B}")


# Initialize model and calculate Whittle indices
#model_circulant = bandit.restless_bandit_from_P0P1_R0R1(CP0, CP1, CR0, CR1)
#whittle_indices = model_circulant.whittle_indices()
#print(f"Whittle Indices: {whittle_indices}")
#whittle_order = np.argsort(-whittle_indices)
#print("Whittle Order:", whittle_order)



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

#G = sort_data(CP0,CP1,CR0,CR1,C,whittle_order)
#print(f"Sorted States according to Whittle Priority Policy = {G}")
#new_CP0, new_CP1, new_CR0, new_CR1, new_states = G
#print(f"Number of arms after sorting (Sorted States) according to Whittle Priority Policy = {new_states}")



def actives2(states, M):
    """
    Returns (i, a) where:
    - states 0 to i-1 are fully active
    - state i is active with number of bandits a

    Raises an error if M exceeds the total number of bandits (sum(states)).
    """
    # Check if the budget exceeds the total number of arms
    if M > sum(states):
        raise ValueError("The budget M exceeds the total number of bandits (sum(states)).")

    i = 0
    while i < len(states) and sum(states[:i]) < M:
        i += 1

    if i == 0:  # If no states are fully active, allocate all resources to state 0
        return 0, M
    elif sum(states[:i]) >= M:  # M is partially allocated to state i
        return i - 1, M - sum(states[:i - 1])
    else:  # If M exceeds total resources (unlikely with the above check)
        return len(states) - 1, M - sum(states[:-1])  ## i: Index of the last state that receives active resources,   a: Number of active resources allocated to state i


## It simulates how the states of bandits evolve after one decision step, given the allocation of active resources and the state-transition probabilities
## It returns the new number of bandits in each state after transitions
def simulate_one_step(states,M,P0,P1):
    i, a = actives2(states, M)
    n = len(states)
    data = []
    for j in range(i):
        data.append(np.random.multinomial(states[j], P1[j]))
        print("************************`Data inside 1 loop : ",data)

    data.append(np.random.multinomial(a, P1[i]))
    print("************************`Data after 2 append : ",data)
    data.append(np.random.multinomial(states[i]-a, P0[i]))
    print("************************`Data after 3 append : ",data)
    for j in range(i+1,n):
        data.append(np.random.multinomial(states[j], P0[j]))
        print("************************`Data after 4 append : ",data)
    
    return sum(data)  # Compute the total transition outcomes using sum(data), returning the resulting "next state".

#E = simulate_one_step(new_states,M,new_CP0,new_CP1)
#print(f"After Transition = {E}")

def reward_one_step(states,M,R0,R1):
    i, a = actives2(states,M)
    if i < len(R0):                               # If total number of active states is less than the total number of states
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))      # average reward received
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    
#F = reward_one_step(new_states,M,new_CR0,new_CR1)
#print(f"One step Reward = {F}")

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

#J = simulate_one_step_order(C,M,CP0,CP1,CR0,CR1,whittle_order)
#real_next, reward = J
# Print real_next and reward
#print("Real Next States:", real_next)
#print("Reward:", reward)


# def Whittle_run_simulation_with_independent_epochs(P0, P1, R0, R1, T, states, M, epochs, num_states, order):
#     """
#     Simulate independent epochs with synchronized initial states and Markovian transitions.

#     Parameters:
#         P0, P1: Transition probability matrices (passive and active).
#         R0, R1: Reward vectors (passive and active).
#         T: Time horizon.
#         states: Initial state distribution.
#         M: Number of active resources.
#         epochs: Number of independent epochs.
#         num_states: Number of states in the system.

#     Returns:
#         epoch_rewards: Per-epoch average rewards.
#         state_counts_over_time: Array tracking state dynamics for all epochs.
#     """
#     # Initialize variables to store results
#     epoch_rewards = np.zeros(epochs)  # Total reward for each epoch
#     state_counts_over_time = np.zeros((epochs, T + 1, num_states), dtype=int)  # State dynamics
#     N = sum(states)
#     rewards = []

#     # Initialize lists to store states for plotting
#     states_before_sorting = []
#     states_after_sorting = []


#     for epoch in range(epochs):
#         # Initialize state distribution for the epoch
#         current_states = np.array(states[:])  # Start with initial states [0, 0, 0, 100]
#         state_counts_over_time[epoch, 0, :] = current_states
#         total_reward = 0.0

#         print(f"\nEpoch {epoch + 1}: Initial States: {current_states}")

#         for t in range(1, T + 1):

#             # Print current step
#             print(f"\nTime Step {t + 1}:")

#             # Before sorting
#             print(f"States before sorting: {current_states}")
#             states_before_sorting.append(current_states.copy())

#             # Sort and prioritize based on the Whittle index using states from the previous step
#             P0_sorted, P1_sorted, R0_sorted, R1_sorted, sorted_states = sort_data(
#                 P0, P1, R0, R1, current_states, whittle_order
#             )
#             print(f"States after sorting: {sorted_states}")
#             states_after_sorting.append(sorted_states.copy())

#             # Calculate reward and transition to the next states
#             reward = reward_one_step(sorted_states, M, R0_sorted, R1_sorted)
#             print(f"Reward: {reward}")
#             rewards.append(reward)

#             # next_states = simulate_one_step(sorted_states, M, P0_sorted, P1_sorted)
#             # Reverse sorting to realign states
#             reverse_order = np.argsort(order)
#             real_next_states = np.zeros(len(order), dtype=int)
#             for i in range(len(order)):
#                 real_next_states[i] = simulate_one_step(sorted_states, M, P0_sorted, P1_sorted)[reverse_order[i]]

#             # Update state counts and rewards
#             current_states = real_next_states
#             state_counts_over_time[epoch, t, :] = current_states
#             total_reward += reward

#             # Print dynamics for this time step
#             print(f"  Time {t}: States = {current_states}, Reward = {reward:.4f}")

#         # Store average reward for the epoch
#         epoch_rewards[epoch] = total_reward / (T*N)
#         print(f"Epoch {epoch + 1}: Average Reward = {epoch_rewards[epoch]:.4f}")


#     # Convert states to arrays for plotting
#     states_before_sorting = np.array(states_before_sorting)
#     states_after_sorting = np.array(states_after_sorting)



#     # Plot rewards
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, T + 1), rewards, label="Reward at each step", color="blue")
#     plt.xlabel("Time Steps")
#     plt.ylabel("Reward")
#     plt.title("Reward at Each Time Step")
#     plt.grid()
#     plt.legend()
#     plt.show()

#     # Plot the average number of arms in each state over time
#     plt.figure(figsize=(12, 6))
#     time_steps = np.arange(T + 1)
#     for state_idx in range(num_states):
#         avg_state_counts = np.mean(state_counts_over_time[:, :, state_idx], axis=0)
#         plt.plot(time_steps, avg_state_counts, label=f"State {state_idx + 1}")

#     plt.xlabel("Time Steps")
#     plt.ylabel("Average Number of Arms")
#     plt.title("Average Number of Arms in Each State Over Time")
#     plt.legend()
#     plt.grid()
#     plt.show()

#     return epoch_rewards, state_counts_over_time, rewards


# # Update initial states and run the simulation
# # initial_states = [0, 0, 0, 100]  # Ensure correct initial states
# epoch_rewards, dynamics, rewards = Whittle_run_simulation_with_independent_epochs(
#     CP0, CP1, CR0, CR1, T, C, M, epochs, num_states, whittle_order
# )

# # Print average rewards across epochs and time horizon
# print(f"Average Rewards Per Epoch: {epoch_rewards}")
# print(f"Overall Average Reward: {np.mean(epoch_rewards):.4f}")
# print(f"Dynamics: {dynamics}")


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


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_simulation_with_time(P0, P1, R0, R1, T, states, M, epochs, num_states, lp_order, whittle_order): 
    """
    Simulate independent epochs with synchronized initial states and Markovian transitions.

    Parameters:
        P0, P1: Transition probability matrices (passive and active).
        R0, R1: Reward vectors (passive and active).
        T: Time horizon.
        states: Initial state distribution.
        M: Number of active resources.
        epochs: Number of independent epochs.
        num_states: Number of states in the system.
        order: Sorting order based on Whittle indices.

    Returns:
        epoch_rewards: Per-epoch average rewards.
        state_counts_over_time: Array tracking state dynamics for all epochs.
    """

    # Initialize variables to store results
    lp_epoch_rewards = np.zeros(epochs)  # Total reward for each epoch
    whittle_epoch_rewards = np.zeros(epochs)  # Total reward for each epoch

    lp_state_counts_over_time = np.zeros((epochs, T + 1, num_states), dtype=int)  # State dynamics
    whittle_state_counts_over_time = np.zeros((epochs, T + 1, num_states), dtype=int)  # State dynamics

    N = sum(states)
    lp_rewards = []
    whittle_rewards = []

    # Initialize lists to store states for plotting
    lp_states_before_sorting = []
    lp_states_after_sorting = []

    whittle_states_before_sorting = []
    whittle_states_after_sorting = []


    for epoch in range(epochs):
        
        # Initialize state distribution for the epoch
        lp_current_states = np.array(states[:])  # Start with initial states [0, 0, 0, 100]
        whittle_current_states = np.array(states[:])  # Start with initial states [0, 0, 0, 100]

        lp_state_counts_over_time[epoch, 0, :] = lp_current_states
        whittle_state_counts_over_time[epoch, 0, :] = whittle_current_states

        lp_total_reward = 0.0
        whittle_total_reward = 0.0

        for t in range(1, T + 1):
            # Before sorting
            lp_states_before_sorting.append(lp_current_states.copy())

            # Sort and prioritize based on the Whittle index using states from the previous step
            print(f"LP Order is : {lp_order}")
            print(f"Whittle Order is : {whittle_order}")

            lp_P0_sorted, lp_P1_sorted, lp_R0_sorted, lp_R1_sorted, lp_sorted_states = sort_data(
                P0, P1, R0, R1, lp_current_states, lp_order
            )
            lp_states_after_sorting.append(lp_sorted_states.copy())

            # Calculate reward and transition to the next states
            lp_reward = reward_one_step(lp_sorted_states, M, lp_R0_sorted, lp_R1_sorted)
            lp_rewards.append(lp_reward / (T * N ))

            # Simulate next states and reverse sorting to realign states
            lp_reverse_order = np.argsort(lp_order)
            lp_real_next_states = np.zeros(len(lp_order), dtype=int)
            for i in range(len(lp_order)):
                lp_real_next_states[i] = simulate_one_step(lp_sorted_states, M, lp_P0_sorted, lp_P1_sorted)[lp_reverse_order[i]]

            # Update state counts and rewards
            lp_current_states = lp_real_next_states
            lp_state_counts_over_time[epoch, t, :] = lp_current_states
            lp_total_reward += lp_reward

        # Store average reward for the epoch
        lp_epoch_rewards[epoch] = lp_total_reward / (T * N)

######################################################################################################################

        # Initialize state distribution for the epoch
        
        for t in range(1, T + 1):
            # Before sorting
            whittle_states_before_sorting.append(whittle_current_states.copy())

            whittle_P0_sorted, whittle_P1_sorted, whittle_R0_sorted, whittle_R1_sorted, whittle_sorted_states = sort_data(
                P0, P1, R0, R1, whittle_current_states, whittle_order
            )
            whittle_states_after_sorting.append(whittle_sorted_states.copy())

            # Calculate reward and transition to the next states
            whittle_reward = reward_one_step(whittle_sorted_states, M, whittle_R0_sorted, whittle_R1_sorted)
            whittle_rewards.append(whittle_reward/ (T * N ))

            # Simulate next states and reverse sorting to realign states
            whittle_reverse_order = np.argsort(whittle_order)
            whittle_real_next_states = np.zeros(len(whittle_order), dtype=int)
            for i in range(len(lp_order)):
                whittle_real_next_states[i] = simulate_one_step(whittle_sorted_states, M, whittle_P0_sorted, whittle_P1_sorted)[whittle_reverse_order[i]]

            # Update state counts and rewards
            whittle_current_states = whittle_real_next_states
            whittle_state_counts_over_time[epoch, t, :] = whittle_current_states
            whittle_total_reward += whittle_reward

        # Store average reward for the epoch
        whittle_epoch_rewards[epoch] = whittle_total_reward / (T * N)

        lp_moving_avg_rewards = moving_average(lp_total_reward, window_size=3)
        whittle_moving_avg_rewards = moving_average(whittle_total_reward, window_size=3)

        # Plot rewards for the current epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(lp_moving_avg_rewards)), lp_moving_avg_rewards, label="LP Rewards", color="blue")
        plt.plot(range(len(whittle_moving_avg_rewards)), whittle_moving_avg_rewards, label="Whittle Rewards", color="red")
        plt.xlabel("Time Steps")
        plt.ylabel("Reward")
        plt.title(f"Comparison of LP and Whittle Rewards with Number of arms {N}")
        plt.grid()
        plt.legend()
        
        # Ensure the plots directory exists
        os.makedirs(plot_output_folder, exist_ok=True)
    
        # Save the figure to the specified plot folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(plot_output_folder, f"lp_whittle_reward_Restart_plot_{timestamp}_N{N}_M{M}.png")
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    
        # Show the plot
        plt.show()

    return lp_moving_avg_rewards, whittle_moving_avg_rewards

# Update initial states and run the simulation
# initial_states = [0, 0, 0, 100]  # Ensure correct initial states  P0, P1, R0, R1, T, states, M, epochs, num_states, order

lp_order, lp_indices = infinite_lp_index(CP0, CP1, CR0, CR1, alpha)
print(f"lp indices = {lp_indices}")
print(f"lp order = {lp_order}")

model = bandit.restless_bandit_from_P0P1_R0R1(CP0, CP1, CR0, CR1)
whittle_indices = model.whittle_indices()
print(f"Whittle Indices: {whittle_indices}")
whittle_order = np.argsort(-whittle_indices)
print("Whittle Order:", whittle_order)

###########3run_simulation_with_time(P0, P1, R0, R1, T, states, M, epochs, num_states, lp_order, whittle_order): 

for n in Num_arms:
    initial_states = distribute_arms_exponential(n, num_states, scale=1.0)
    initial_states = pre_rounding(initial_states)
    M = alpha*n
    print(f"N  = {n}, M = {M}")
    lp_rewards, whittle_rewards = run_simulation_with_time(CP0, CP1, CR0, CR1, T, initial_states, M, epochs, num_states, lp_order, whittle_order)


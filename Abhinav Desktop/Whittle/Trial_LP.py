import csv
import datetime
import markovianbandit as bandit
import os
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD

# Circulant Input and Output file paths
input_file = "Circulant_input_parameters.csv"
output_files = "Circulant_output_results.txt"

# Log file path
log_file = output_files

# Input Parameters
N = 100
num_states = 4  # Representing the number of bandits in each state
T = 350
epochs = 10
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



def actives2(states, M):   ##  How active resources (M) are allocated among state ## states = [5, 3, 7, 4]
    """
    Returns (i, a) where:
    - states 0 to i-1 are fully active
    - state i is active with number of bandits a
    """
    i = 0
    while i < len(states) and sum(states[:i]) < M:
        i += 1
    if i == 0:  # If no states are fully active, allocate all resources to state 0
        return 0, M
    elif sum(states[:i]) >= M:  # M is partially allocated to state i
        return i - 1, M - sum(states[:i - 1])
    else:  # If M exceeds total resources, allocate the remainder to the last state
        return len(states) - 1, M - sum(states[:-1])    ## i: Index of the last state that receives active resources,   a: Number of active resources allocated to state i
    
D = actives2(new_states,M)
print(f"Active States = {D}")




def simulate_one_step(states,M,P0,P1):    ## It simulates how the states of bandits evolve after one decision step, given the allocation of active resources and the state-transition probabilities
    i, a = actives2(states, M)              ## It returns the new number of bandits in each state after transitions
    n = len(states)
    data = []
    for j in range(i):
        data.append(np.random.multinomial(states[j], P1[j]))
    data.append(np.random.multinomial(a, P1[i]))
    data.append(np.random.multinomial(states[i]-a, P0[i]))
    for j in range(i+1,n):
        data.append(np.random.multinomial(states[j], P0[j]))
    return sum(data)    # Compute the total transition outcomes using sum(data), returning the resulting "next state".

E = simulate_one_step(new_states,M,new_CP0,new_CP1)
print(f"After Transition = {E}")



def reward_one_step(states,M,R0,R1):
    i, a = actives2(states,M)
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




# Function to simulate infinite horizon process for one epoch
def Whittle_infinite_sim(P0, P1, R0, R1, T, states, M):
    """Simulate the infinite horizon process for one epoch."""
    N = sum(states)
    reward = 0.0
    for _ in range(T):
        P0,P1,R0,R1,Sorted_states = sort_data(P0,P1,R0,R1,states,whittle_order)
        r = reward_one_step(Sorted_states, M, R0, R1)
        After_transition_states = simulate_one_step(Sorted_states, M, P0, P1)  
        reward += r
    return reward / (T), After_transition_states

H = Whittle_infinite_sim(CP0,CP1,CR0,CR1,T,C,M)
print(f"Whittle Reward Generation = {H}")



# # Function to run 100 epochs and calculate the average reward with plot
# def run_simulation_for_epochs(P0, P1, R0, R1, T, states, M, epochs):
#     """Run the simulation for the specified number of epochs and log the results, also plot average reward."""
#     total_reward = 0.0
#     epoch_rewards = []  # List to store rewards for each epoch
    
#     for epoch in range(epochs):
#         # Reset states for each epoch
#         current_states = states[:]
#         print(f"Current States = {current_states}")
#         epoch_reward, After_transition_states_1 = Whittle_infinite_sim(P0, P1, R0, R1, T, current_states, M)
#         total_reward += epoch_reward
#         epoch_rewards.append(epoch_reward)  # Store the reward for this epoch
#         print(f"Epoch {epoch + 1}: Reward = {epoch_reward:.4f}")
    
#     average_reward = total_reward / epochs
#     return average_reward

# I = run_simulation_for_epochs(CP0, CP1, CR0, CR1, T, C, M, epochs)
# print(f"Average reward = {I}")


def run_simulation_with_independent_epochs(P0, P1, R0, R1, T, states, M, epochs, num_states):
    """
    Run the simulation for multiple epochs, independently evolving over the time horizon.
    Plot the number of arms in each state over time and compute average rewards.
    
    Parameters:
        P0, P1: Transition probability matrices.
        R0, R1: Reward vectors.
        T: Time horizon.
        states: Initial state distribution.
        M: Total active resources.
        epochs: Number of independent epochs.
        num_states: Number of states in the system.
    
    Returns:
        avg_epoch_rewards: Average reward across epochs.
        state_counts_over_time: List of arrays tracking the number of arms in each state across epochs and time steps.
    """
    epoch_rewards = []  # Store total rewards for each epoch
    state_counts_over_time = np.zeros((epochs, T + 1, num_states), dtype=int)  # Track arms in each state
    
    for epoch in range(epochs):
        current_states = np.array(states[:])  # Initialize states for the epoch
        total_reward = 0.0
        
        # Store initial state counts
        state_counts_over_time[epoch, 0, :] = current_states
        
        for t in range(T):
            # Simulate one step
            reward = reward_one_step(current_states, M, R0, R1)
            next_states = simulate_one_step(current_states, M, P0, P1)
            
            # Update states and rewards
            current_states = next_states
            total_reward += reward
            
            # Track state counts at this time step
            state_counts_over_time[epoch, t + 1, :] = current_states
        
        epoch_rewards.append(total_reward / T)  # Average reward per epoch
    
    # Compute average reward across epochs
    avg_epoch_rewards = np.mean(epoch_rewards)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(T + 1)
    
    for state_idx in range(num_states):
        avg_state_counts = np.mean(state_counts_over_time[:, :, state_idx], axis=0)
        plt.plot(time_steps, avg_state_counts, label=f"State {state_idx + 1}")
    
    plt.xlabel("Time Steps")
    plt.ylabel("Average Number of Arms")
    plt.title("Number of Arms in Each State Over Time")
    plt.legend()
    plt.grid()
    plt.show()
    
    return avg_epoch_rewards, state_counts_over_time

# Run the simulation
avg_reward, state_counts = run_simulation_with_independent_epochs(
    CP0, CP1, CR0, CR1, T, C, M, epochs, num_states
)
print(f"Average Reward Across Epochs: {avg_reward}")
# I = run_simulation_for_epochs(CP0, CP1, CR0, CR1, T, C, M, epochs)
print(f"State Counts = {state_counts}")


def run_simulation_epochs_independent(P0, P1, R0, R1, T, states, M, epochs):
    """
    Run the simulation for multiple epochs and calculate rewards with plots of state dynamics.
    
    Parameters:
        P0, P1 (ndarray): Transition probabilities for passive and active states.
        R0, R1 (ndarray): Rewards for passive and active states.
        T (int): Time horizon for the simulation.
        states (list): Initial state distribution of arms.
        M (int): Number of active resources.
        epochs (int): Number of independent epochs.
        
    Returns:
        list: Rewards for each epoch over the time horizon.
    """
    state_dynamics = np.zeros((T, epochs, len(states)))  # Shape: (time, epochs, states)
    epoch_rewards = np.zeros((epochs, T))  # Rewards for each epoch at each time step

    # Initialize states for all epochs
    current_states = [states[:] for _ in range(epochs)]

    for t in range(T):
        for e in range(epochs):
            # Apply Whittle priority ordering
            P0_sorted, P1_sorted, R0_sorted, R1_sorted, sorted_states = sort_data(
                P0, P1, R0, R1, current_states[e], whittle_order
            )

            # Calculate rewards and simulate state transitions
            epoch_rewards[e, t] = reward_one_step(sorted_states, M, R0_sorted, R1_sorted)
            current_states[e] = simulate_one_step(sorted_states, M, P0_sorted, P1_sorted)

        # Sync states across epochs for uniformity
        average_states = np.mean(current_states, axis=0).astype(int)
        current_states = [average_states.tolist() for _ in range(epochs)]
        
        # Log the states at this time step
        state_dynamics[t] = current_states

    # Plot the number of arms in each state over the time horizon
    for state_idx in range(len(states)):
        plt.figure(figsize=(10, 6))
        plt.title(f"Number of Arms in State {state_idx + 1} Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Number of Arms")
        for e in range(epochs):
            plt.plot(
                range(T), state_dynamics[:, e, state_idx], label=f"Epoch {e + 1}", alpha=0.7
            )
        plt.legend()
        plt.grid(True)
        plt.show()

    return epoch_rewards

# Run the simulation and plot results
rewards = run_simulation_epochs_independent(CP0, CP1, CR0, CR1, T, C, M, epochs)

# Print average rewards across epochs and time horizon
average_rewards_over_epochs = np.mean(rewards, axis=1)
print(f"Average Rewards Per Epoch: {average_rewards_over_epochs}")
print(f"Overall Average Reward: {np.mean(average_rewards_over_epochs)}")




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

LP_order, LP_indices = infinite_lp_index(CP0, CP1, CR0, CR1, alpha)
print(f"LP Priority Order = {LP_order}")
print(f"LP Indices = {LP_indices}")
log_output(f"LP Priority Order = {LP_order}", log_file)
log_output(f"LP Indices: {LP_indices}", log_file)



def infinite_fix_point(P0, P1, R0, R1, alpha):
    """Calculate m* values using LP solution."""
    d = len(R0)
    P = [P0, P1]
    R = [R0, R1]
    V_1, V_2 = [], []

    action = range(2)
    state = range(d)
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y", (action, state), lowBound=0, upBound=1.0)

    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss] * P[a][ss][s] for a in action for ss in state])
    prob += lpSum([variables[a][s] for a in action for s in state]) == 1.0
    prob += lpSum([variables[a][s] * R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))

    m_star = np.zeros(d)
    for i in range(d):
        V1 = variables[0][i]
        V2 = variables[1][i]
        v1 = V1.varValue
        v2 = V2.varValue
        V_1.append(v1)
        V_2.append(v2)
        m_star[i] = v1 + v2

    return m_star, V_1, V_2

# Calculate m* values
m_star, V_0, V_1 = infinite_fix_point(CP0, CP1, CR0, CR1, alpha)
print("Passive Optimal solution:", V_0)
print("Active Optimal solution:", V_1)
print("m* Values:", m_star)
log_output(f"m* Values: {m_star}", log_file)


# Function to simulate infinite horizon process for one epoch
def LP_infinite_sim(P0, P1, R0, R1, T, states, M):
    """Simulate the infinite horizon process for one epoch."""
    N = sum(states)
    reward = 0.0
    for _ in range(T):
        P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,LP_order)
        r = reward_one_step(states, M, R0, R1)
        states = simulate_one_step(states, M, P0, P1)
        reward += r
    return reward / (T * N)

X = LP_infinite_sim(CP0,CP1,CR0,CR1,T,C,M)
print(f"LP Reward Generation = {X}")


current_states = [0, 0, 0, 100]

# Function to run 100 epochs and calculate the average reward with plot
def run_simulation_for_time(N, num_states, P0, P1, R0, R1, T, M, whittle_order, epochs, log_file, prob):
    """Run the simulation for the specified number of epochs and log the results."""
    
    whittle_rewards = []
    LP_rewards = []
    for t in range(T):
        whittle_total_reward = 0
        LP_total_reward = 0
        for epoch in range(epochs):
            try:
                whittle_epoch_reward, After_transition_3 = Whittle_infinite_sim(P0, P1, R0, R1, T, current_states, M)
                LP_epoch_reward = LP_infinite_sim(P0, P1, R0, R1, T, current_states, M)
            except ValueError as e:
                print(f"Error encountered during simulation at time {t}, epoch {epoch}: {e}")
                whittle_epoch_reward = 0  # Assign default value if error occurs
                LP_epoch_reward = 0
            whittle_total_reward += whittle_epoch_reward
            LP_total_reward += LP_epoch_reward
        
        whittle_avg_reward = whittle_total_reward / epochs if epochs > 0 else 0
        whittle_rewards.append(whittle_avg_reward)
        LP_avg_reward = LP_total_reward / epochs if epochs > 0 else 0
        LP_rewards.append(LP_avg_reward)

        
        log_output(f"Time step {t}: Whittle Reward = {whittle_avg_reward:.4f}", log_file)
        log_output(f"Time step {t}: LP Reward = {LP_avg_reward:.4f}", log_file)


    # Plotting the expected reward graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(whittle_rewards)), whittle_rewards, label="Whittle Reward per time step", color='r')
    plt.plot(range(len(LP_rewards)), LP_rewards, label="LP Reward per time step", color='g')
    plt.xlabel('Time steps')
    plt.ylabel('Reward')
    plt.title(f'Expected Reward per Time step for {prob} Problem with activation fraction {M/N}')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

    return whittle_rewards, LP_rewards

K = run_simulation_for_time(N, num_states, CP0, CP1, CR0, CR1, T, M, whittle_order, epochs, log_file, "Circulant")
print(f"Avg reward = {K}")








def run_simulation_for_arms(N_values, num_states, P0, P1, R0, R1, T, M, whittle_order, epochs, log_file, prob):
    whittle_rewards = []

    if isinstance(N_values, int):
        N_values = [N_values]

    for n in N_values:
        if n <= 0:
            print(f"Skipping invalid arm count: {n}")
            continue

        rounded_whittle_states = generate_uniform_states(n, num_states)
        whittle_P0, whittle_P1, whittle_R0, whittle_R1, rounded_whittle_states = sort_data(
            P0, P1, R0, R1, rounded_whittle_states, whittle_order
        )

        total_reward_whittle = 0
        for epoch in range(epochs):
            try:
                if M > n:
                    M = n  # Cap M to the number of arms
                reward = Whittle_infinite_sim(whittle_P0, whittle_P1, whittle_R0, whittle_R1, T, rounded_whittle_states, M)
                total_reward_whittle += reward
            except ValueError as e:
                print(f"Error encountered during simulation for arms {n}, epoch {epoch}: {e}")
                total_reward_whittle += 0

        avg_reward_whittle = total_reward_whittle / epochs if epochs > 0 else 0
        whittle_rewards.append(avg_reward_whittle)
        log_output(f"Number of arms {n}, States = {num_states}: Whittle Reward = {avg_reward_whittle:.4f}", log_file)

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, whittle_rewards, label="Whittle rewards", color='g')
    plt.xlabel('Number of Arms')
    plt.ylabel('Reward')
    plt.title(f'Expected Reward per Time Step for {prob} Problem with activation fraction {alpha}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return whittle_rewards

# (N_values, num_states, P0, P1, R0, R1, T, M, whittle_order, epochs, log_file, prob)
M = run_simulation_for_arms(Num_arms, num_states, CP0, CP1, CR0, CR1, T, M, whittle_order, epochs, log_file, "circulant")
print(f"Average reward (Modified arms) = {M}")






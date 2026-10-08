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
T = 300
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
    
    """ Distributes the total number of bandits (N) uniformly across the given number of states (num_states).
    
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

BC = distribute_arms_exponential(100, 4)
print(f"States = {BC}")

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

    #active_state = np.zeros((1, num_states))
    i, a = actives2(states,M)
    if i < len(R0):                           # If total number of active states is less than the total number of states
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))      # average reward received
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    
F = reward_one_step(new_states,M,new_CR0,new_CR1)
print(f"One step Reward = {F}")

def reward_one_step_order(states,M,R0,R1):

    i, a = actives2(states,M)
    if i < len(R0):                           # If total number of active states is less than the total number of states
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))     # average reward received
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    

def simulate_one_step_order(states,M,P0,P1,R0,R1,order):
    n = len(order)
    P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,order)
    next_states = simulate_one_step(states,M,P0,P1)
    reward = reward_one_step_order(states,M,R0,R1)
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


# Function to simulate infinite horizon process for one epoch       Whittle_infinite_sim(P0, P1, R0, R1, t, current_states, M, epoch)
def Whittle_infinite_sim(P0, P1, R0, R1, T, states, M):                      
    """Simulate the infinite horizon process for one epoch."""
    N = sum(states)
    reward = 0.0
    whittle_active = 0
    state_0 = []
    state_4 = []
    s0_index = np.where(whittle_order == 0)[0][0]
    s4_index = np.where(whittle_order == 3)[0][0]
    for _ in range(T):
        P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,whittle_order)
        r = reward_one_step(states, M, R0, R1)
        states = simulate_one_step(states, M, P0, P1)
        print(f"Whittle current states = {states}")
        state_0.append(states[s0_index])
        state_4.append(states[s4_index])
        reward += r
        

    
    return reward / (T * N), state_0, state_4


H = Whittle_infinite_sim(CP0,CP1,CR0,CR1,T,C,M)
print(f"Whittle Reward Generation = {H}")


# Function to run 100 epochs and calculate the average reward with plot
def run_simulation_for_epochs(P0, P1, R0, R1, T, states, M, epochs):
    """Run the simulation for the specified number of epochs and log the results, also plot average reward."""
    total_reward = 0.0
    epoch_rewards = []  # List to store rewards for each epoch
    
    for epoch in range(epochs):
        # Reset states for each epoch
        current_states = states[:]
        epoch_reward = Whittle_infinite_sim(P0, P1, R0, R1, T, current_states, M)
        total_reward += epoch_reward
        epoch_rewards.append(epoch_reward)  # Store the reward for this epoch
        print(f"Epoch {epoch + 1}: Reward = {epoch_reward:.4f}")
    
    average_reward = total_reward / epochs
    return average_reward


I = run_simulation_for_epochs(CP0, CP1, CR0, CR1, T, C, M, epochs)
print(f"Average reward = {I}")


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
    lp_active = 0
    
    s_0 = []
    s_4 = []
    
    s0_index = LP_order.index(0)
    s4_index = LP_order.index(3)

    for _ in range(T):
        P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,LP_order)
        r = reward_one_step(states, M, R0, R1)
        states = simulate_one_step(states, M, P0, P1)
        print(f"lp current states = {states}")
        s_0.append(states[s0_index])
        s_4.append(states[s4_index])
        reward += r
       
    return reward / (T * N), s_0, s_4

X = LP_infinite_sim(CP0,CP1,CR0,CR1,T,C,M)
print(f"LP Reward Generation = {X}")

# Function to run 100 epochs and calculate the average reward with plot
def run_simulation_for_time(N, num_states, P0, P1, R0, R1, T, M, epochs, log_file, prob):
    """Run the simulation for the specified number of epochs and log the results."""
    
    whittle_rewards = []
    LP_rewards = []
    current_states = generate_uniform_states(N, num_states)
    lp_active_per_state = np.zeros((T, num_states))
    whittle_active_per_state = np.zeros((T, num_states))
    
    whittle_state_4 = []
    whittle_state_0 = []
    lp_state_4 = []
    lp_state_0 = []
    

    #for t in range(1,T+1):
    whittle_total_reward = 0
    LP_total_reward = 0
        
        #whittle_state_4.append(current_states[-1])
        #whittle_state_0.append(current_states[0])

        #lp_state_4.append(current_states[-1])
        #lp_state_0.append(current_states[0])

    print(f"Current states = {current_states}")

        #for epoch in range(epochs):
            #print(f"Current states = {current_states} for epoch = {epoch}")
            #print(f"time = {t}, epoch = {epoch}")
    try:

        whittle_epoch_reward, whittle_s0, whittle_s4 = Whittle_infinite_sim(P0, P1, R0, R1, T, current_states, M)
        LP_epoch_reward, lp_s0, lp_s4 = LP_infinite_sim(P0, P1, R0, R1, T, current_states, M)
    except ValueError as e:
        print(f"Error encountered during simulation")
        whittle_epoch_reward = 0  # Assign default value if error occurs
        LP_epoch_reward = 0
    whittle_total_reward = whittle_epoch_reward
    LP_total_reward = LP_epoch_reward
        
    whittle_state_0.extend(whittle_s0)
    whittle_state_4.extend(whittle_s4)

    lp_state_0.extend(lp_s0)
    lp_state_4.extend(lp_s4)

    whittle_avg_reward = whittle_total_reward 
    whittle_rewards.append(whittle_avg_reward)
    LP_avg_reward = LP_total_reward 
    LP_rewards.append(LP_avg_reward)

        
    log_output(f"Time step {T}: Whittle Reward = {whittle_avg_reward:.4f}", log_file)
    log_output(f"Time step {T}: LP Reward = {LP_avg_reward:.4f}", log_file)

    whittle_running_sum = np.cumsum(whittle_rewards)  # Cumulative sum of rewards
    time_steps = np.arange(1, len(whittle_rewards) + 1)  # Time step indices (1-based)
    whittle_running_avg_rewards = whittle_running_sum / time_steps  # Calculate running average

    LP_running_sum = np.cumsum(LP_rewards)  # Cumulative sum of rewards
    LP_running_avg_rewards = LP_running_sum  # Calculate running average

    # Print the sum of arms in state 4 and 0
    print(f"The sum of arms in state 4 in lp = {sum(lp_state_4)}")
    print(f"The sum of arms in state 0 in lp = {sum(lp_state_0)}")

    print(f"The sum of arms in state 4 in whittle = {sum(whittle_state_4)}")
    print(f"The sum of arms in state 0 in whittle = {sum(whittle_state_0)}")

# Plot the number of arms in state 0
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(lp_state_0)), lp_state_0, label="state 0", color='r')
    plt.xlabel('Time steps')
    plt.ylabel('Arms in state 0')
    plt.title(f'Arms in state 0 in lp')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

    # Plot the number of arms in state 0
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(lp_state_0)), lp_state_4, label="state 0", color='r')
    plt.xlabel('Time steps')
    plt.ylabel('Arms in state 4')
    plt.title(f'Arms in state 4 in lp')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(whittle_state_0)), whittle_state_0, label="state 0", color='r')
    plt.xlabel('Time steps')
    plt.ylabel('Arms in state 0')
    plt.title(f'Arms in state 0 in whittle')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(whittle_state_0)), whittle_state_4, label="state 0", color='r')
    plt.xlabel('Time steps')
    plt.ylabel('Arms in state 4')
    plt.title(f'Arms in state 4 in whittle')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

    
    # Plotting the expected reward graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(whittle_running_avg_rewards)), whittle_running_avg_rewards, label="Whittle Reward", color='r')
    plt.plot(range(len(LP_running_avg_rewards)), LP_running_avg_rewards, label="LP Reward", color='b')
    plt.xlabel('Time steps')
    plt.ylabel('Time running Reward')
    plt.title(f'Expected Time running reward per Time step for {prob} Problem with activation fraction {M/N}')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

    # Plot the number of active arms in each state for lp index
    time = np.arange(T)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bar chart for first epoch
    for state in range(num_states):
        ax.bar(time, lp_active_per_state[:, state], label=f"State {state+1}", bottom=np.sum(lp_active_per_state[:, :state], axis=1))

    ax.set_title("Arms Activated in Each State Over Time (First Epoch) for LP Index Policy")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Arms Activated")
    ax.legend()
    plt.show()

    # Plot the number of active arms in each state for whittle index

    #time = np.arange(T)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bar chart for first epoch
    for state in range(num_states):
        ax.bar(time, whittle_active_per_state[:, state], label=f"State {state+1}", bottom=np.sum(whittle_active_per_state[:, :state], axis=1))

    ax.set_title("Arms Activated in Each State Over Time (First Epoch) for Whittle Index Policy")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Arms Activated")
    ax.legend()
    plt.show()

    return whittle_rewards, LP_rewards

K = run_simulation_for_time(100, num_states, CP0, CP1, CR0, CR1, T, M, 10, log_file, "Circulant")
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
#M = run_simulation_for_arms(Num_arms, num_states, CP0, CP1, CR0, CR1, T, M, whittle_order, epochs, log_file, "circulant")3
# print(f"Average reward (Modified arms) = {M}")






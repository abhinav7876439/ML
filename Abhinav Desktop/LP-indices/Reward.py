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
plot_output_folder = "Reward plot"

# Input Parameters
N = 100
num_states = 4  # Representing the number of bandits in each state
T = 10
epochs = 1
Num_arms = [100, 1000, 2000, 3000, 4000, 5000,10000, 15000]



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

P0, P1, R0, R1, alpha  = read_parameters_from_csv(input_file)
print(f"CP0 = {P0}")
print(f"CP1 = {P1}")
print(f"CR0 = {R0}")
print(f"CR1 = {R1}")
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


def pre_rounding(states):
    rounded_states = np.floor(states).astype(int)  # Start with the floor of each value
    fractional_parts = states - rounded_states
    deficit = int(round(sum(states))) - sum(rounded_states)  # Adjust to maintain the total

    # Randomly round up based on fractional parts
    if deficit > 0:
        indices = np.argsort(-fractional_parts)[:deficit]
        rounded_states[indices] += 1

    return rounded_states


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




## It returns the new number of bandits in each state after transitions
def simulate_one_step(states,M,P0,P1):
    i, a = actives2(states, M)
    n = len(states)

    print("***********value of i : ",i)
    print("***********value of a : ",a)
    print("***********value of M : ",M)



    data = []
    for j in range(i):
        data.append(np.random.multinomial(states[j], P1[j]))
    data.append(np.random.multinomial(a, P1[i]))
    data.append(np.random.multinomial(states[i]-a, P0[i]))
    for j in range(i+1,n):
        data.append(np.random.multinomial(states[j], P0[j]))
    return sum(data)



def reward_one_step(states,M,R0,R1):
    i, a = actives2(states,M)
    if i < len(R0):                               # If total number of active states is less than the total number of states
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i]+np.dot(states[i+1:],R0[i+1:]))      # average reward received
    else:
        return (np.dot(states[0:i],R1[0:i])+a*R1[i]+(states[i]-a)*R0[i])
    

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

def sort_para(P0,P1,R0,R1,order):
    n = len(order)
    new_P0 = np.zeros((n,n))
    new_P1 = np.zeros((n,n))
    new_R0 = np.zeros(n)
    new_R1 = np.zeros(n)
    for i in range(n):
        for j in range(n):
            new_P0[i,j] = P0[order[i], order[j]]
            new_P1[i,j] = P1[order[i], order[j]]
        new_R0[i] = R0[order[i]]
        new_R1[i] = R1[order[i]]
    return new_P0, new_P1, new_R0, new_R1

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
        lp_order, whittle_order: Sorting orders based on LP and Whittle indices.

    Returns:
        lp_moving_avg_rewards, whittle_moving_avg_rewards: Moving average rewards for LP and Whittle strategies.
    """
    # Initialize variables
    lp_rewards_over_time = np.zeros((epochs, T))  # Per-time-step LP rewards
    whittle_rewards_over_time = np.zeros((epochs, T))  # Per-time-step Whittle rewards

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Initialize state distributions for the epoch
        lp_current_states = np.array(states[:])  # LP initial states
        whittle_current_states = np.array(states[:])  # Whittle initial states

        for t in range(T):
            print(f"\nTime Step {t + 1}/{T}")

            # LP: Sort, calculate reward, and simulate transitions
            print(f"\n[LP] States before sorting: {lp_current_states}")
            lp_P0_sorted, lp_P1_sorted, lp_R0_sorted, lp_R1_sorted, lp_sorted_states = sort_data(
                P0, P1, R0, R1, lp_current_states, lp_order)
            print(f"[LP] States after sorting: {lp_sorted_states}")
            lp_reward = reward_one_step(lp_sorted_states, M, lp_R0_sorted, lp_R1_sorted)
            lp_real_next_states = np.zeros(len(lp_order), dtype=int)
            #lp_reverse_order = np.argsort(lp_order)
            for i in range(len(lp_order)):
                lp_real_next_states[i] = simulate_one_step(lp_sorted_states, M, lp_P0_sorted, lp_P1_sorted)[lp_order[i]]
            lp_current_states = lp_real_next_states
            lp_rewards_over_time[epoch, t] = lp_reward
            print(f"[LP] States after transition: {lp_current_states}")
            print(f"[LP] Reward at this step: {lp_reward}")

            # Whittle: Sort, calculate reward, and simulate transitions
            print(f"\n[Whittle] States before sorting: {whittle_current_states}")
            whittle_P0_sorted, whittle_P1_sorted, whittle_R0_sorted, whittle_R1_sorted, whittle_sorted_states = sort_data(
                P0, P1, R0, R1, whittle_current_states, whittle_order)

            print(f"[Whittle] States after sorting: {whittle_sorted_states}")
            whittle_reward = reward_one_step(whittle_sorted_states, M, whittle_R0_sorted, whittle_R1_sorted)
            whittle_real_next_states = np.zeros(len(whittle_order), dtype=int)
            whittle_reverse_order = np.argsort(whittle_order)
            for i in range(len(whittle_order)):
                whittle_real_next_states[i] = simulate_one_step(whittle_sorted_states, M, whittle_P0_sorted, whittle_P1_sorted)[whittle_reverse_order[i]]
            whittle_current_states = whittle_real_next_states
            whittle_rewards_over_time[epoch, t] = whittle_reward
            print(f"[Whittle] States after transition: {whittle_current_states}")
            print(f"[Whittle] Reward at this step: {whittle_reward}")

    # Compute moving averages across epochs
    lp_moving_avg_rewards = moving_average(lp_rewards_over_time.mean(axis=0), window_size=3)
    whittle_moving_avg_rewards = moving_average(whittle_rewards_over_time.mean(axis=0), window_size=3)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(lp_moving_avg_rewards)), lp_moving_avg_rewards, label="LP Moving Avg Rewards", color="blue")
    plt.plot(range(len(whittle_moving_avg_rewards)), whittle_moving_avg_rewards, label="Whittle Moving Avg Rewards", color="red")
    plt.xlabel("Time Steps")
    plt.ylabel("Moving Average Reward")
    plt.title(f"Comparison of LP and Whittle Moving Average Rewards for Circulant (M={M}, N={sum(states)})")
    plt.grid()
    plt.legend()

    # Ensure the plots directory exists
    os.makedirs(plot_output_folder, exist_ok=True)

    # Save the figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(plot_output_folder, f"lp_whittle_moving_avg_rewards_ciculant_{timestamp}_N{sum(states)}_M{M}.png")
    plt.savefig(plot_filename)
    print(f"\nPlot saved as {plot_filename}")

    # Show the plot
    plt.show()

    return lp_moving_avg_rewards, whittle_moving_avg_rewards


lp_order, lp_indices = infinite_lp_index(P0, P1, R0, R1, alpha)
print(f"lp indices = {lp_indices}")
print(f"lp order = {lp_order}")

model_circulant = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
whittle_indices = model_circulant.whittle_indices()
print(f"Whittle Indices: {whittle_indices}")
whittle_order = np.argsort(-whittle_indices)
print("Whittle Order:", whittle_order)


# Example usage:
# Assuming all required inputs (P0, P1, R0, R1, etc.) are defined elsewhere.
# for n in N:
#     initial_states = distribute_arms_exponential(n, num_states, scale=1.0)
#     initial_states = pre_rounding(initial_states)
#     M = int(alpha * n)
#     print(f"\nN = {n}, M = {M}")
#     lp_rewards, whittle_rewards = run_simulation_with_time(P0, P1, R0, R1, T, initial_states, M, epochs, num_states, lp_order, whittle_order)



initial_states = distribute_arms_exponential(N, num_states, scale=1.0)
initial_states = pre_rounding(initial_states)
M = int(alpha * N)
print(f"\nN = {N}, M = {M}")
lp_rewards, whittle_rewards = run_simulation_with_time(P0, P1, R0, R1, T, initial_states, M, epochs, num_states, lp_order, whittle_order)

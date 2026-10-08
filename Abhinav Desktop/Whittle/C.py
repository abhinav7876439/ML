import os 
import numpy as np
import csv
import datetime
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt

import markovianbandit as bandit

# Input Parameters
N = 100
num_states = 4  # Representing the number of bandits in each state
alpha = 0.2
M = int(alpha * N)  # Total active resources
T = 1000
order = []  # Activation states, will be determined dynamically

circulant_input_file = "Circulant_input_parameters_1.csv"
circulant_output_file = "Circulant_output_results.txt"

# Circulant Input and Output file paths
restart_input_file  = "Restart_input_parameters_1.csv"
restart_output_file = "Restart_output_results.txt"

output_folder = "results"
plot_output_folder = "plots"  # New folder for saving plot images

# Helper Functions
def read_parameters_from_csv(file_path):
    """Reads parameters from a CSV file."""
    P0, P1 = [], []
    R0, R1 = None, None
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip the header row

        for row in reader:
            P0.append([float(value) for value in row[0].split(",")])
            P1.append([float(value) for value in row[1].split(",")])
            R0 = [float(value) for value in row[2].split(",")]
            R1 = [float(value) for value in row[3].split(",")]

    return np.array(P0), np.array(P1), np.array(R0), np.array(R1)


def log_output(output, file_path):
    """Log output with timestamp to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as logfile:
        logfile.write(f"[{timestamp}] {output}\n")


def write_output_to_text_file(file_path, whittle_indices, alpha):
    """Write output to a text file."""
    with open(file_path, "a") as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Date and Time: {current_time}\n")
        file.write(f"Alpha: {alpha}\n")
        file.write(f"Whittle Indices: {whittle_indices}\n\n")
    print(f"Output successfully written to text file: {file_path}")


def save_output_to_csv(folder_path, base_filename, data, headers=None):
    """Saves results to a CSV file."""
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"{base_filename}_{timestamp}.csv")

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)

    print(f"Output saved to {file_path}")


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

    order = list(np.argsort(-I))
    print("order is : ", order)
    return list(np.argsort(-I)), I


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


def actives2(states, M):
    """
    Returns (i, a) where:
    - states 0 to i-1 are active
    - state i is active with number of bandits a
    """
    i = 0
    while i < len(states) and sum(states[:i]) < M:
        i += 1
    if i == 0:
        return 0, M
    elif i >= len(states):
        return len(states) - 1, states[-1]
    else:
        return i - 1, M - sum(states[:i - 1])


# def simulate_one_step(states, M, P0, P1):
#     """Simulate one step of the process."""
#     i, a = actives2(states, M)
#     n = len(states)
#     data = []
#     for j in range(i):
#         data.append(np.random.multinomial(states[j], P1[j]))
#     data.append(np.random.multinomial(a, P1[i]))
#     data.append(np.random.multinomial(states[i] - a, P0[i]))
#     for j in range(i + 1, n):
#         data.append(np.random.multinomial(states[j], P0[j]))
#     return sum(data)

# def simulate_one_step(states, M, P0, P1):
#     """
#     Simulate one step of the process with normalized probabilities.
#     """
#     i, a = actives2(states, M)
#     n = len(states)
#     data = []

#     # Normalize P0 and P1 to ensure each row sums to 1
#     P0 = np.array([row / row.sum() for row in P0])
#     P1 = np.array([row / row.sum() for row in P1])

#     for j in range(i):
#         data.append(np.random.multinomial(states[j], P1[j]))
#     data.append(np.random.multinomial(a, P1[i]))
#     data.append(np.random.multinomial(states[i] - a, P0[i]))
#     for j in range(i + 1, n):
#         data.append(np.random.multinomial(states[j], P0[j]))

#     return sum(data)



def simulate_one_step(states, M, P0, P1):
    """
    Simulate one step of the process with normalized probabilities.
    Handles cases where row sums are zero.
    """
    i, a = actives2(states, M)
    n = len(states)
    data = []

    # Normalize P0 and P1 safely, replacing zero rows with uniform distribution
    def normalize_matrix(matrix):
        normalized_matrix = []
        for row in matrix:
            row_sum = row.sum()
            if row_sum == 0:
                # Replace zero rows with a uniform distribution
                normalized_matrix.append(np.ones_like(row) / len(row))
            else:
                normalized_matrix.append(row / row_sum)
        return np.array(normalized_matrix)

    P0 = normalize_matrix(P0)
    P1 = normalize_matrix(P1)

    for j in range(i):
        data.append(np.random.multinomial(states[j], P1[j]))
    data.append(np.random.multinomial(a, P1[i]))
    data.append(np.random.multinomial(states[i] - a, P0[i]))
    for j in range(i + 1, n):
        data.append(np.random.multinomial(states[j], P0[j]))

    return sum(data)






def reward_one_step(states, M, R0, R1):
    """Calculate the reward for one step."""
    i, a = actives2(states, M)
    if i < len(R0):
        return (
            np.sum(np.array(states[:i]) * np.array(R1[:i]))
            + a * R1[i]
            + (states[i] - a) * R0[i]
        )
    return np.sum(np.array(states) * np.array(R0))


def sort_data(P0, P1, R0, R1, states, order):  # order = Activation States for LP
    # Convert input matrices to NumPy arrays if they are not already
    P0 = np.array(P0)
    P1 = np.array(P1)
    R0 = np.array(R0)
    R1 = np.array(R1)
    states = np.array(states)
    
    n = len(order)

    # Reassign the sorted matrices to the original variables
    for i in range(n):
        for j in range(n):
            P0[i, j] = P0[order[i], order[j]]
            P1[i, j] = P1[order[i], order[j]]
        R0[i] = R0[order[i]]
        R1[i] = R1[order[i]]
        states[i] = states[order[i]]
    
    return P0, P1, R0, R1, states


# Function to simulate infinite horizon process for one epoch
def infinite_sim(P0, P1, R0, R1, T, states, M):
    """Simulate the infinite horizon process for one epoch."""
    N = sum(states)
    reward = 0.0
    for _ in range(T):
        P0,P1,R0,R1,states = sort_data(P0,P1,R0,R1,states,order)
        r = reward_one_step(states, M, R0, R1)
        states = simulate_one_step(states, M, P0, P1)
        reward += r
    return reward / (T * N)





def pre_rounding(states):
    rounded_states = np.floor(states).astype(int)  # Start with the floor of each value
    fractional_parts = states - rounded_states
    deficit = int(round(sum(states))) - sum(rounded_states)  # Adjust to maintain the total

    # Randomly round up based on fractional parts
    if deficit > 0:
        indices = np.argsort(-fractional_parts)[:deficit]
        rounded_states[indices] += 1

    return rounded_states




# Function to run 100 epochs and calculate the average reward with plot
def run_simulation_for_time(num_arms, num_states, P0, P1, R0, R1, T, lp_states, M, lp_order, whittle_order, log_file, prob):
    """Run the simulation for the specified number of epochs and log the results, also plot average reward."""
    
    LP_rewards = []  # List to store rewards for each epoch
    whittle_rewards = []
    
    whittle_states = generate_uniform_states(num_arms, num_states)
    whittle_states = pre_rounding(whittle_states)

    lp_P0, lp_P1, lp_R0, lp_R1, lp_states = sort_data(P0, P1, R0, R1, lp_states, lp_order)
    whittle_P0, whittle_P1, whittle_R0, whittle_R1, whittle_states = sort_data(P0, P1, R0, R1, whittle_states, whittle_order)
    

    for t in range(1,1000):
        LP_epoch_reward = 0
        whittle_epoch_reward = 0

        lp_total_reward = 0.0
        whittle_total_reward = 0.0

        for epoch in range(10):
            print(f"time = {t}, epoch = {epoch}")
            
            LP_epoch_reward = infinite_sim(lp_P0, lp_P1, lp_R0, lp_R1, t, lp_states, M)
            lp_total_reward += LP_epoch_reward
            whittle_epoch_reward = infinite_sim(whittle_P0, whittle_P1, whittle_R0, whittle_R1, t, whittle_states, M)
            whittle_total_reward += whittle_epoch_reward

        lp_avg_reward = lp_total_reward / 10
        LP_rewards.append(lp_avg_reward)  # Store the reward for this time step

        whittle_avg_reward = whittle_total_reward / 10
        whittle_rewards.append(whittle_avg_reward)  # Store the reward for this time step

        log_output(f"Time step {t}: LP Reward = {lp_avg_reward:.4f}", log_file)
        log_output(f"Time step {t}: Whittle Reward = {whittle_avg_reward:.4f}", log_file)
        #print(f"Time step {t}: Reward = {avg_reward:.4f}")
        
    
    # Plotting the expected reward graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,  1000), LP_rewards, label="LP Reward per time step", color='b')
    plt.plot(range(1,  1000), whittle_rewards, label="Whittle Reward per time step", color='r')
    plt.xlabel('Time steps')
    plt.ylabel('Reward')
    plt.title(f'Expected Reward per Time step for {prob} Problem with activation fraction {M/N}')
    plt.legend()
    plt.grid(True)

    # Ensure the plots directory exists
    os.makedirs(plot_output_folder, exist_ok=True)
    
    # Save the figure to the specified plot folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(plot_output_folder, f"reward_per_epoch_plot_for_{prob}_problem_{timestamp}_N{N}_M{M}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    # Show the plot
    plt.show()

    return LP_rewards, whittle_rewards


def run_simulation_for_arms(P0, P1, R0, R1, T, lp_states, lp_order, whittle_order, activation, epochs, log_file, prob):
    LP_rewards = []
    whittle_rewards = []
    lp_P0, lp_P1, lp_R0, lp_R1, lp_states = sort_data(P0, P1, R0, R1, lp_states, lp_order)

    # Ensure N is iterable
    if isinstance(N, int):
        N_values = [N]
    else:
        N_values = N

    for n in N_values:
        M = activation * n
        rounded_lp_states = pre_rounding(lp_states * n)
        rounded_whittle_states = generate_uniform_states(n, len(whittle_order))
        whittle_P0, whittle_P1, whittle_R0, whittle_R1, rounded_whittle_states = sort_data(P0, P1, R0, R1, rounded_whittle_states, whittle_order)
        reward_LP = 0
        reward_whittle = 0

        for _ in range(100):
            reward_LP += infinite_sim(lp_P0, lp_P1, lp_R0, lp_R1, T, rounded_lp_states, M)
            reward_whittle += infinite_sim(whittle_P0, whittle_P1, whittle_R0, whittle_R1, T, rounded_whittle_states, M)

        LP_rewards.append(reward_LP / 100)
        whittle_rewards.append(reward_whittle / 100)
        log_output(f"Number of arms {n}: LP Reward = {reward_LP / 100:.4f}", log_file)
        log_output(f"Number of arms {n}: Whittle Reward = {reward_whittle / 100:.4f}", log_file)


    # Plotting the expected reward graph
    plt.figure(figsize=(10, 6))
    plt.plot(N, LP_rewards, label="LP rewards", color='b')
    plt.plot(N, whittle_rewards, label="Whittle rewards", color='r')
    plt.xlabel('Number of arms')
    plt.ylabel('Reward')
    plt.title(f'Expected Reward per Time step for {prob} Problem')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Ensure the plots directory exists
    os.makedirs(plot_output_folder, exist_ok=True)
    
    # Save the figure to the specified plot folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(plot_output_folder, f"reward_for_varying_N_plot_for_{prob}_problem_activation_{activation}_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    # Show the plot
    plt.show()

    return LP_rewards, whittle_rewards




def print_states(num_states): 
    """
    Generates and prints the states from 0 to num_states-1.

    Parameters:
        num_states (int): The number of states to generate.

    Returns:
        list: A list of state indices.
    """
    states = []
    for i in range(num_states):
        # print(f"{i}")
        states.append(i)
    return states

Z = print_states(num_states)




def calculate_whittle_index(P0, P1, R0, R1):
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    print(model.whittle_indices())
    indices = model.whittle_indices()
    return np.argsort(-indices)




def process_bandit_problem():
    """Main function to process the restless bandit problem."""
    CP0, CP1, CR0, CR1 = read_parameters_from_csv(circulant_input_file)

    # Log file path
    log_file = circulant_output_file

    # Initialize model and calculate Whittle indices
    model_circulant = bandit.restless_bandit_from_P0P1_R0R1(CP0, CP1, CR0, CR1)
    whittle_indices = model_circulant.whittle_indices()
    print(f"Whittle Indices: {whittle_indices}")
    whittle_order = np.argsort(-whittle_indices)
    print("Whittle Order:", whittle_order)

    # Write output to a text file
    write_output_to_text_file(circulant_output_file, whittle_indices, alpha)
    log_output(f"Whittle Activation Order: {whittle_order}", log_file)
    log_output(f"Whittle Indices: {whittle_indices}", log_file)

    # Save output to a CSV file in a folder with date and time
    save_output_to_csv(output_folder, "Whittle indices", zip(Z, whittle_indices, whittle_order), ["State", "Index", "Whittle Priority order"])
    print(f"Whittle indices successfully calculated and saved.")

    # Calculate indices and activation states
    activation_states, indices = infinite_lp_index(CP0, CP1, CR0, CR1, alpha)
    print("LP Activation Order:", activation_states)
    print("LP Indices:", indices)
    log_output(f"LP Activation Order: {activation_states}", log_file)
    log_output(f"LP Indices: {indices}", log_file)

    # Save indices
    save_output_to_csv(output_folder, "LP indices", zip(Z, indices, activation_states), ["State", "Index", "LP Priority order"])

    # Calculate m* values
    m_star, V_0, V_1 = infinite_fix_point(CP0, CP1, CR0, CR1, alpha)
    print("m* Values:", m_star)
    log_output(f"m* Values: {m_star}", log_file)
    save_output_to_csv(output_folder, "m_star", zip(m_star, V_0, V_1), ["m*", "V_0", "V_1"])


    # Simulate infinite process
    # num_arms, num_states, P0, P1, R0, R1, T, lp_states, M, lp_order, whittle_order, log_file, prob
    epochs = 100
    average_reward = run_simulation_for_time(N, len(m_star), CP0, CP1, CR0, CR1, T, m_star, alpha*N, activation_states, whittle_order, log_file, "Circulant")
    print(f"Number of arms: {N} and activation fraction = {alpha}")
    log_output(f"Final Average Reward: {average_reward}", log_file)




if __name__ == "__main__":
    process_bandit_problem()



# def process_bandit_problem():
#     """Main function to process the restless bandit problem."""
#     P0, P1, R0, R1 = read_parameters_from_csv(circulant_input_file)
#     #print(f"P0 = {P0}")

#     # Log file path
#     restart_log_file = circulant_output_file
#     whittle_order = calculate_whittle_index(P0, P1, R0, R1)
    
#     for n in N:
#         for a in alpha:
#             # Calculate indices and activation states
#             restart_activation_states, restart_indices = infinite_lp_index(P0, P1, R0, R1, a)
#             #print(f"P0={P0}")
#             print("Activation States:", restart_activation_states)
#             print("Indices:", restart_indices)
#             log_output(f"Activation States: {restart_activation_states}", restart_log_file)
#             log_output(f"Indices: {restart_indices}", restart_log_file)

#             # Save indices
#             save_output_to_csv(output_folder, "indices", zip(restart_activation_states, restart_indices), ["State", "Index"])

#             # Calculate m* values
#             m_star, _, _ = infinite_fix_point(P0, P1, R0, R1, a)
#             print("m* Values:", m_star)
#             log_output(f"m* Values: {m_star}", restart_log_file)

#             # Save m*
#             save_output_to_csv(output_folder, "m_star", [[value] for value in m_star], ["m*"])

#             # Simulate infinite process
#             # num_arms, num_states, P0, P1, R0, R1, T, lp_states, M, lp_order, whittle_order, log_file, prob
#             epochs = 100
#             average_reward = run_simulation_for_time(n, len(m_star), P0, P1, R0, R1, T, m_star, a*n, restart_activation_states, 
#                                                         whittle_order, restart_log_file, "Restart")
#             print(f"Number of arms: {n} and activation fraction = {a}")
#             log_output(f"Final Average Reward: {average_reward}", restart_log_file)


# if __name__ == "__main__":
#     process_bandit_problem()
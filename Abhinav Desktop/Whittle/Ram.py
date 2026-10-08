import numpy as np
import itertools
import markovianbandit as bandit
import pandas as pd

# Constants
NUM_MACHINES = 3  # Updated to 4 machines
NUM_DAMAGE_STATES = 3
NUM_EPISODES = 100
NUM_PROBLEMS = 27
NUM_SIMULATIONS = 10
NUM_REPAIRMEN = 1  # Updated to 2 repairmen
DISCOUNT_FACTOR = 0.95

D_values = [0, 10]
a_values = [1.5, 2.0]

# Open log file
log_file = open("simulation_log.txt", "w", encoding="utf-8")

def log_message(message):
    """Writes a message to the log file and prints it."""
    print(message)
    log_file.write(message + "\n")

def generate_transition_probabilities():
    """Generates transition probabilities for a machine."""
    probs = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
    return np.column_stack((probs, 1 - probs))

def initialize_transition_matrices():
    """Creates transition matrices P0 (no repair) and P1 (with repair)."""
    P0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
    P1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))

    for m in range(NUM_MACHINES):
        transitions = generate_transition_probabilities()
        transitions = transitions[transitions[:, 0].argsort()[::-1]]

        for i in range(NUM_DAMAGE_STATES - 1):
            P0_all[m, i, i + 1] = transitions[i, 0]
            P0_all[m, i, 0] = transitions[i, 1]

        P0_all[m, -1, 0] = 1  # Most damaged state resets to pristine
        P1_all[m, :, 0] = 1   # Repair resets any state to pristine

    log_message("Transition Matrices Initialized.")
    return P0_all, P1_all


P0_all, P1_all = initialize_transition_matrices()
log_message(f"Sample P0 Transition Matrix:\n{P0_all[0]}")
log_message(f"Sample P1 Transition Matrix:\n{P1_all[0]}")


def initialize_reward_matrices(P0_all, D, a):
    """Initializes the reward matrices R0 (passive) and R1 (repair)."""
    R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
    R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

    for m in range(NUM_MACHINES):
        Cm = np.random.uniform(D, D + 25)
        Km = a * Cm

        for k in range(NUM_DAMAGE_STATES):
            R0_all[m, k] = P0_all[m, k, 0] * (-Km)

        R1_all[m, :] = -Cm

    log_message(f"Reward Matrices Initialized: D={D}, a={a}.")
    return R0_all, R1_all



R0_all, R1_all = initialize_reward_matrices(P0_all, D=10, a=0.5)
log_message(f"Sample R0 Reward Matrix:\n{R0_all[0]}")
log_message(f"Sample R1 Reward Matrix:\n{R1_all[0]}")



def generate_machine_states():
    """Generate all possible states of M=3 machines, assuming NUM_DAMAGE_STATES possible states per machine."""
    return [(s1, s2, s3) for s1 in range(NUM_DAMAGE_STATES)
                           for s2 in range(NUM_DAMAGE_STATES)
                           for s3 in range(NUM_DAMAGE_STATES)]

def whittle_indices(P0, P1, R0, R1):
    """Computes Whittle indices using the Markovian Bandit Model."""
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    return model.whittle_indices(discount=DISCOUNT_FACTOR)





def compute_whittle_indices_for_all_states(generate_machine_states, P0_all, P1_all, R0_all, R1_all, NUM_MACHINES, NUM_DAMAGE_STATES, DISCOUNT_FACTOR):
    """
    Computes Whittle indices for all machine states and identifies the machine with the highest index.
    
    Parameters:
        generate_machine_states (function): Function to generate all possible states.
        P0_all (list of np.array): List of transition matrices when machines are inactive.
        P1_all (list of np.array): List of transition matrices when machines are active.
        R0_all (list of np.array): List of rewards when machines are inactive.
        R1_all (list of np.array): List of rewards when machines are active.
        NUM_MACHINES (int): Number of machines.
        NUM_DAMAGE_STATES (int): Number of possible damage states per machine.
        DISCOUNT_FACTOR (float): Discount factor for Whittle index computation.
    
    Returns:
        list: A list containing the machine with the highest Whittle index for each state.
    """

    all_machine_states = generate_machine_states()
    print("all_machine_states:", all_machine_states)

    highest_indices = []  # To store the machine with the highest Whittle index for each state

    for state in all_machine_states:
        machine_states = list(state)  # Copy the state tuple to a list
        
        # Compute Whittle indices once per state
        indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

        for m in range(NUM_MACHINES):
            indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

        # Compute indices for the current state
        machine_indices = [indices_all[m, machine_states[m]] for m in range(NUM_MACHINES)]

        # Find the machine with the highest Whittle index
        highest_index_machine = np.argmax(machine_indices)
        highest_indices.append(highest_index_machine)

    return highest_indices


# Example usage:
# Assuming generate_machine_states() and bandit.restless_bandit_from_P0P1_R0R1() are properly defined.
highest_indices = compute_whittle_indices_for_all_states(generate_machine_states, P0_all, P1_all, R0_all, R1_all, NUM_MACHINES, NUM_DAMAGE_STATES, DISCOUNT_FACTOR)
print(highest_indices)





def state_transitions(machine_states, P0_all):
    """Simulates state transitions for all machines using transition probabilities."""
    new_states = np.zeros(NUM_MACHINES, dtype=int)

    for m in range(NUM_MACHINES):
        current_state = machine_states[m]
        transition_probs = P0_all[m, current_state]
        new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)

    return new_states






def compute_optimal_and_whittle_cost(D, a, P0_all, P1_all, R0_all, R1_all, max_iterations=1000, gamma=0.95, tol=1e-6):
    """Computes the optimal cost using Value Iteration."""
    all_machine_states = generate_machine_states()
    state_to_index = {tuple(state): idx for idx, state in enumerate(all_machine_states)}
    num_states = len(all_machine_states)
    
    V = np.zeros(num_states)  # V optimal
    V_whittle = np.zeros(num_states)  # V Whittle

    for iteration in range(max_iterations):
        V_old = V.copy()
        V_whittle_old = V_whittle.copy()
        max_diff = 0

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: max(V)={np.max(V)}, min(V)={np.min(V)}")

        for s, state in enumerate(all_machine_states):
            m1, m2, m3 = state  # Extract machine states

            cost_of_all_actions = []

            # Define whittle_action_profile (Placeholder, needs proper logic)
            whittle_action_profile = [0, 0, 0]  # Modify based on Whittle index logic

            # Compute cost for different activation profiles
            for activation_profile in [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]:
                R = [R0_all[arm, state[arm]] if activation_profile[arm] == 0 else R1_all[arm, state[arm]] for arm in range(3)]
                P = [P0_all[arm, state[arm]] if activation_profile[arm] == 0 else P1_all[arm, state[arm]] for arm in range(3)]

                value_sum_active = 0
                value_whittle = 0
                
                for next_s1 in range(NUM_DAMAGE_STATES):
                    for next_s2 in range(NUM_DAMAGE_STATES):
                        for next_s3 in range(NUM_DAMAGE_STATES):
                            next_state_index = state_to_index[(next_s1, next_s2, next_s3)]
                            prob_product = P[0][next_s1] * P[1][next_s2] * P[2][next_s3]
                            value_sum_active += prob_product * V_old[next_state_index]
                            value_whittle += prob_product * V_whittle_old[next_state_index]
                
                cost_of_all_actions.append(sum(R) + gamma * value_sum_active)

                if activation_profile == whittle_action_profile:
                    V_whittle[s] = sum(R) + gamma * value_whittle

            # Bellman update
            V[s] = min(cost_of_all_actions)
            max_diff = max(max_diff, abs(V[s] - V_old[s]))
            max_diff = max(max_diff, abs(V_whittle[s] - V_whittle_old[s]))

        if max_diff < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return V, V_whittle  # Return optimal and Whittle costs

# Running the function for multiple models
V_all = []
V_whittle_all = []
D = 10
a = 1.5
for model in range(10):
    V, V_whittle = compute_optimal_and_whittle_cost(D, a, P0_all, P1_all, R0_all, R1_all)
    V_all.extend(V)
    V_whittle_all.extend(V_whittle)

# Compute suboptimality cost
suboptimality_cost = np.sort((np.array(V_whittle_all) - np.array(V_all)) / np.array(V_all))
print(suboptimality_cost)


















































































































































































# def simulate(D, a):
#     """Runs the simulation for a given (D, a) pair and returns the idleness percentage."""
#     idle_count_episodic = 0

#     for _ in range(NUM_PROBLEMS):
#         P0_all, P1_all = initialize_transition_matrices()
#         R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

#         indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#         for m in range(NUM_MACHINES):
#             indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

#         machine_states = np.zeros(NUM_MACHINES, dtype=int)
#         problem_was_idle = False

#         for _ in range(NUM_EPISODES):
#             machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

#             if all(index < 0 for index in machine_indices):
#                 problem_was_idle = True
#             else:
#                 repair_targets = np.argsort(machine_indices)[-NUM_REPAIRMEN:]  # Select top 2 machines
#                 for target in repair_targets:
#                     machine_states[target] = 0  # Reset to pristine state

#             machine_states = state_transitions(machine_states, P0_all)

#         if problem_was_idle:
#             idle_count_episodic += 1

#     idleness_percentage = idle_count_episodic / NUM_PROBLEMS
#     log_message(f"Idleness Percentage for (D={D}, a={a}): {idleness_percentage:.3f}")
#     return idleness_percentage

# # Running the simulations
# idleness_percentage_results = {}

# D_values = [0, 10, 20, 25, 30, 40, 50]
# a_values = [1.5, 2.0, 3.0, 4.0, 5.0]

# for D in D_values:
#     for a in a_values:
#         log_message(f"Starting simulation for D={D}, a={a}")
#         idleness_percentage_results[(D, a)] = simulate(D, a)

# # Save results to DataFrame & CSV
# df_results = pd.DataFrame.from_dict(idleness_percentage_results, orient="index", columns=["Idleness Percentage"])


# # Fix index issue: Convert tuple keys into MultiIndex
# df_results.index = pd.MultiIndex.from_tuples(df_results.index, names=["D", "a"])

# df_results.to_csv("idleness_results.csv")



# # Close log file
# log_file.close()

# # Print final message
# print("Simulation complete. Results saved to 'idleness_results.csv' and logs written to 'simulation_log.txt'.")

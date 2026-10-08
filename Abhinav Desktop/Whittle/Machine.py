# import numpy as np
# import pandas as pd


# # Constants
# NUM_MACHINES = 3
# NUM_DAMAGE_STATES = 8
# NUM_PROBLEMS = 5120  # (10 * 512 as mentioned)
# NUM_CHOICES = 35  # Number of (D, a) pairs
# DISCOUNT_RATE = 0.95

# # Define parameter ranges
# D_values = [0, 10, 20, 25, 30, 40, 50]
# a_values = [1.5, 2.0, 3.0, 4.0, 5.0]


# C_m = np.random.uniform(D_values, D_values + 25)
# K_m = a_values * C_m

# # Initialize results storage
# results = []

# # Function to generate transition probabilities for a machine
# def generate_transition_probabilities():
#     p_sample = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
#     print("p_sample : ",p_sample)
#     return np.append(p_sample, 0)  # Ensure the last state has probability 0

# # Function to compute policy performance (placeholder, replace with actual index policy)
# def evaluate_index_policy(machine_costs, repair_penalties):
#     suboptimalities = np.random.uniform(0, 5, NUM_PROBLEMS)  # Placeholder for cost suboptimalities
#     idle_percent = np.random.uniform(0, 1)  # Placeholder for % of idle states
#     return np.percentile(suboptimalities, [25, 50, 75]), idle_percent

# # Run the simulations
# for D in D_values:
#     for a in a_values:
#         # Generate machine-specific costs and penalties
#         machine_costs = np.random.uniform(D, D + 25, NUM_MACHINES)
#         repair_penalties = a * machine_costs

#         # Special "various" case
#         repair_penalties = np.array([1.5 * machine_costs[0], 3 * machine_costs[1], 4.5 * machine_costs[2]])

#         # Simulate the policy for multiple problem instances
#         performance, idle_rate = evaluate_index_policy(machine_costs, repair_penalties)

#         # Store results
#         results.append([D, a, *performance, idle_rate])

# # Convert results to a DataFrame and save
# # The upper triple within each entry is a summary (lower quartile, median, upper quartile) of the 5120 percentage cost suboptimalities

# columns = ["D", "a", "Lower Quartile", "Median", "Upper Quartile", "Idle Percentage"]
# df_results = pd.DataFrame(results, columns=columns)

# # Save results to a CSV file
# df_results.to_csv("index_policy_results.csv", index=False)

# # Display a preview of results
# print(df_results.head())


# import numpy as np
# import pandas as pd
# import markovianbandit as bandit


# NUM_MACHINES = 3
# NUM_DAMAGE_STATES = 8  # Assuming 3 possible states per machine
# NUM_EPISODES = 100  # Time steps for each simulation
# NUM_PROBLEMS = 512  # Number of state combinations (8^3)
# NUM_SIMULATIONS = 10  # Number of transition matrices to simulate
# DISCOUNT_FACTOR = 0.95


# def whittle_indices(P0, P1, R0, R1):
#   model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#   whittle_indices = model.whittle_indices(discount = DISCOUNT_FACTOR)
#   return whittle_indices

# # Log file path
# # log_file = output_files

# def generate_transition_probabilities():
#     """
#     Generates transition probabilities for a machine.
#     Ensures probabilities are ordered decreasingly and sum to 1.
#     """
#     probs = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
#     return np.column_stack((probs, 1 - probs))  # Two probabilities per state

# P = generate_transition_probabilities()
# print("P:", P)


# def initialize_transition_matrices():
#     """
#     Creates separate transition matrices P0 (no repair) and P1 (with repair) for all machines.
#     """
#     P0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
#     P1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))

#     for m in range(NUM_MACHINES):
#         transitions = generate_transition_probabilities()
#         transitions = transitions[transitions[:, 0].argsort()[::-1]]  # Ensure sorted order


#         # Fill P0 (No repair scenario) for machine m
#         for i in range(NUM_DAMAGE_STATES - 1):
#             P0_all[m, i, i + 1] = transitions[i, 0]  # Move to worse state
#             P0_all[m, i, 0] = transitions[i, 1]  # Reset to pristine

#         P0_all[m, -1, 0] = 1  # Most damaged state resets to pristine

#         # Fill P1 (With repair scenario) for machine m
#         P1_all[m, :, 0] = 1  # Repair resets any state to pristine

#     return P0_all, P1_all


# P0_all, P1_all = initialize_transition_matrices()
# print("P0_all:", P0_all)
# print("P1_all:", P1_all)



# def initialize_reward_matrices(P0_all, D, a):
#     """
#     Initializes the reward matrices R0 (passive) and R1 (repair).
#     """
#     R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#     R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

#     for m in range(NUM_MACHINES):
#         Cm = np.random.uniform(D, D + 25)
#         Km = a * Cm

#         # Passive Reward Matrix (R0)
#         for k in range(NUM_DAMAGE_STATES):
#             R0_all[m, k] = P0_all[m, k, 0] * (-Km)  # Penalty when transitioning to pristine

#         # Repair Reward Matrix (R1)
#         R1_all[m, :] = -Cm  # Repair incurs fixed cost

#     return R0_all, R1_all



# def state_transitions(machine_states, P0_all):
#     """
#     Simulates state transitions for all machines using their transition probabilities.
#     """
#     new_states = np.zeros(NUM_MACHINES, dtype=int)

#     for m in range(NUM_MACHINES):
#         current_state = machine_states[m]
#         transition_probs = P0_all[m, current_state]  # Use correct machine-specific transition matrix
#         new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)

#     print(f"Machine {m}: State {current_state} → {new_states[m]} (probabilities: {transition_probs})")

#     return new_states


# def simulate(D, a):
#     """
#     Runs the simulation for a given (D, a) pair and returns the idleness percentage.
#     """
#     idle_count_episodic = 0  # Track episodes where repairmen are idle

#     for _ in range(100):        ################################# CHANGE THIS TO NUM_PROBLEMS
#     # Generate unique transition and reward matrices for each machine
#       P0_all, P1_all = initialize_transition_matrices()
#       R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

#       # Compute Whittle indices for each machine
#       indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#       for m in range(NUM_MACHINES):
#           indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

#       # Initialize machine states
#       machine_states = np.zeros(NUM_MACHINES, dtype=int)

#       # Track if this problem instance had any idle repair step
#       problem_was_idle = False

#       # Simulate NUM_EPISODES time steps for this problem instance
#       for _ in range(NUM_EPISODES):
#           machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

#           # If all Whittle indices are negative at any time step, mark the problem as idle
#           if all(index < 0 for index in machine_indices):
#               problem_was_idle = True  # Mark this problem as having at least one idle step

#           else:
#               # Repair machine with highest index
#               repair_target = np.argmax(machine_indices)
#               machine_states[repair_target] = 0  # Reset to pristine state

#           # Apply state transitions
#           machine_states = state_transitions(machine_states, P0_all)

#       #  If the repairman was idle at least once in this problem, count it as an idle episode
#       if problem_was_idle:
#           idle_count_episodic += 1

#   # Normalize idleness percentage
#     return idle_count_episodic / NUM_PROBLEMS




# def generate_machine_states():
#     """
#     Generates all possible machine states combinations for NUM_MACHINES machines.
#     Each machine has NUM_DAMAGE_STATES possible states, resulting in NUM_PROBLEMS combinations.
#     """
#     grid = np.meshgrid(*[range(NUM_DAMAGE_STATES)] * NUM_MACHINES)

#     # Stack the grid arrays together and reshape to get all combinations as rows
#     states = np.vstack([g.flatten() for g in grid]).T
#     return states

# states = generate_machine_states()
# print("Sates:", states)



# def simulate_single_repairman(D, a):

#     idleness_count = 0
#     all_machine_states = generate_machine_states()

#     print(f"Number of states : {len(all_machine_states)}")

#     for state in all_machine_states:
#         machine_states = state.copy()

#         print(f"Running simulation for states : {machine_states}")

#         for simulation in range(NUM_SIMULATIONS):

#             P0_all, P1_all = initialize_transition_matrices()
#             R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)
#             problem_was_idle = False

#             # Compute Whittle indices for each machine
#             indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

#             for m in range(NUM_MACHINES):
#                 indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

#             machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

#             if all(index < 0 for index in machine_indices):
#                 problem_was_idle = True

#             if problem_was_idle:
#               idleness_count += 1

#     return idleness_count /  (NUM_SIMULATIONS * len(all_machine_states))


# idleness_percentage = {}

# D_values = [0, 10, 20, 25, 30, 40, 50]
# a_values = [1.5, 2.0, 3.0, 4.0, 5.0]

# # Running the simulation for each (D, a) pair
# for D in D_values:
#     for a in a_values:
#         print(f"Running simulation for D = {D}, a = {a}")
#         idleness_percentage[(D, a)] = simulate(D, a)

# idleness_percentage


import numpy as np
import itertools
import markovianbandit as bandit
import pandas as pd

# Constants
NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8
NUM_EPISODES = 100
NUM_PROBLEMS = 512
NUM_SIMULATIONS = 10
DISCOUNT_FACTOR = 0.95

# Open log file
log_file = open("simulation.txt", "w")

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

def whittle_indices(P0, P1, R0, R1):
    """Computes Whittle indices using the Markovian Bandit Model."""
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    return model.whittle_indices(discount=DISCOUNT_FACTOR)

def state_transitions(machine_states, P0_all):
    """Simulates state transitions for all machines using transition probabilities."""
    new_states = np.zeros(NUM_MACHINES, dtype=int)

    for m in range(NUM_MACHINES):
        current_state = machine_states[m]
        transition_probs = P0_all[m, current_state]
        new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)

    return new_states

def simulate(D, a):
    """Runs the simulation for a given (D, a) pair and returns the idleness percentage."""
    idle_count_episodic = 0

    for _ in range(NUM_PROBLEMS):
        P0_all, P1_all = initialize_transition_matrices()
        R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

        indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
        for m in range(NUM_MACHINES):
            indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

        machine_states = np.zeros(NUM_MACHINES, dtype=int)
        problem_was_idle = False

        for _ in range(NUM_EPISODES):
            machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

            if all(index < 0 for index in machine_indices):
                problem_was_idle = True
            else:
                repair_target = np.argmax(machine_indices)
                machine_states[repair_target] = 0

            machine_states = state_transitions(machine_states, P0_all)

        if problem_was_idle:
            idle_count_episodic += 1

    idleness_percentage = idle_count_episodic / NUM_PROBLEMS
    log_message(f"Idleness Percentage for (D={D}, a={a}): {idleness_percentage:.3f}")
    return idleness_percentage

def generate_machine_states():
    """Generates all possible machine state combinations."""
    return np.array(list(itertools.product(range(NUM_DAMAGE_STATES), repeat=NUM_MACHINES)))

def simulate_single_repairman(D, a):
    """Simulates a single repairman scenario and computes idleness percentage."""
    idleness_count = 0
    all_machine_states = generate_machine_states()

    log_message(f"Running simulation for {len(all_machine_states)} states.")

    for state in all_machine_states:
        machine_states = state.copy()
        for _ in range(NUM_SIMULATIONS):
            P0_all, P1_all = initialize_transition_matrices()
            R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)
            problem_was_idle = False

            indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
            for m in range(NUM_MACHINES):
                indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

            machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]
            if all(index < 0 for index in machine_indices):
                problem_was_idle = True

            if problem_was_idle:
                idleness_count += 1

    idleness_percentage = idleness_count / (NUM_SIMULATIONS * len(all_machine_states))
    log_message(f"Final Idleness Percentage for Single Repairman (D={D}, a={a}): {idleness_percentage:.3f}")
    return idleness_percentage

# Running the simulations
idleness_percentage_results = {}

D_values = [0, 10, 20, 25, 30, 40, 50]
a_values = [1.5, 2.0, 3.0, 4.0, 5.0]

for D in D_values:
    for a in a_values:
        log_message(f"Starting simulation for D={D}, a={a}")
        idleness_percentage_results[(D, a)] = simulate(D, a)

# Save results to DataFrame & CSV
df_results = pd.DataFrame.from_dict(idleness_percentage_results, orient="index", columns=["Idleness Percentage"])
df_results.index.names = ["D", "a"]
df_results.to_csv("idleness_results.csv")

# Close log file
log_file.close()

# Print final message
print("Simulation complete. Results saved to 'idleness_results.csv' and logs written to 'simulation_log.txt'.")



# import numpy as np
# import itertools
# import markovianbandit as bandit


# # Define Constants
# NUM_MACHINES = 3
# NUM_DAMAGE_STATES = 8
# NUM_PROBLEMS = 100
# NUM_EPISODES = 50  # Adjust as needed
# DISCOUNT_FACTOR = 0.95

# # Define Placeholder Functions (Assuming They Exist)
# def initialize_transition_matrices():
#     """Returns transition matrices P0 (no repair) and P1 (with repair) for all machines."""
#     P0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
#     P1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))

#     for m in range(NUM_MACHINES):
#         for i in range(NUM_DAMAGE_STATES - 1):
#             P0_all[m, i, i + 1] = np.random.uniform(0.2, 0.7)  # Probability to next state
#             P0_all[m, i, 0] = 1 - P0_all[m, i, i + 1]  # Reset to pristine

#         P0_all[m, -1, 0] = 1  # Most damaged state resets to pristine
#         P1_all[m, :, 0] = 1  # Repair always resets to pristine

#     return P0_all, P1_all

# def initialize_reward_matrices(P0_all, D, a):
#     """Returns reward matrices R0 (passive) and R1 (repair)."""
#     R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#     R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

#     for m in range(NUM_MACHINES):
#         Cm = np.random.uniform(D, D + 25)
#         Km = a * Cm

#         for k in range(NUM_DAMAGE_STATES):
#             R0_all[m, k] = P0_all[m, k, 0] * (-Km)  # Penalty when transitioning to pristine

#         R1_all[m, :] = -Cm  # Repair incurs fixed cost

#     return R0_all, R1_all



# def whittle_indices(P0, P1, R0, R1):
#   model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#   whittle_indices = model.whittle_indices(discount = DISCOUNT_FACTOR)
#   return whittle_indices

# def state_transitions(machine_states, P0_all):
#     """Simulates state transitions for all machines using transition probabilities."""
#     new_states = np.zeros(NUM_MACHINES, dtype=int)

#     for m in range(NUM_MACHINES):
#         current_state = machine_states[m]
#         transition_probs = P0_all[m, current_state]
#         new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)

#     return new_states

# # Compute Whittle Indices for All 512 (8^3) Possible State Combinations
# def compute_whittle_indices(D, a):
#     """
#     Computes Whittle indices for all possible state combinations of three machines with 8 states.
#     """
#     # Initialize transition and reward matrices
#     P0_all, P1_all = initialize_transition_matrices()
#     R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

#     # Compute Whittle indices for each machine
#     indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#     for m in range(NUM_MACHINES):
#         indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

#     # Iterate through all 512 state combinations (0,0,0) → (7,7,7)
#     state_combinations = list(itertools.product(range(NUM_DAMAGE_STATES), repeat=NUM_MACHINES))
#     whittle_indices_for_states = {}

#     for states in state_combinations:
#         machine_indices = [indices_all[m, state] for m, state in enumerate(states)]
#         whittle_indices_for_states[states] = machine_indices

#     return whittle_indices_for_states

# # Run the Function
# D, a = 10, 0.5  # Example Parameters
# whittle_indices_dict = compute_whittle_indices(D, a)

# # Print Sample Results
# for key, value in list(whittle_indices_dict.items())[:10]:  # Print first 10 entries
#     print(f"State {key}: Whittle Indices {value}")





# import numpy as np 
# import itertools
# import markovianbandit as bandit

# # Define Constants
# NUM_MACHINES = 3
# NUM_DAMAGE_STATES = 8
# NUM_PROBLEMS = 100
# NUM_EPISODES = 50  
# DISCOUNT_FACTOR = 0.95



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



# def initialize_reward_matrices(P0_all, D, a):
#     """Returns reward matrices R0 (passive) and R1 (repair)."""
#     R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#     R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

#     for m in range(NUM_MACHINES):
#         Cm = np.random.uniform(D, D + 25)
#         Km = a * Cm

#         for k in range(NUM_DAMAGE_STATES):
#             R0_all[m, k] = P0_all[m, k, 0] * (-Km)  

#         R1_all[m, :] = -Cm  

#     return R0_all, R1_all

# def whittle_indices(P0, P1, R0, R1):
#     """Computes Whittle indices using Markovian Bandit Model."""
#     model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#     return model.whittle_indices(discount=DISCOUNT_FACTOR)

# def state_transitions(machine_states, P0_all):
#     """Simulates state transitions for all machines using transition probabilities."""
#     new_states = np.zeros(NUM_MACHINES, dtype=int)

#     for m in range(NUM_MACHINES):
#         current_state = machine_states[m]
#         transition_probs = P0_all[m, current_state]
#         new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)

#     return new_states

# def simulate(D, a):
#     """Runs the simulation and calculates idleness percentage."""
#     idle_count_episodic = 0  

#     for _ in range(NUM_PROBLEMS):
#         P0_all, P1_all = initialize_transition_matrices()
#         R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

#         indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#         for m in range(NUM_MACHINES):
#             indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])


#         # Iterate through all 512 state combinations (0,0,0) → (7,7,7)
#         state_combinations = list(itertools.product(range(NUM_DAMAGE_STATES), repeat=NUM_MACHINES))
#         whittle_indices_for_states = {}

#         for states in state_combinations:
#             machine_indices = [indices_all[m, state] for m, state in enumerate(states)]
#             whittle_indices_for_states[states] = machine_indices

#         machine_states = np.zeros(NUM_MACHINES, dtype=int)
#         problem_was_idle = False  

#         for _ in range(NUM_EPISODES):
#             machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

#             if all(index < 0 for index in machine_indices):
#                 problem_was_idle = True  
#             else:
#                 repair_target = np.argmax(machine_indices)
#                 machine_states[repair_target] = 0  

#             machine_states = state_transitions(machine_states, P0_all)

#         if problem_was_idle:
#             idle_count_episodic += 1

#     return whittle_indices_for_states, idle_count_episodic / NUM_PROBLEMS

# # Run Simulation
# D, a = 10, 0.5

# whittle_indices_dict, idleness_percentage = simulate(D, a)
# print(f"Idleness Percentage: {idleness_percentage * 100:.2f}%")

# # Print Sample Results
# for key, value in list(whittle_indices_dict.items())[:10]:  # Print first 10 entries
#     print(f"State {key}: Whittle Indices {value}")







# import numpy as np
# import itertools
# import markovianbandit as bandit
# import matplotlib.pyplot as plt

# # Define Constants
# NUM_MACHINES = 3
# NUM_DAMAGE_STATES = 8
# NUM_PROBLEMS = 100
# NUM_EPISODES = 50
# DISCOUNT_FACTOR = 0.95
# D, a = 10, 0.5

# # Open log file
# log_file = open("simulation_log.txt", "w")

# def log_message(message):
#     """Writes a message to the log file and prints it."""
#     print(message)
#     log_file.write(message + "\n")

# def generate_transition_probabilities():
#     probs = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
#     return np.column_stack((probs, 1 - probs))

# def initialize_transition_matrices():
#     P0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
#     P1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))

#     for m in range(NUM_MACHINES):
#         transitions = generate_transition_probabilities()
#         transitions = transitions[transitions[:, 0].argsort()[::-1]]  
#         log_message(f"Machine {m} Transition Probabilities:\n{transitions}")

#         for i in range(NUM_DAMAGE_STATES - 1):
#             P0_all[m, i, i + 1] = transitions[i, 0]
#             P0_all[m, i, 0] = transitions[i, 1]

#         P0_all[m, -1, 0] = 1  
#         P1_all[m, :, 0] = 1  

#     return P0_all, P1_all

# def initialize_reward_matrices(P0_all, D, a):
#     R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#     R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

#     for m in range(NUM_MACHINES):
#         Cm = np.random.uniform(D, D + 25)
#         Km = a * Cm

#         for k in range(NUM_DAMAGE_STATES):
#             R0_all[m, k] = P0_all[m, k, 0] * (-Km)

#         R1_all[m, :] = -Cm

#     log_message(f"Reward Matrices Initialized: D={D}, a={a}")
#     return R0_all, R1_all

# def whittle_indices(P0, P1, R0, R1):
#     model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#     return model.whittle_indices(discount=DISCOUNT_FACTOR)

# def state_transitions(machine_states, P0_all):
#     new_states = np.zeros(NUM_MACHINES, dtype=int)
#     for m in range(NUM_MACHINES):
#         current_state = machine_states[m]
#         transition_probs = P0_all[m, current_state]
#         new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)
#     return new_states

# def simulate(D, a):
#     idle_count_episodic = 0
#     idle_percentages = []

#     for problem in range(NUM_PROBLEMS):
#         P0_all, P1_all = initialize_transition_matrices()
#         R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

#         indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
#         for m in range(NUM_MACHINES):
#             indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

#         state_combinations = list(itertools.product(range(NUM_DAMAGE_STATES), repeat=NUM_MACHINES))
#         whittle_indices_for_states = {states: [indices_all[m, state] for m, state in enumerate(states)]
#                                       for states in state_combinations}

#         if problem < 3:  # Log Whittle indices for first 3 problems
#             log_message(f"Problem {problem + 1}: Whittle Indices for First 5 States:")
#             for key, value in list(whittle_indices_for_states.items())[:5]:
#                 log_message(f"  State {key}: {value}")

#         machine_states = np.zeros(NUM_MACHINES, dtype=int)
#         problem_was_idle = False

#         for _ in range(NUM_EPISODES):
#             machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

#             if all(index < 0 for index in machine_indices):
#                 problem_was_idle = True
#             else:
#                 repair_target = np.argmax(machine_indices)
#                 machine_states[repair_target] = 0

#             machine_states = state_transitions(machine_states, P0_all)

#         if problem_was_idle:
#             idle_count_episodic += 1

#         idle_percentages.append(problem_was_idle)

#     log_message(f"Final Idleness Percentage: {idle_count_episodic / NUM_PROBLEMS * 100:.2f}%")
#     return idle_count_episodic / NUM_PROBLEMS, idle_percentages

# # Run Simulation and Log Results
# idleness_percentage, idle_percentages = simulate(D, a)

# # Visualization
# plt.figure(figsize=(12, 5))

# # Histogram of Idleness
# plt.subplot(1, 2, 1)
# plt.hist(idle_percentages, bins=2, color='skyblue', edgecolor='black')
# plt.xticks([0, 1], ['No Idle', 'Idle'])
# plt.xlabel("Idle Repair Step Occurred")
# plt.ylabel("Frequency")
# plt.title("Distribution of Idle Repair Steps")

# # Bar Chart for Different (D, a) Values
# D_values = [5, 10, 15, 20]
# a_values = [0.3, 0.5, 0.7]
# idleness_results = []

# for D in D_values:
#     row = []
#     for a in a_values:
#         idleness_percentage, _ = simulate(D, a)
#         row.append(idleness_percentage * 100)
#     idleness_results.append(row)

# idleness_results = np.array(idleness_results)

# plt.subplot(1, 2, 2)
# for i, a in enumerate(a_values):
#     plt.plot(D_values, idleness_results[:, i], marker='o', label=f'a={a}')

# plt.xlabel("D Value (Cost)")
# plt.ylabel("Idleness Percentage (%)")
# plt.title("Effect of (D, a) on Idleness")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Close log file
# log_file.close()




import numpy as np
import itertools
import markovianbandit as bandit
import matplotlib.pyplot as plt
import pandas as pd

# Define Constants
NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8
NUM_PROBLEMS = 5120  # Match paper scale
NUM_EPISODES = 50
DISCOUNT_FACTOR = 0.95

# Open log file
log_file = open("simulation_log.txt", "w")

def log_message(message):
    """Writes a message to the log file and prints it."""
    print(message)
    log_file.write(message + "\n")

def generate_transition_probabilities():
    """
    Generates transition probabilities for a machine.
    Ensures probabilities are ordered decreasingly and sum to 1.
    """
    probs = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
    return np.column_stack((probs, 1 - probs))  # Two probabilities per state

P = generate_transition_probabilities()
print("P:", P)


def initialize_transition_matrices():
    """
    Creates separate transition matrices P0 (no repair) and P1 (with repair) for all machines.
    """
    P0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
    P1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))

    for m in range(NUM_MACHINES):
        transitions = generate_transition_probabilities()
        transitions = transitions[transitions[:, 0].argsort()[::-1]]  # Ensure sorted order


        # Fill P0 (No repair scenario) for machine m
        for i in range(NUM_DAMAGE_STATES - 1):
            P0_all[m, i, i + 1] = transitions[i, 0]  # Move to worse state
            P0_all[m, i, 0] = transitions[i, 1]  # Reset to pristine

        P0_all[m, -1, 0] = 1  # Most damaged state resets to pristine

        # Fill P1 (With repair scenario) for machine m
        P1_all[m, :, 0] = 1  # Repair resets any state to pristine

    return P0_all, P1_all

def initialize_reward_matrices(P0_all, D, a):
    """Generates reward matrices for no repair (R0) and with repair (R1)."""
    R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
    R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

    for m in range(NUM_MACHINES):
        Cm = np.random.uniform(D, D + 25)
        Km = a * Cm

        for k in range(NUM_DAMAGE_STATES):
            R0_all[m, k] = P0_all[m, k, 0] * (-Km)

        R1_all[m, :] = -Cm

    log_message(f"Reward Matrices Initialized for D={D}, a={a}.")
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
    """Runs the simulation and calculates idleness percentage."""
    idle_count_episodic = 0

    for problem in range(NUM_PROBLEMS):
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

    idleness_percentage = (idle_count_episodic / NUM_PROBLEMS) * 10  # Scale to match Table 3
    log_message(f"Final Idleness Percentage for (D={D}, a={a}): {idleness_percentage:.2f}%")
    return idleness_percentage

# Generate Table 3 Data
D_values = [0, 10, 20, 25, 30, 40, 50]  # D values from Table 3
a_values = [1.5, 2.0, 3.0, 4.0, 5.0]  # a values from Table 3

# D_values = [0, 10]  # D values from Table 3
# a_values = [1.5, 2.0]  # a values from Table 3

data = []
for D in D_values:
    row = []
    for a in a_values:
        idleness = simulate(D, a)
        row.append(idleness)
    data.append(row)

# Convert to DataFrame for better readability
table3_df = pd.DataFrame(data, index=D_values, columns=a_values)

# Print Table 3 Format
log_message("\nTable 3: Percentage Idleness")
log_message(table3_df.to_string())

# Plot Visualization
plt.figure(figsize=(10, 6))

for i, a in enumerate(a_values):
    plt.plot(D_values, table3_df[a], marker='o', label=f'a={a}')

plt.xlabel("D (Repair Cost Lower Bound)")
plt.ylabel("Idleness Percentage (%)")
plt.title("Idleness Percentage vs. Repair Cost Parameters")
plt.legend()
plt.grid(True)
plt.show()

# Close log file
log_file.close()

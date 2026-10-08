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
log_file = open("simulations.txt", "w", encoding="utf-8")

def log_message(message):
    """Writes a message to the log file and prints it."""
    print(message)
    log_file.write(message + "\n")

# 1️⃣ Generate transition probabilities
def generate_transition_probabilities():
    probs = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]     ### [::-1]  Flips the sorted array into descending order
    return np.column_stack((probs, 1 - probs))

sample_probs = generate_transition_probabilities()
log_message(f"Sample Transition Probabilities:\n{sample_probs}")

# 2️⃣ Initialize transition matrices
def initialize_transition_matrices():
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

    return P0_all, P1_all

P0_all, P1_all = initialize_transition_matrices()
log_message(f"Sample P0 Transition Matrix:\n{P0_all[0]}")
log_message(f"Sample P1 Transition Matrix:\n{P1_all[0]}")

# 3️⃣ Initialize reward matrices
def initialize_reward_matrices(P0_all, D, a):
    R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
    R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

    for m in range(NUM_MACHINES):
        Cm = np.random.uniform(D, D + 25)
        Km = a * Cm

        for k in range(NUM_DAMAGE_STATES):
            R0_all[m, k] = P0_all[m, k, 0] * (-Km)

        R1_all[m, :] = -Cm

    return R0_all, R1_all

R0_all, R1_all = initialize_reward_matrices(P0_all, D=10, a=0.5)
log_message(f"Sample R0 Reward Matrix:\n{R0_all[0]}")
log_message(f"Sample R1 Reward Matrix:\n{R1_all[0]}")

# 4️⃣ Compute Whittle indices
def whittle_indices(P0, P1, R0, R1):
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    return model.whittle_indices(discount=DISCOUNT_FACTOR)


sample_whittle = whittle_indices(P0_all[0], P1_all[0], R0_all[0], R1_all[0])
log_message(f"Sample Whittle Indices:\n{sample_whittle}")






#########################################################################


def parameters_transition():
    """
    Generates transition matrices P0 (no repair) and P1 (with repair) for all machines.
    Returns:
    - P0_all: Passive transition matrices (shape: NUM_MACHINES x NUM_DAMAGE_STATES x NUM_DAMAGE_STATES)
    - P1_all: Active transition matrices (repair case, resets to pristine state)
    """
    P0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
    P1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
    p_k_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES - 1))  # Store only the next-state transition probabilities

    for m in range(NUM_MACHINES):
        transitions = generate_transition_probabilities()  # Get unique transition probabilities per machine
        p_k_all[m] = transitions[:, 0]  # Extract probability of transitioning to next state

        for i in range(NUM_DAMAGE_STATES - 1):
            P0_all[m, i, i + 1] = transitions[i, 0]  # Move to worse state
            P0_all[m, i, 0] = transitions[i, 1]  # Jump to pristine state

        P0_all[m, -1, 0] = 1  # Most damaged state resets to pristine
        P1_all[m, :, 0] = 1  # Every state transitions to pristine when repaired

    return P0_all, P1_all, p_k_all



def generate_reward_matrices(P0_all):
    """
    Generates reward matrices R0 (passive) and R1 (active) for all machines.
    Also generates the cost values `C_m` (repair cost) and `K_m` (breakdown cost).

    Parameters:
    - P0_all: Passive transition matrices (NUM_MACHINES x NUM_DAMAGE_STATES x NUM_DAMAGE_STATES)
    - D: Base cost parameter (varies across simulations)

    Returns:
    - R0_all: Passive reward matrices (NUM_MACHINES x NUM_DAMAGE_STATES)
    - R1_all: Active reward matrices (NUM_MACHINES x NUM_DAMAGE_STATES)
    - C_m_values: Repair costs for each machine (NUM_MACHINES,)
    - K_m_values: Breakdown costs for each machine (NUM_MACHINES,)
    """
    R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
    R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
    C_m_values = np.zeros(NUM_MACHINES)
    K_m_values = np.zeros(NUM_MACHINES)

    for m in range(NUM_MACHINES):
        # Generate repair and breakdown costs
        C_m = np.random.uniform(50, 75)  # Repair cost sampled from U(D, D+25)
        K_m = (m + 1) * 1.5 * C_m  # Varying breakdown cost as in the problem description

        C_m_values[m] = C_m
        K_m_values[m] = K_m

        # Compute reward matrices
        for k in range(NUM_DAMAGE_STATES):
            R0_all[m, k] = P0_all[m, k, 0] * (-K_m)  # Passive penalty when resetting to pristine
            R1_all[m, k] = -C_m  # Repair incurs fixed cost

    return R0_all, R1_all, C_m_values, K_m_values


def compute_G_H(p, beta):
    """ Computes G(k) and H(k) based on transition probabilities. """
    num_states = len(p)
    G = np.zeros(num_states + 1)
    H = np.zeros(num_states)
    
    for k in range(num_states):
        H[k] = beta**(k+1) * np.prod(p[:k])
        G[k+1] = G[k] + (1 - p[k]) * H[k]
    
    return G, H

def whittle_paper_discounted(K, C, p, beta=0.95):
    """ Computes the Whittle index for a machine. """
    p = np.append(p, 0)
    G, H = compute_G_H(p, beta)
    print("p : ",p)
    print("Length of  p : ", len(p))
    num_states = len(p)
    W = np.zeros(num_states)

    for k in range(num_states):
        #print(f" {k}th Iteration")
        numerator = (1 - G[k+1]) - p[k] * (1 - beta * G[k])
        denominator = (1 - G[k+1]) - beta * p[k] * (1 - G[k])
        W[k] = K * (numerator / denominator) - C

    return W

P0_all, P1_all, p_k_all = parameters_transition()
R0_all, R1_all, C_m_values, K_m_values = generate_reward_matrices(P0_all)


whittle_paper = []
whittle_package = []
for m in range(NUM_MACHINES):
        # Get machine-specific values
        p_k = p_k_all[m]       # Only transition probabilities needed for the paper
        P0, P1 = P0_all[m], P1_all[m]  # Full matrices needed for package
        R0, R1 = R0_all[m], R1_all[m]  # Full reward matrices
        K_m, C_m = K_m_values[m], C_m_values[m]

        # Compute Whittle indices using the paper's formula
        print(f"Machine {m} Whittle Index : {whittle_paper_discounted( K_m, C_m, p_k)}")
        whittle_paper.append(whittle_paper_discounted( K_m, C_m, p_k))

        # Compute Whittle indices using the package
        whittle_package.append(whittle_indices(P0, P1, R0, R1))

#############################################################################################3

# 5️⃣ Simulate state transitions
def state_transitions(machine_states, P0_all):
    new_states = np.zeros(NUM_MACHINES, dtype=int)
    for m in range(NUM_MACHINES):
        current_state = machine_states[m]
        transition_probs = P0_all[m, current_state]
        new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)
    return new_states

machine_states = np.zeros(NUM_MACHINES, dtype=int)
new_machine_states = state_transitions(machine_states, P0_all)
log_message(f"State Transitions: {machine_states} → {new_machine_states}")

# 6️⃣ Run full simulation
def simulate(D, a):
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
    return idleness_percentage

idleness = simulate(10, 0.5)
log_message(f"Idleness Percentage for D=10, a=0.5: {idleness:.3f}")

# 7️⃣ Run single repairman simulation
def simulate_single_repairman(D, a):
    idleness_count = 0
    all_machine_states = np.array(list(itertools.product(range(NUM_DAMAGE_STATES), repeat=NUM_MACHINES)))

    for state in all_machine_states:
        machine_states = state.copy()
        for _ in range(NUM_SIMULATIONS):
            P0_all, P1_all = initialize_transition_matrices()
            R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

            indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
            for m in range(NUM_MACHINES):
                indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

            machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]
            if all(index < 0 for index in machine_indices):
                idleness_count += 1

    idleness_percentage = idleness_count / (NUM_SIMULATIONS * len(all_machine_states))
    return idleness_percentage

single_repairman_idleness = simulate_single_repairman(10, 0.5)
log_message(f"Single Repairman Idleness Percentage for D=10, a=0.5: {single_repairman_idleness:.3f}")

# Close log file
log_file.close()

print("Simulation complete. Logs saved to 'simulations.txt'.")


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
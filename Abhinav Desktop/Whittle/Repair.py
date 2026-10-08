import numpy as np
import markovianbandit as bandit

# Constants
NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8
NUM_EPISODES = 100
NUM_PROBLEMS = 5120
DISCOUNT_FACTOR = 0.95


def whittle_indices(P0, P1, R0, R1):
  model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
  whittle_indices = model.whittle_indices(discount = DISCOUNT_FACTOR)
  return whittle_indices

# Log file path
# log_file = output_files

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


P0_all, P1_all = initialize_transition_matrices()
print("P0_all:", P0_all)
print("P1_all:", P1_all)



def initialize_reward_matrices(P0_all, D, a):
    """
    Initializes the reward matrices R0 (passive) and R1 (repair).
    """
    R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
    R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

    for m in range(NUM_MACHINES):
        Cm = np.random.uniform(D, D + 25)
        Km = a * Cm

        # Passive Reward Matrix (R0)
        for k in range(NUM_DAMAGE_STATES):
            R0_all[m, k] = P0_all[m, k, 0] * (-Km)  # Penalty when transitioning to pristine

        # Repair Reward Matrix (R1)
        R1_all[m, :] = -Cm  # Repair incurs fixed cost

    return R0_all, R1_all


D = 50
a = 0.2
R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)
print("R0_all:", R0_all)
print("R1_all:", R1_all)




## The changes we need to make is to compute Whittle index for each 512 (machine_states) and then repeat the process for 10 set of parameters.

def state_transitions(machine_states, P0_all):
    """
    Simulates state transitions for all machines using their transition probabilities.
    """
    new_states = np.zeros(NUM_MACHINES, dtype=int)

    for m in range(NUM_MACHINES):
        current_state = machine_states[m]
        transition_probs = P0_all[m, current_state]  # Use correct machine-specific transition matrix
        new_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=transition_probs)

    print(f"Machine {m}: State {current_state} → {new_states[m]} (probabilities: {transition_probs})")

    return new_states


# Define initial states for machines (randomly chosen)
machine_states = np.array([0, 1, 2])
new_states = state_transitions(machine_states, P0_all)
print("Old States:", machine_states)
print("New States:", new_states)



#####     Idleness Criteria : If in the entire episode the repairman is idle for even once, that particular problem instance is taken to be idle.

def simulate(D, a):
    """
    Runs the simulation for a given (D, a) pair and returns the idleness percentage.
    """
    idle_count_episodic = 0  # Track episodes where repairmen are idle

    for _ in range(100):        ################################# CHANGE THIS TO NUM_PROBLEMS
    # Generate unique transition and reward matrices for each machine
      P0_all, P1_all = initialize_transition_matrices()
      R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

      # Compute Whittle indices for each machine
      indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
      for m in range(NUM_MACHINES):
          indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

      # Initialize machine states
      machine_states = np.zeros(NUM_MACHINES, dtype=int)

      # Track if this problem instance had any idle repair step
      problem_was_idle = False

      # Simulate NUM_EPISODES time steps for this problem instance
      for _ in range(NUM_EPISODES):
          machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

          # If all Whittle indices are negative at any time step, mark the problem as idle
          if all(index < 0 for index in machine_indices):
              problem_was_idle = True  # Mark this problem as having at least one idle step

          else:
              # Repair machine with highest index
              repair_target = np.argmax(machine_indices)
              machine_states[repair_target] = 0  # Reset to pristine state

          # Apply state transitions
          machine_states = state_transitions(machine_states, P0_all)

      #  If the repairman was idle at least once in this problem, count it as an idle episode
      if problem_was_idle:
          idle_count_episodic += 1

  # Normalize idleness percentage
    return idle_count_episodic / NUM_PROBLEMS



idleness_percentage = {}

D_values = [0, 10, 20, 25, 30, 40, 50]
a_values = [1.5, 2.0, 3.0, 4.0, 5.0]

for D in D_values:
    for a in a_values:
        print(f"Running simulation for D = {D}, a = {a}")
        idleness_percentage[(D, a)] = simulate(D, a)







def simulate_average(D, a):
    """
    Runs the simulation for a given (D, a) pair and returns the idleness percentage.
    """
    idle_count_episodic = 0  # Track episodes where repairmen are idle

    for _ in range(NUM_PROBLEMS):
    # Generate unique transition and reward matrices for each machine
      P0_all, P1_all = initialize_transition_matrices()
      R0_all, R1_all = initialize_reward_matrices(P0_all, D, a)

      # Compute Whittle indices for each machine
      indices_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
      for m in range(NUM_MACHINES):
          indices_all[m] = whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m])

      # Initialize machine states
      machine_states = np.zeros(NUM_MACHINES, dtype=int)

      # Track if this problem instance had any idle repair step
      idle_count_per_episode = 0

      # Simulate NUM_EPISODES time steps for this problem instance
      for _ in range(NUM_EPISODES):
          machine_indices = [indices_all[m, state] for m, state in enumerate(machine_states)]

          # If all Whittle indices are negative at any time step, mark the problem as idle
          if all(index < 0 for index in machine_indices):
              idle_count_per_episode += 1   # Mark this problem as having at least one idle step

          else:
              # Repair machine with highest index
              repair_target = np.argmax(machine_indices)
              machine_states[repair_target] = 0  # Reset to pristine state

          # Apply state transitions
          machine_states = state_transitions(machine_states, P0_all)

      #  If the repairman was idle at least once in this problem, count it as an idle episode
      if idle_count_per_episode > 0:
          idle_count_episodic += idle_count_per_episode / NUM_EPISODES

  # Normalize idleness percentage
    return idle_count_episodic / NUM_PROBLEMS



idleness_percentage = {}

D_values = [0, 10, 20, 25, 30, 40, 50]
a_values = [1.5, 2.0, 3.0, 4.0, 5.0]

for D in D_values:
    for a in a_values:
        print(f"Running simulation for D = {D}, a = {a}")
        idleness_percentage[(D, a)] = simulate(D, a)





def compute_T_k(p_k):
    """
    Compute expected time to reach state k from the pristine state for a single machine.

    Parameters:
    - p_k: Transition probabilities for a single machine (shape: (NUM_DAMAGE_STATES - 1,))

    Returns:
    - T_k: Expected time to reach state k (shape: (NUM_DAMAGE_STATES - 1,))
    """
    T_k = np.zeros(NUM_DAMAGE_STATES - 1)  # Initialize expected time array

    for k in range(1, NUM_DAMAGE_STATES - 1):  # Iterate over damage states
        for j in range(k):  # Compute time based on previous states
            T_k[k] += p_k[j] * T_k[j]  # Recursive accumulation
        T_k[k] += 1  # Add base step (accounts for at least one step to reach k)

    return T_k



def compute_whittle_indices_paper(p_k, K_m, C_m):
    """
    Computes Whittle indices for a single machine using the formula from Corollary 2.

    Parameters:
    - p_k: Transition probabilities for a single machine (shape: (NUM_DAMAGE_STATES - 1,))
    - K_m: Breakdown cost for the machine
    - C_m: Repair cost for the machine

    Returns:
    - W_m_k: Whittle indices (shape: (NUM_DAMAGE_STATES - 1,))
    """
    T_k = compute_T_k(p_k)  # Compute T(k)
    W_m_k = np.zeros(NUM_DAMAGE_STATES - 1)  # Store Whittle indices

    for k in range(NUM_DAMAGE_STATES - 1):
        term = (T_k[k] - p_k[k] * T_k[k - 1]) if k > 0 else T_k[k]
        W_m_k[k] = K_m * (1 - p_k[k] * term) - C_m

    return W_m_k



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




P0_all, P1_all, p_k_all = parameters_transition()
R0_all, R1_all, C_m_values, K_m_values = generate_reward_matrices(P0_all)



### Whittle Index Computation Based on Paper


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


print("Whittle Paper:", whittle_paper)
print("Whittle Packages:", whittle_package)
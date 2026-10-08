import numpy as np
import heapq

# Constants
NUM_MACHINES = 3  # Number of machines
NUM_DAMAGE_STATES = 8  # Number of discrete damage states
DISCOUNT_RATE = 0.95  # Discount factor
NUM_ACTIVE_SLOTS = 1  # Number of machines that can be repaired at a time
TIME_STEPS = 50  # Simulation duration

# Generate random transition probabilities for deterioration
transition = np.random.uniform(0, 1, NUM_DAMAGE_STATES)
transition = -np.sort(-transition)  # Sorting in descending order

# # Transition matrices
# P0 = np.zeros((NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))  # Passive (no repair)
# P1 = np.zeros((NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))  # Active (repair)


P0 = [[transition[0], 1 - transition[0], 0, 0, 0, 0, 0, 0],
      [transition[1], 0, 1 - transition[1], 0, 0, 0, 0, 0],
      [transition[2], 0, 0, 1 - transition[2], 0, 0, 0, 0],
      [transition[3], 0, 0, 0, 1 - transition[3], 0, 0, 0],
      [transition[4], 0, 0, 0, 0, 1 - transition[4], 0, 0],
      [transition[5], 0, 0, 0, 0, 0, 1 - transition[5], 0],
      [transition[6], 0, 0, 0, 0, 0, 0, 1 - transition[6]],
      [1, 0, 0, 0, 0, 0, 0, 0]       # For state 7: It becomes an absorbing state (permanent failure) # P0[-1, -1] = 1  # Last state is absorbing (catastrophic failure)
      ]
P0 = np.array(P0)  # Convert to NumPy array
print(f"P0: {P0}")

P0 = P0 / P0.sum(axis=1, keepdims=True)


P1 = [[1, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0]]
P1 = np.array(P1)  # Convert to NumPy array
print(f"P1: {P1}")

P1 = P1 / P1.sum(axis=1, keepdims=True)


# # Fill transition matrices
# for k in range(NUM_DAMAGE_STATES - 1):
#     P0[k, k] = transition[k]  # Probability of staying
#     P0[k, k + 1] = 1 - transition[k]  # Probability of worsening damage

# P0[-1, -1] = 1  # Last state is absorbing (catastrophic failure)

# for k in range(NUM_DAMAGE_STATES):
#     P1[k, 0] = 1  # Repair always resets to state 0

# Generate random costs for each machine
repair_costs = np.random.uniform(10, 30, NUM_MACHINES)  # Random repair costs
breakdown_penalty = repair_costs * np.random.uniform(2, 5, NUM_MACHINES)  # Higher penalty for breakdown

# Compute Whittle Index (Simplified for now)
def compute_whittle_index(machine, state):
    """
    Computes Whittle index for a given machine at a given state.
    """
    p_k = transition[state]  # Probability of staying in current state
    expected_future_cost = DISCOUNT_RATE * np.sum(P0[state] * repair_costs[machine])
    
    index = (breakdown_penalty[machine] - expected_future_cost) / (1 - DISCOUNT_RATE * p_k)
    return index

# Initialize machine states (random damage levels)
machine_states = np.random.randint(0, NUM_DAMAGE_STATES, NUM_MACHINES)





# Simulation Loop
for t in range(TIME_STEPS):
    whittle_indices = []

    # Compute Whittle index for each machine
    for m in range(NUM_MACHINES):
        index = compute_whittle_index(m, machine_states[m])
        whittle_indices.append((index, m))  # Store index with machine ID

    # Select top machines for activation (repair)
    active_machines = [m for _, m in heapq.nlargest(NUM_ACTIVE_SLOTS, whittle_indices)]

# Ensure transition probabilities sum to 1
P0 = P0 / P0.sum(axis=1, keepdims=True)

# Update machine states based on actions
for m in range(NUM_MACHINES):
    if m in active_machines:
        machine_states[m] = 0  # Reset to healthy state
    else:
        next_state_probs = P0[machine_states[m]]
        next_state_probs /= next_state_probs.sum()  # Normalize to sum to 1
        machine_states[m] = np.random.choice(range(NUM_DAMAGE_STATES), p=next_state_probs)

    # Print status
    print(f"Time {t}: Machine states: {machine_states}, Active machines: {active_machines}")

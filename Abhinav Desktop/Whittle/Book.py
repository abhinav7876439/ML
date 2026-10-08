import numpy as np

NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8  # Assuming 3 possible states per machine
NUM_EPISODES = 100  # Time steps for each simulation
NUM_PROBLEMS = 512  # Number of state combinations (8^3)
NUM_SIMULATIONS = 100  # Number of transition matrices to simulate

def generate_machine_states():
    """
    Generates all possible machine states combinations for NUM_MACHINES machines.
    Each machine has NUM_DAMAGE_STATES possible states, resulting in NUM_PROBLEMS combinations.
    """
    grid = np.meshgrid(*[range(NUM_DAMAGE_STATES)] * NUM_MACHINES)

    # Stack the grid arrays together and reshape to get all combinations as rows
    states = np.vstack([g.flatten() for g in grid]).T
    return states




def value_iteration(P0_all, R0_all, R1_all, gamma=0.95, tol=1e-6):
    """
    Solves the MDP using Value Iteration to find the optimal expected cost.
    
    Parameters:
    - P0_all: Transition probability matrix for no repair.
    - R0_all: Reward matrix for passive operation.
    - R1_all: Reward matrix for repair.
    - gamma: Discount factor (default 0.95).
    - tol: Convergence tolerance (default 1e-6).

    Returns:
    - V: Optimal cost-to-go values for each state.
    - policy: Optimal action policy for each state.
    """
    
    NUM_MACHINES = P0_all.shape[0]
    NUM_DAMAGE_STATES = P0_all.shape[1]

    # Initialize value function V(s) to zero for all states
    V = np.zeros(NUM_DAMAGE_STATES)

    while True:
        V_new = np.zeros_like(V)

        for s in range(NUM_DAMAGE_STATES):
            # Cost for doing nothing (passive cost)
            cost_no_repair = R0_all[0, s] + gamma * np.sum(P0_all[0, s] * V)

            # Cost for repairing (active repair cost)
            cost_repair = R1_all[0, s] + gamma * V[0]  # Repair always resets to pristine

            # Choose the action that minimizes cost
            V_new[s] = min(cost_no_repair, cost_repair)

        # Check convergence
        if np.max(np.abs(V - V_new)) < tol:
            break
        V = V_new

    # Compute optimal policy π*(s)
    policy = np.zeros(NUM_DAMAGE_STATES, dtype=int)

    for s in range(NUM_DAMAGE_STATES):
        cost_no_repair = R0_all[0, s] + gamma * np.sum(P0_all[0, s] * V)
        cost_repair = R1_all[0, s] + gamma * V[0]

        policy[s] = 0 if cost_no_repair < cost_repair else 1  # Choose the best action

    return V, policy

# Example Usage
NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8
DISCOUNT_FACTOR = 0.95

# Example Transition and Reward Matrices
P0_all = np.random.rand(NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES)
P0_all /= P0_all.sum(axis=2, keepdims=True)  # Normalize to make it a valid probability matrix
R0_all = -np.random.randint(1, 10, (NUM_MACHINES, NUM_DAMAGE_STATES))  # Example passive costs
R1_all = -np.random.randint(20, 50, (NUM_MACHINES, NUM_DAMAGE_STATES))  # Example repair costs

V_optimal, optimal_policy = value_iteration(P0_all, R0_all, R1_all, DISCOUNT_FACTOR)

print("Optimal Cost-to-Go Values:", V_optimal)
print("Optimal Policy (0=No Repair, 1=Repair):", optimal_policy)

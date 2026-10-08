import numpy as np
import markovianbandit as bandit
import matplotlib.pyplot as plt

# Constants
NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8
NUM_PROBLEMS = 5120
NUM_EPISODES = 50
DISCOUNT_FACTOR = 0.95

D_values = [0, 10]
a_values = [1.5, 2.0]

# Logging
log_file = open("cost_suboptimality_log.txt", "w")

def log_message(message):
    """Writes a message to the log file and prints it."""
    print(message)
    log_file.write(message + "\n")

def initialize_matrices(D, a):
    """Initialize transition and reward matrices."""
    P0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
    P1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
    R0_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))
    R1_all = np.zeros((NUM_MACHINES, NUM_DAMAGE_STATES))

    for m in range(NUM_MACHINES):
        p_values = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
        p_values = np.append(p_values, 0)  # Ensure last state transition probability is zero
        for i in range(NUM_DAMAGE_STATES - 1):
            P0_all[m, i, i + 1] = p_values[i]
            P0_all[m, i, 0] = 1 - p_values[i]

        P0_all[m, -1, 0] = 1
        P1_all[m, :, 0] = 1

        Cm = np.random.uniform(D, D + 25)
        Km = a * Cm
        for k in range(NUM_DAMAGE_STATES):
            R0_all[m, k] = P0_all[m, k, 0] * (-Km)
        R1_all[m, :] = -Cm

    return P0_all, P1_all, R0_all, R1_all

def whittle_indices(P0, P1, R0, R1):
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    return model.whittle_indices(discount=DISCOUNT_FACTOR)

def compute_optimal_cost(P0, R0):
    """Use Dynamic Programming to compute the optimal cost."""
    NUM_DAMAGE_STATES = len(R0)
    V = np.zeros(NUM_DAMAGE_STATES)  # Ensure V has correct dimensions

    for _ in range(100):  # Iterative DP
        V_new = np.min(R0[:, None] + DISCOUNT_FACTOR * (P0 @ V), axis=1)  # Ensure matrix dimensions match
        if np.max(np.abs(V - V_new)) < 1e-6:
            break
        V = V_new

    return np.sum(V)


def compute_cost_suboptimality(D, a):
    """Compute percentage cost suboptimality for given D and a."""
    cost_suboptimalities = []
    idleness_percentages = []

    for _ in range(NUM_PROBLEMS):
        P0_all, P1_all, R0_all, R1_all = initialize_matrices(D, a)
        indices_all = np.array([whittle_indices(P0_all[m], P1_all[m], R0_all[m], R1_all[m]) for m in range(NUM_MACHINES)])

        # Compute optimal and index policy costs
        optimal_cost = sum(compute_optimal_cost(P0_all[m], R0_all[m]) for m in range(NUM_MACHINES))
        index_policy_cost = sum(np.min(R0_all[m] + DISCOUNT_FACTOR * (P0_all[m] @ indices_all[m])) for m in range(NUM_MACHINES))

        suboptimality = 100 * (index_policy_cost - optimal_cost) / optimal_cost
        cost_suboptimalities.append(suboptimality)

        # Check idleness percentage
        idleness_percentages.append(any(index < 0 for index in indices_all.flatten()))

    # Compute statistics
    lower_quartile, median, upper_quartile = np.percentile(cost_suboptimalities, [25, 50, 75])
    idle_percentage = np.mean(idleness_percentages) * 100

    log_message(f"D={D}, a={a}: Suboptimality (Q1={lower_quartile:.3f}, Median={median:.3f}, Q3={upper_quartile:.3f}), Idleness={idle_percentage:.2f}%")
    
    return (lower_quartile, median, upper_quartile, idle_percentage)

# Compute for all (D, a) pairs
results = {}
for D in D_values:
    results[D] = {}
    for a in a_values:
        results[D][a] = compute_cost_suboptimality(D, a)

# Close log file
log_file.close()

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

D_labels = [str(D) for D in D_values]
x = np.arange(len(D_values))

for i, a in enumerate(a_values[:-1]):  
    y = [results[D][a][1] for D in D_values]
    ax.plot(x, y, marker='o', label=f"a={a}")

ax.set_xticks(x)
ax.set_xticklabels(D_labels)
ax.set_xlabel("D Value (Cost)")
ax.set_ylabel("Percentage Cost Suboptimality (Median)")
ax.set_title("Percentage Cost Suboptimality vs. Cost (D) for Different a values")
ax.legend()
ax.grid(True)

plt.show()

# import markovianbandit as bandit
# import numpy as np
# import pandas as pd


# for i in range(5120):
#   prob1 = np.random.uniform(0, 1, 7)
#   prob1 /= np.sum(prob1)
#   prob1 = np.sort(-prob1)
#   P1 = [[1 - prob1[0], prob1[0], 0, 0, 0, 0, 0, 0],
#         [1 - prob1[1], 0, prob1[1], 0, 0, 0, 0, 0],
#         [1 - prob1[2], 0, 0, prob1[2], 0, 0, 0, 0],
#         [1 - prob1[3], 0, 0, 0, prob1[3], 0, 0, 0],
#         [1 - prob1[4], 0, 0, 0, 0, prob1[4], 0, 0],
#         [1 - prob1[5], 0, 0, 0, 0, 0, prob1[5], 0],
#         [1 - prob1[6], 0, 0, 0, 0, 0, 0, prob1[6]],
#         [1, 0, 0, 0, 0, 0, 0, 0]]
#   Cm =


# model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
# print(model.is_indexable())
# print(model.whittle_indices())



# # Constants

# NUM_PROBLEMS = 5120
# NUM_CHOICES = 35

# results = []

# # Function to generate transition probabilities for a machine
# def generate_transition_probabilities():
#     p_sample = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
#     print("p_sample : ",p_sample)
#     return np.append(p_sample, 0)

# def whittle_indices(P0, P1, R0, R1):
#   model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#   whittle_indices = model.whittle_indices()
#   return whittle_indices

# # Function to compute policy performance (placeholder, replace with actual index policy)
# def evaluate_index_policy(machine_costs, repair_penalties):
#   pass

# def order_states(indices):
#   return np.argsort(-indices)

# columns = ["D", "a", "Lower Quartile", "Median", "Upper Quartile", "Idle Percentage"]
# df_results = pd.DataFrame(results, columns=columns)

# df_results.to_csv("index_policy_results.csv", index=False)
# print(df_results.head())

# def whittle_indices(P0, P1, R0, R1):
#   model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#   whittle_indices = model.whittle_indices()
#   return whittle_indices


import markovianbandit as bandit
import numpy as np


NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8  # states (0: pristine, 1-6: damage, 7: breakdown)
D_values = [0, 10, 20, 25, 30, 40, 50]
a_values = [1.5, 2.0, 3.0, 4.0, 5.0]
DISCOUNT_RATE = 0.95


def whittle_indices(P0, P1, R0, R1):
  model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
  whittle_indices = model.whittle_indices()
  return whittle_indices


def compute_whittle_indices(P, K, C):
    num_states = len(P)
    T = np.zeros(num_states)

    # Compute expected duration T(k)
    for k in range(1, num_states):
        T[k] = sum(np.prod(list(P[j, j+1] for j in range(l+1) if j+1 < num_states)) for l in range(k))

    # Compute Whittle indices
    W = []
    for k in range(num_states):
        if k == 0:
            W.append(K - C)
        else:
            h = 1  # Normalization factor (adjust if needed)
            
            # Ensure index k+1 is within bounds
            if k + 1 < num_states:
                fraction_term = T[k] - P[k, k+1] * T[k - 1]
                W.append(K * (1 - P[k, k+1] * (1 / fraction_term)) - C)
            else:
                W.append(K - C)  # Handle last state differently

    return W




def generate_transition_probabilities():
    p_sample = np.sort(np.random.uniform(0, 1, NUM_DAMAGE_STATES - 1))[::-1]
    p_breakdown = 1 - p_sample  # Breakdown probability
    return np.vstack((p_sample, p_breakdown)).T

# Initialize transition matrices
P0 = np.zeros((NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))
P1 = np.zeros((NUM_DAMAGE_STATES, NUM_DAMAGE_STATES))

# Generate transition probabilities
transitions = generate_transition_probabilities()

# Fill P0 (No repair scenario)
for i in range(NUM_DAMAGE_STATES - 1):  # Exclude breakdown state
    if i < NUM_DAMAGE_STATES - 2:  # Ensure we don’t go out of bounds
        P0[i, i + 1] = transitions[i, 0]  # Move to worse state
        P0[i, -1] = transitions[i, 1]  # Jump to breakdown
    else:
        P0[i, -1] = 1  # Last state (breakdown) resets to pristine

# Breakdown state transitions to pristine state
P0[-1, 0] = 1  # After breakdown, return to pristine state

# Fill P1 (With repair scenario) - Always goes back to pristine state
P1[:, 0] = 1  # Every state transitions to pristine when repaired

for D in D_values:
    Cm = np.random.uniform(D, D + 25)
    for a in a_values:
        # Generate machine-specific costs and penalties
        Km = a * Cm

        # Reward Matrices
        R0 = np.zeros(NUM_DAMAGE_STATES)
        R1 = np.full(NUM_DAMAGE_STATES, -Cm)

        # Assign breakdown cost in R0
        R0[-1] = -Km  # Breakdown incurs high penalty

        # Print results
        print("P0 (No Repair) Transition Matrix:\n", P0)
        print("P1 (Repair) Transition Matrix:\n", P1)
        print("Reward for Passive Action (R0):", R0)
        print("Reward for Repair Action (R1):", R1)
        print("Cm : ",Cm)
        print("Alpha : ",a)

        # Calculate and print whittle indices
        indices = whittle_indices(P0, P1, R0, R1)
        closed_indices = compute_whittle_indices(P0, Km, Cm)
        print("Whittle Indices:", indices)

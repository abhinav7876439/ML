

import numpy as np

# Parameters
max_inventory = 15
inventory_levels = range(max_inventory + 1)
actions = [0, 1]
a_prev_values = [0, 1]
S = len(inventory_levels) * len(a_prev_values)
t_end = 5000

# Cost parameters
h = 1
p = 2
l = 5
gamma = 0.1
lam = 0.6
mu = 0.8
switch_cost = 3
dt = 1

def w(i):
    return 0


def C_switch(i):
    return switch_cost + w(i)

def C(i, a_prev, a):
    switch_penalty = C_switch(i) if a != a_prev else 0
    return h * i + p * gamma * i + l * lam * (i == 0) + switch_penalty


# State space: (inventory, previous action)
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]
state_to_idx = {s: idx for idx, s in enumerate(states)}


# Simulate environment
def next_state(i, a_prev, a):
    # --- Production ---
    if a == 1:
        produce = np.random.exponential(1/mu * dt)
        produced = int(produce)
    else:
        produced = 0
    # --- Demand arrivals ---
    demand = np.random.poisson(lam)
    # --- Perishability ---
    perish = np.random.binomial(i, gamma)
    # --- Inventory update ---
    next_i = max(0, min(i + produced - demand - perish, max_inventory))
    next_a_prev = a
    reward = -C(next_i, next_a_prev, a)
    return (next_i, next_a_prev), reward

# Step sizes
alpha = lambda t: 1.0 / (t + 1)
beta = lambda t: 1.0 / (10 * (t + 1))

# Initialize Q and Lambda
Q_values = np.zeros((S, 2))
Lambda_values = np.zeros(S)
Lambda_history = np.zeros((S, t_end + 1))

# Initial state
curr_state = (0, 0)
curr_idx = state_to_idx[curr_state]

for t in range(1, t_end + 1):
    # Epsilon-greedy: random action with small probability
    if np.random.rand() < 0.1:
        a = np.random.choice(actions)
    else:
        a = np.argmax(Q_values[curr_idx])

    # Simulate environment
    next_state_tuple, reward = next_state(curr_state[0], curr_state[1], a)
    next_idx = state_to_idx[next_state_tuple]

    # Fast time-scale Q update
    alpha_step = alpha(t)
    next_Q = np.max(Q_values[next_idx])
    Q_values[curr_idx, a] = (1 - alpha_step) * Q_values[curr_idx, a] + alpha_step * (reward + Lambda_values[curr_idx] + next_Q)

    # Slow time-scale Lambda update
    beta_step = beta(t)
    Lambda_values[curr_idx] += beta_step * (Q_values[curr_idx, 1] - Q_values[curr_idx, 0])
    Lambda_history[curr_idx, t] = Lambda_values[curr_idx]

    # Move to next state
    curr_state = next_state_tuple
    curr_idx = next_idx

print("Learned Lambda history (Whittle index approximation):")
print(Lambda_history[:, ::500])  # Print every 500 steps for brevity
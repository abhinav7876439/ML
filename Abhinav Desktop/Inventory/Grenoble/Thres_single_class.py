
######################


import numpy as np

# Parameters
max_inventory = 10
inventory_levels = range(max_inventory + 1)
actions = [0, 1]
a_prev_values = [0, 1]
reference_state = (0, 1)

# Cost parameters
mu = 0.8         # production rate
lam = 0.6        # demand rate
gamma = 0.1      # perishability rate
holding_cost = 1
perish_penalty = 2
lost_sales_cost = 5
switch_cost = 3

# State space: (inventory, previous action)
states = [(i1, a_prev) for i1 in inventory_levels for a_prev in a_prev_values]

# Initialize relative value function
h_val = {s: 0.0 for s in states}
policy = {}
rho = 0.0
dt = 1.0

def next_state(x, a):
    """
    Simulates one-step transition in integer-valued inventory dynamics.
    Parameters:
        x        : current inventory level (integer)
        a        : action (0 = passive, 1 = active)
    Returns:
        x_next   : next inventory level (integer)
    """
    # --- Production ---
    if a == 1:
        produce = np.random.exponential(1/mu * dt)
        produced = int(produce)
    else:
        produced = 0

    # --- Demand arrivals ---
    demand = np.random.poisson(lam * dt)
    
    # --- Perishability ---
    perish_prob = 1 - np.exp(-gamma * dt)
    perished = np.random.binomial(x, perish_prob)

    # --- Inventory update ---
    x_next = x + produced - demand - perished
    x_next = max(0, min(int(round(x_next)), max_inventory))  # enforce boundaries and integer

    return x_next

# Simulation example
x = 0  # initial inventory
for t in range(13):
    a = np.random.choice(actions)
    x = next_state(x, a)
    print(f"t={t}, a={a}, inventory={x}")

# Cost function
def c_hat(i1, a_prev, a_curr):
    holding = holding_cost * i1
    perish = perish_penalty * gamma * i1
    lost_sales = lost_sales_cost * (lam * (i1 == 0))
    switch = switch_cost if a_curr != a_prev else 0
    return holding + perish + lost_sales + switch

# Relative value iteration
for iteration in range(1000):
    h_new = {}
    for s in states:
        costs = []
        for a in actions:
            s_next_inv = next_state(s[0], a)
            s_next = (s_next_inv, a)
            costs.append(c_hat(s[0], s[1], a) + h_val[s_next])
        h_new[s] = min(costs)
    # Normalize using reference state
    g = min(c_hat(reference_state[0], reference_state[1], a) + h_val[(next_state(reference_state[0], a), a)] for a in actions)
    for s in states:
        h_new[s] -= g
    rho = g
    delta = max(abs(h_new[s] - h_val[s]) for s in states)
    h_val = h_new
    if delta < 1e-4:
        break

# Extract policy
for s in states:
    costs = []
    for a in actions:
        s_next_inv = next_state(s[0], a)
        s_next = (s_next_inv, a)
        costs.append(c_hat(s[0], s[1], a) + h_val[s_next])
    policy[s] = actions[np.argmin(costs)]

print("\nOptimal policy (state: action):")
for s in states:
    print(f"{s}: {policy[s]}")
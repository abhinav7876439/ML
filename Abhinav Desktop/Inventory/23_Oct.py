import numpy as np

# Parameters
max_inventory = 15
inventory_levels = range(max_inventory + 1)
actions = [0, 1]  # 0 = passive, 1 = active
a_prev_values = [0, 1]
reference_state = (0, 1)

# Cost parameters
h = 1              # holding cost
p = 2              # perish penalty
l = 5              # lost sales cost
gamma = 0.1        # perishability rate
lam = 0.6          # demand rate
mu = 0.8           # production rate
switch_cost = 3    # switching cost

# Penalty function (optional, can modify if needed)
def w(i):
    return 0

# Base cost function C(i, a_prev)
def C(i, a_prev):
    return h * i + p * gamma * i + l * lam * (i == 0) + w(i) * (a_prev == 1)

def C_switch(i):
    return switch_cost

# State space (inventory, previous action)
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]

# Initialize value function and policy
V = {s: 0.0 for s in states}
policy = {}
rho = 0.0

# ------------------------------
# Q-function as per your equations
# ------------------------------
def Q(i, a_prev, a_t, V):
    # Case 1: Passive (a_t = 0)
    if a_t == 0:
        if i > 0:
            return (C(i, a_prev)
                    + (i * gamma + lam) * V[(i - 1, 0)]
                    + (1 - (i * gamma + lam)) * V[(i, 0)])
        else:
            return C(i, a_prev) + V[(i, 0)]

    # Case 2: Active (a_t = 1)
    else:
        # Handle switching cost rule: Q(i, 0, 1) = C_switch + Q(i, 1, 1)
        if a_prev == 0:
            return C_switch(i) + Q(i, 1, 1, V)

        # a_prev == 1 case
        if i > 0:
            return (C(i, 1)
                    + mu * V[(min(i + 1, max_inventory), 0)]
                    + (i * gamma + lam) * V[(i - 1, 0)]
                    + (1 - (i * gamma + lam + mu)) * V[(i, 0)])
        else:
            return C(i, 1) + V[(i, 1)]

# ------------------------------
# Relative Value Iteration
# ------------------------------
for iteration in range(1000):
    V_new = {}
    for s in states:
        i, a_prev = s
        Q0 = Q(i, a_prev, 0, V)
        Q1 = Q(i, a_prev, 1, V)
        V_new[s] = max(Q0, Q1)
    
    # Reference normalization (to fix relative scale)
    g = max(Q(reference_state[0], reference_state[1], a, V) for a in actions)
    for s in states:
        V_new[s] -= g
    
    rho = g
    delta = max(abs(V_new[s] - V[s]) for s in states)
    V = V_new
    if delta < 1e-5:
        print(f"Converged at iteration {iteration}")
        break

# ------------------------------
# Extract Optimal Policy
# ------------------------------
for s in states:
    i, a_prev = s
    Q0 = Q(i, a_prev, 0, V)
    Q1 = Q(i, a_prev, 1, V)
    policy[s] = actions[np.argmax([Q0, Q1])]

# ------------------------------
# Display Results
# ------------------------------
print("\nOptimal policy (state: action):")
for s in states:
    print(f"{s}: {policy[s]}")




##############################################################



import numpy as np

# -----------------------------
# Parameters
# -----------------------------
max_inventory = 15
inventory_levels = range(max_inventory + 1)
actions = [0, 1]            # 0 = idle, 1 = produce
a_prev_values = [0, 1]
reference_state = (0, 1)

# Cost parameters
h = 1.0
p = 2.0
l = 5.0
gamma = 0.1
lam = 0.6
mu = 0.8
switch_cost = 3.0

# -----------------------------
# Cost and Transition Functions
# -----------------------------
def C(i, a_prev, a_t):
    """Average cost with switching penalty."""
    base_cost = h * i + p * gamma * i + l * lam * (i == 0)
    switch_penalty = switch_cost if a_prev != a_t else 0
    return base_cost + switch_penalty

# State space (inventory, prev_action)
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]

# Initialize value and gain
V = {s: 0.0 for s in states}
rho = 0.0

# -----------------------------
# Bellman Q-function
# -----------------------------
def Q(i, a_prev, a_t, V):
    cost = C(i, a_prev, a_t)
    if a_t == 0:  # idle
        if i > 0:
            prob_down = min(1.0, (i * gamma + lam) / (1 + i * gamma + lam))
            return cost + prob_down * V[(i - 1, 0)] + (1 - prob_down) * V[(i, 0)]
        else:
            return cost + V[(i, 0)]
    else:  # produce
        prob_up = mu / (1 + mu + i * gamma + lam)
        prob_down = (i * gamma + lam) / (1 + mu + i * gamma + lam)
        prob_stay = 1 - prob_up - prob_down
        return cost + prob_up * V[(min(i + 1, max_inventory), 1)] + prob_down * V[(max(i - 1, 0), 1)] + prob_stay * V[(i, 1)]

# -----------------------------
# Relative Value Iteration (RVI)
# -----------------------------
for iteration in range(2000):
    V_new = {}
    for s in states:
        i, a_prev = s
        Q0 = Q(i, a_prev, 0, V)
        Q1 = Q(i, a_prev, 1, V)
        V_new[s] = min(Q0, Q1)  # minimizing cost
    # Gain normalization (average cost)
    g = V_new[reference_state]
    for s in states:
        V_new[s] -= g
    rho = g
    delta = max(abs(V_new[s] - V[s]) for s in states)
    V = V_new
    if delta < 1e-5:
        break

# -----------------------------
# Extract optimal policy
# -----------------------------
policy = {}
for s in states:
    i, a_prev = s
    Q0 = Q(i, a_prev, 0, V)
    Q1 = Q(i, a_prev, 1, V)
    policy[s] = np.argmin([Q0, Q1])

print(f"\nConverged in {iteration} iterations, ρ ≈ {rho:.4f}")
print("\nOptimal policy (state → action):")
for s in states:
    print(f"{s}: {policy[s]}")


###############################################3




import numpy as np

# -----------------------------
# Parameters
# -----------------------------
max_inventory = 15
inventory_levels = range(max_inventory + 1)
actions = [0, 1]  # 0 = idle, 1 = produce
a_prev_values = [0, 1]
reference_state = (0, 1)

# Cost parameters
h = 1.0           # holding cost
p = 2.0           # perish penalty
l = 5.0           # lost sales cost
gamma = 0.1       # perishability rate
lam = 0.6         # demand rate
mu = 0.8          # production rate
switch_cost = 3.0 # switching cost

# -----------------------------
# Penalty function w(i)
# -----------------------------
def w(i):
    """Index-based or workload penalty."""
    return 0.2 * i  # Example: linear penalty; can be nonlinear (e.g., i^2)

# -----------------------------
# Immediate cost function
# -----------------------------
def C(i, a_prev, a_t):
    base_cost = h * i + p * gamma * i + l * lam * (i == 0) + w(i) * (a_prev == 1)
    switch_penalty = switch_cost if a_prev != a_t else 0
    return base_cost + switch_penalty

# -----------------------------
# State space and initialization
# -----------------------------
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]
V = {s: 0.0 for s in states}
rho = 0.0

# -----------------------------
# Transition dynamics
# -----------------------------
def Q(i, a_prev, a_t, V):
    cost = C(i, a_prev, a_t)

    if a_t == 0:  # Idle action
        if i > 0:
            prob_down = (i * gamma + lam) / (1 + i * gamma + lam)
            return cost + prob_down * V[(i - 1, 0)] + (1 - prob_down) * V[(i, 0)]
        else:
            return cost + V[(i, 0)]

    else:  # Produce
        prob_up = mu / (1 + mu + i * gamma + lam)
        prob_down = (i * gamma + lam) / (1 + mu + i * gamma + lam)
        prob_stay = 1 - prob_up - prob_down
        return (cost
                + prob_up * V[(min(i + 1, max_inventory), 1)]
                + prob_down * V[(max(i - 1, 0), 1)]
                + prob_stay * V[(i, 1)])

# -----------------------------
# Relative Value Iteration (RVI)
# -----------------------------
for iteration in range(2000):
    V_new = {}
    for s in states:
        i, a_prev = s
        Q0 = Q(i, a_prev, 0, V)
        Q1 = Q(i, a_prev, 1, V)
        V_new[s] = min(Q0, Q1)  # minimizing total cost
    # Normalize with respect to reference state
    g = V_new[reference_state]
    for s in states:
        V_new[s] -= g
    rho = g
    delta = max(abs(V_new[s] - V[s]) for s in states)
    V = V_new
    if delta < 1e-6:
        break

# -----------------------------
# Extract optimal policy
# -----------------------------
policy = {}
for s in states:
    i, a_prev = s
    Q0 = Q(i, a_prev, 0, V)
    Q1 = Q(i, a_prev, 1, V)
    policy[s] = np.argmin([Q0, Q1])

print(f"\nConverged in {iteration} iterations, ρ ≈ {rho:.6f}")
print("\nOptimal policy (state → action):")
for s in states:
    print(f"{s}: {policy[s]}")

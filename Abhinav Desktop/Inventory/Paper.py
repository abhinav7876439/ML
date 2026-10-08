import numpy as np

# Parameters
max_inventory = 15
inventory_levels = range(max_inventory + 1)
print("Inventory levels:", list(inventory_levels))
actions = [0, 1]
a_prev_values = [0, 1]
reference_state = (0, 1)

# Cost parameters
h = 1              # holding cost
p = 2              # perish penalty
l = 5              # lost sales cost
gamma = 0.1        # perishability rate  (No assumption on distribution here)
lam = 0.6          # demand rate
mu = 0.8           # production rate
switch_cost = 3    # switching cost

# Penalty function
def w(i):   # penalty for switching, can be customized
    return 0

# Cost function
def C(i, a_prev):
    return h * i + p * gamma * i + l * lam * (i == 0) + w(i) * (a_prev == 1)

# Cost function
# def C(i, a_prev):
#     base_cost = h * i + p * gamma * i + l * lam * (i == 0)
#     if a_prev == 0:
#         return base_cost
#     elif a_prev == 1:
#         return base_cost - w(i)
#     else:
#         raise ValueError("a_prev must be 0 or 1")



def C_switch(i):
    return switch_cost

# State space: (inventory, previous action)
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]

# Initialize value function
V = {s: 0.0 for s in states}
policy = {}
rho = 0.0

# Bellman update functions
def Q(i, a_prev, a_t, V):
    # For a_t = 0
    if a_t == 0:
        if i > 0:
            cost = C(i, a_prev)
            prob_down = i * gamma + lam
            return cost + prob_down * V[(i-1, 0)] + (1 - prob_down) * V[(i, 0)]
        else:
            cost = C(i, a_prev)
            return cost + V[(i, 0)]
    # For a_t = 1
    else:
        cost = C(i, a_prev) - w(i)
        if a_prev == 0:
            cost += C_switch(i)
        if i > 0:
            prob_up = mu
            prob_down = i * gamma + lam
            prob_stay = 1 - (prob_up + prob_down)
            return cost + prob_up * V[(min(i+1, max_inventory), 1)] + prob_down * V[(i-1, 1)] + prob_stay * V[(i, 1)]
        else:
            return cost + V[(i, 1)]

# Relative value iteration
for iteration in range(1000):
    V_new = {}
    for s in states:
        i, a_prev = s
        Q0 = Q(i, a_prev, 0, V)
        Q1 = Q(i, a_prev, 1, V)
        V_new[s] = max(Q0, Q1)
    # Normalize using reference state
    g = max(Q(reference_state[0], reference_state[1], a, V) for a in actions)
    for s in states:
        V_new[s] -= g
    rho = g
    delta = max(abs(V_new[s] - V[s]) for s in states)
    V = V_new
    if delta < 1e-4:
        break

# Extract policy
for s in states:
    i, a_prev = s
    Q0 = Q(i, a_prev, 0, V)
    Q1 = Q(i, a_prev, 1, V)
    policy[s] = actions[np.argmax([Q0, Q1])]

print("\nOptimal policy (state: action):")
for s in states:
    print(f"{s}: {policy[s]}")


############################




# ##################### K Classes Inventory with Threshold Policy ##############
import numpy as np
from itertools import product

# Parameters
K = 3  # number of arms/items
max_inventory = 5
inventory_levels = range(max_inventory + 1)
actions = list(range(0, K+1))  # 0 = no production, 1..K = produce arm
print("Actions (produce arm):", actions)
a_prev_values = list(range(0, K+1))  # previous action can also be 0
reference_state = tuple([0]*K + [0])

# Cost parameters (can be arrays for each arm)
h = np.ones(K)              # holding cost per arm
p = np.ones(K)*2            # perish penalty per arm
l = np.ones(K)*5            # lost sales cost per arm
gamma = np.ones(K)*0.1      # perishability rate per arm
lam = np.ones(K)*0.6        # demand rate per arm
mu = np.ones(K)*0.8         # production rate per arm
switch_cost = 3             # switching cost

def w(i_vec):
    # Example: penalty for switching, can be customized
    return 0

def C(i_vec, a_prev):
    holding = np.sum(h * i_vec)
    perish = np.sum(p * gamma * i_vec)
    lost_sales = np.sum(l * lam * (np.array(i_vec) == 0))
    return holding + perish + lost_sales

def C_switch(i_vec):
    return switch_cost + w(i_vec)

# State space: (I_1, ..., I_K, a_prev)
states = [tuple(list(i_vec) + [a_prev]) for i_vec in product(inventory_levels, repeat=K) for a_prev in a_prev_values]

# Initialize value function
V = {s: 0.0 for s in states}
policy = {}
rho = 0.0

def Q(i_vec, a_prev, a_t, V):
    i_vec = list(i_vec)
    cost = C(i_vec, a_prev)
    if a_prev != a_t:
        cost += C_switch(i_vec)
    next_i_vec = i_vec.copy()
    # Production for chosen arm (convert action from 1-based to 0-based index)
    if a_t != 0:
        arm_idx = a_t - 1
        next_i_vec[arm_idx] = min(next_i_vec[arm_idx] + 1, max_inventory)
    # Perishing and demand for all arms (expected values)
    for k in range(K):
        next_i_vec[k] = max(next_i_vec[k] - int(next_i_vec[k] * gamma[k]), 0)
        next_i_vec[k] = max(next_i_vec[k] - int(next_i_vec[k] * lam[k]), 0)
    next_state = tuple(next_i_vec + [a_t])
    return cost + V[next_state]

# Relative value iteration
for iteration in range(1000):
    V_new = {}
    for s in states:
        i_vec = list(s[:-1])
        a_prev = s[-1]
        Qs = [Q(i_vec, a_prev, a_t, V) for a_t in actions]
        V_new[s] = min(Qs)
    # Normalize using reference state
    g = min(Q(list(reference_state[:-1]), reference_state[-1], a_t, V) for a_t in actions)
    for s in states:
        V_new[s] -= g
    rho = g
    delta = max(abs(V_new[s] - V[s]) for s in states)
    V = V_new
    if delta < 1e-4:
        break

# Extract policy
for s in states:
    i_vec = list(s[:-1])
    a_prev = s[-1]
    Qs = [Q(i_vec, a_prev, a_t, V) for a_t in actions]
    policy[s] = actions[np.argmin(Qs)]

print("\nOptimal policy (state: next arm to produce, 0 means no production):")
for s in states:
    print(f"{s}: {policy[s]}")
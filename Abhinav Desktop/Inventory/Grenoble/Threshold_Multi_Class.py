



################### --------------------------------  ##############

import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 2
max_inventory = 3


# Cost parameters
holding_cost = 1
perish_rate = 0.1
perish_penalty = 2
lost_sales_cost = 5
demand_rate = 0.5
production_rate = 1.0
switching_cost = 3

# State space and actions
states = [(i1, i2, m) for i1 in range(max_inventory + 1)
                      for i2 in range(max_inventory + 1)
                      for m in [1, 2]]
actions = [1, 2]

# Initialize value function and policy
V = {s: 0.0 for s in states}
policy = {}
g = 0.0  # average cost per stage
reference_state = (0, 0, 1)

def running_cost(i1, i2):
    c1 = (holding_cost + perish_penalty * perish_rate) * i1
    c2 = (holding_cost + perish_penalty * perish_rate) * i2
    if i1 == 0:
        c1 += lost_sales_cost * demand_rate
    if i2 == 0:
        c2 += lost_sales_cost * demand_rate
    return c1 + c2

def next_state(s, a):
    i1, i2, m = s
    if a == 1 and i1 < max_inventory:
        i1 += 1
    elif a == 2 and i2 < max_inventory:
        i2 += 1
    i1 = max(0, i1 - int(np.random.rand() < demand_rate) - int(np.random.rand() < perish_rate))
    i2 = max(0, i2 - int(np.random.rand() < demand_rate) - int(np.random.rand() < perish_rate))
    return (i1, i2, a)

def immediate_cost(s, a):
    i1, i2, m = s
    cost = running_cost(i1, i2)
    if a != m:
        cost += switching_cost
    return cost

# Relative value iteration
for iteration in range(1000):
    V_new = {}
    for s in states:
        costs = []
        for a in actions:
            s_next = next_state(s, a)
            cost = immediate_cost(s, a) + V[s_next]
            costs.append(cost)
        V_new[s] = min(costs)
        policy[s] = actions[np.argmin(costs)]
    g = V_new[reference_state] - V[reference_state]
    for s in states:
        V[s] = V_new[s] - V_new[reference_state]
    if max(abs(V[s] - V_new[s]) for s in states) < 1e-4:
        break


# Visualize policy
def plot_policy(m_setup):
    grid = np.zeros((max_inventory + 1, max_inventory + 1))
    for i1 in range(max_inventory + 1):
        for i2 in range(max_inventory + 1):
            grid[i1, i2] = policy[(i1, i2, m_setup)]
    plt.figure(figsize=(6, 5))
    plt.imshow(grid, origin='lower', cmap='coolwarm', extent=[0, max_inventory, 0, max_inventory])
    plt.colorbar(label='Optimal Action')
    plt.xlabel('Inventory Class 2')
    plt.ylabel('Inventory Class 1')
    plt.title(f'Optimal Action Map (Machine Setup = {m_setup})')
    plt.xticks(range(max_inventory + 1))
    plt.yticks(range(max_inventory + 1))
    plt.show()

plot_policy(m_setup=1)
plot_policy(m_setup=2)

################### --------------------------------  ##############

import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
K = 1
max_inventory = 3 
inventory_levels = range(max_inventory + 1)
# print("Inventory levels:", list(inventory_levels))
actions = [0,1]
a_prev_values = [0,1]
reference_state = (0, 1)

# Cost parameters
holding_cost = 1
perish_rate = 0.1
perish_penalty = 2
lost_sales_cost = 5
lambdas = [0.5]
switch_cost = 3

# State space
states = [(i1, a_prev) for i1 in inventory_levels
                                 for a_prev in a_prev_values]

# Initialize relative value function
h_val = {s: 0.0 for s in states}
policy = {}
rho = 0.0

# Cost function
def c_hat(i1, a_prev, a_curr):
    holding = holding_cost * (i1)
    perish = perish_penalty * perish_rate * (i1)
    lost_sales = lost_sales_cost * (
        lambdas[0] * (i1 == 0))
    switch = switch_cost if a_curr != a_prev else 0
    return holding + perish + lost_sales + switch

# Transition function
def next_state(s, a_curr):
    i1, _ = s
    inventories = [i1]
    if inventories[a_curr - 1] < max_inventory:
        inventories[a_curr - 1] += 1
    # Apply demand and perish
    for k in range(K):
        if np.random.rand() < lambdas[k] + perish_rate:
            inventories[k] = max(0, inventories[k] - 1)
    return (inventories[0], inventories[1], inventories[2], a_curr)

# Relative value iteration
for iteration in range(1000):
    h_new = {}
    for s in states:
        costs = []
        for a in actions:
            s_next = next_state(s, a)
            costs.append(c_hat(*s, a) + h_val[s_next])
        h_new[s] = min(costs)
    # Normalize using reference state
    g = min(c_hat(*reference_state, a) + h_val[next_state(reference_state, a)] for a in actions)
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
        s_next = next_state(s, a)
        costs.append(c_hat(*s, a) + h_val[s_next])
    policy[s] = actions[np.argmin(costs)]

# Visualization
os.makedirs("/mnt/data", exist_ok=True)
for a_prev in a_prev_values:
    grid = np.zeros((len(inventory_levels), len(inventory_levels), len(inventory_levels)))
    for i1 in inventory_levels:
        for i2 in inventory_levels:
            for i3 in inventory_levels:
                grid[i1, i2, i3] = policy[(i1, i2, i3, a_prev)]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(inventory_levels, inventory_levels, inventory_levels, indexing='ij')
    ax.scatter(x, y, z, c=grid.flatten(), cmap='coolwarm', marker='o')
    ax.set_xlabel('I1')
    ax.set_ylabel('I2')
    ax.set_zlabel('I3')
    ax.set_title(f'Optimal Action for a_prev = {a_prev}')
    output_path = f"/mnt/data/policy_aprev_{a_prev}.png"
    plt.show()
    plt.close()


# 2D Heatmap for fixed I3
a_prev = 2
fixed_i3 = 1  # Fix inventory of class 3

# Create 2D grid for I1 and I2
grid = np.zeros((len(inventory_levels), len(inventory_levels)))

for i1 in inventory_levels:
    for i2 in inventory_levels:
        state = (i1, i2, fixed_i3, a_prev)
        grid[i1, i2] = policy[state]

# Plot heatmap
plt.figure(figsize=(6, 5))
plt.imshow(grid.T, origin='lower', cmap='coolwarm', extent=[0, max_inventory, 0, max_inventory])
plt.colorbar(label='Optimal Action')
plt.xlabel('Inventory Class 1 (I1)')
plt.ylabel('Inventory Class 2 (I2)')
plt.title(f'Optimal Action Map (a_prev = {a_prev}, I3 = {fixed_i3})')
plt.xticks(inventory_levels)
plt.yticks(inventory_levels)
plt.grid(False)
plt.show()

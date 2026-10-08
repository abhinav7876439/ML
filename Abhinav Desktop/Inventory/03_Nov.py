# ===============================================================
# Relative Value Iteration for Average-Cost MDP with Switching
# ===============================================================
# Checks if the optimal policy has a threshold structure in inventory


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
max_inventory = 15
inventory_levels = range(max_inventory + 1)
print("Inventory levels:", list(inventory_levels))
actions = [0, 1]           # 0 = passive, 1 = active
a_prev_values = [0, 1]
reference_state = (0, 1)   # state used for normalization

# Cost parameters
h = 1              # holding cost
p = 2              # perish penalty
l = 5              # lost sales cost
gamma = 0.1        # perishability rate
lam = 0.6          # demand rate
mu = 0.8           # production rate
switch_cost = 3    # switching cost

# Optional penalty function
def w(i):
    return 0

# Base cost function C(i, a_prev)
def C(i, a_prev):
    return h * i + p * gamma * i + l * lam * (1 if i == 0 else 0) + w(i) * (1 if a_prev == 1 else 0)

# Switching cost
def C_switch(i):
    return switch_cost



# State Space
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]
state_index = {s: idx for idx, s in enumerate(states)}
print("state_index:", state_index)
nS = len(states)



def transitions(state, action):
    i, a_prev = state
    trans = {}

    prod_prob = mu if action == 1 else 0.0
    demand_prob = lam
    perish_prob = gamma

    # Enumerate production
    for prod in [0, 1]:
        prob_prod = prod_prob if prod == 1 else (1 - prod_prob)
        # Enumerate demand
        for dem in [0, 1]:
            prob_dem = demand_prob if dem == 1 else (1 - demand_prob)
            # Enumerate perishing (1 unit can perish)
            for per in [0, 1]:
                prob_per = perish_prob if per == 1 else (1 - perish_prob)
                # Total probability for this event combination
                prob = prob_prod * prob_dem * prob_per
                # Inventory update
                j = i + prod - dem - per
                j = max(0, min(max_inventory, j))
                trans[(j, action)] = trans.get((j, action), 0) + prob

    return trans

def get_rates(i, a):
    """
    Compute the continuous-time rates for up and down events.
    i: current inventory
    a: current action (0=passive, 1=active)
    """
    r_up = mu if a == 1 else 0        # production rate if active
    r_down = lam + gamma * i          # demand + perish rate (per unit)
    return r_up, r_down


def transitions_ctmc(state, action, tau=1.0):
    """
    CTMC-inspired transition probabilities for small tau.
    state: (inventory, previous action)
    action: 0=passive, 1=active
    tau: time step
    Returns: dict of next_state -> probability
    """
    i, a_prev = state
    trans = {}

    r_up, r_down = get_rates(i, action)
    R = r_up + r_down

    if R == 0:  # no event possible (i=0, a=0)
        trans[(i, action)] = 1.0
        return trans

    # Probabilities for this small time step
    p_up = r_up / R * (1 - np.exp(-R * tau))
    p_down = r_down / R * (1 - np.exp(-R * tau))
    p_stay = np.exp(-R * tau)

    # Next states
    i_up = min(i + 1, max_inventory)
    i_down = max(i - 1, 0)

    trans[(i_up, action)] = p_up
    trans[(i_down, action)] = p_down
    trans[(i, action)] = p_stay

    return trans


# Cost Function
def one_step_cost(state, action):
    """Total instantaneous cost including switching"""
    i, a_prev = state
    base = C(i, a_prev)

    # Add switching cost if a changes
    if a_prev == 0 and action == 1:
        base += C_switch(i)
    if a_prev == 1 and action == 0:
        base += C_switch(i)

    return base


# Precompute P(s,a) and c(s,a)
P = {}
c = {}

for s in states:
    si = state_index[s]
    for a in actions:
        trans = transitions_ctmc(s, a, tau=1.0)
        # trans = transitions_ctmc(s, a, 1)
        P[(si, a)] = {state_index[s2]: prob for s2, prob in trans.items()}
        c[(si, a)] = one_step_cost(s, a)



# Relative Value Iteration (RVI)
def relative_value_iteration(P, c, ref_state, tol=1e-6, max_iter=20000):
    nS = len(states)
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ref_idx = state_index[ref_state]     ## h(s) = relative value function (value measured relative to a reference state)
    rho = 0.0    ## ρ = long-run average cost per stage (gain)

    for it in range(max_iter):
        V_old = V.copy()
        Q = np.zeros((nS, 2))

        # Bellman update
        for si in range(nS):
            for a in actions:
                q = c[(si, a)] + sum(p * V_old[sj] for sj, p in P[(si, a)].items())
                Q[si, a] = q

        V_new = np.min(Q, axis=1)
        policy = np.argmin(Q, axis=1)

        # Normalization (relative value): set reference state value to 0
        shift = V_new[ref_idx]                ## relative value constraint
        V_new = V_new - shift

        # Optional tracking of rho
        rho += shift
        V = V_new

        delta = np.max(np.abs(V - V_old))
        if delta < tol:
            break

    # Final recomputation of Q and policy
    Q = np.zeros((nS, 2))
    for si in range(nS):
        for a in actions:
            Q[si, a] = c[(si, a)] + sum(p * V[sj] for sj, p in P[(si, a)].items())

    policy = np.argmin(Q, axis=1)
    rho_est = rho

    return V, policy, Q, rho_est, it + 1


# Run RVI
V, policy, Q, rho_est, iters = relative_value_iteration(P, c, reference_state, tol=1e-8, max_iter=20000)



# Organize and Analyze Policy
policy_table = {0: [], 1: []}

for i in inventory_levels:
    for a_prev in a_prev_values:
        si = state_index[(i, a_prev)]
        policy_table[a_prev].append(policy[si])

# Create table
rows = []
for i in inventory_levels:
    row = {"inventory": i}
    for a_prev in a_prev_values:
        row[f"opt_action_a_prev={a_prev}"] = policy_table[a_prev][i]
    rows.append(row)

policy_df = pd.DataFrame(rows)

# Check threshold structure
threshold_info = {}
for a_prev in a_prev_values:
    actions_list = policy_table[a_prev]
    switches = [i for i in range(1, len(actions_list)) if actions_list[i] != actions_list[i - 1]]
    threshold_info[a_prev] = {
        "actions": actions_list,
        "num_switches": len(switches),
        "switch_locations": switches
    }


# ---------------------------
# Results
# ---------------------------
# print("Relative Value Iteration completed in", iters, "iterations.")
# print("Estimated cumulative rho:", rho_est)
# print("\nPolicy table:")
# print(policy_df.to_string(index=False))

print("\nThreshold check:")
for a_prev in a_prev_values:
    info = threshold_info[a_prev]
    print(f"a_prev={a_prev}: num_switches={info['num_switches']}, switch_indices={info['switch_locations']}")

# # Save for visualization
# policy_df.to_csv("policy_table_inventory_prevaction.csv", index=False)
# print("\nPolicy table saved as 'policy_table_inventory_prevaction.csv'")


# Check Monotonicity and Indexability


# Grid of alpha values for the penalty function w(i) = alpha * i
alpha_grid = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]

# Store results
policy_vs_alpha = []

for alpha in alpha_grid:
    # redefine w(i)
    def w(i):
        return alpha * i

    # Recompute costs with new w(i)
    c = {}
    for s in states:
        si = state_index[s]
        for a in actions:
            trans = transitions(s, a)
            P[(si, a)] = {state_index[s2]: prob for s2, prob in trans.items()}
            c[(si, a)] = one_step_cost(s, a)
    
    # Run RVI
    V, policy, Q, rho_est, iters = relative_value_iteration(P, c, reference_state, tol=1e-8, max_iter=20000)
    print(f"Alpha={alpha}: RVI completed in {iters} iterations, rho_est={rho_est:.4f}")
    print(f"Policy actions for alpha={alpha}:")
    for i in inventory_levels:
        for a_prev in a_prev_values:
            si = state_index[(i, a_prev)]
            print(f"  Inventory={i}, a_prev={a_prev} -> action={policy[si]}")
    
    print("Value function V:", V[:])
    print("Q-values:", Q[:,:])
    
    # Store policy
    for a_prev in a_prev_values:
        actions_list = [policy[state_index[(i, a_prev)]] for i in inventory_levels]
        policy_vs_alpha.append({
            "alpha": alpha,
            "a_prev": a_prev,
            "actions": actions_list,
            "monotone": all(actions_list[i] <= actions_list[i+1] for i in range(len(actions_list)-1))
        })

# Convert to DataFrame
policy_alpha_df = pd.DataFrame(policy_vs_alpha)

# ---------------------------
# Print monotonicity results
# ---------------------------
print("Monotonicity check for each alpha and previous action:")
for a_prev in a_prev_values:
    df_sub = policy_alpha_df[policy_alpha_df["a_prev"]==a_prev]
    for _, row in df_sub.iterrows():
        print(f"a_prev={a_prev}, alpha={row['alpha']}: monotone={row['monotone']}")



# Indexability check
print("\nIndexability check (action should not increase with alpha for each inventory):")
for a_prev in a_prev_values:
    print(f"\na_prev={a_prev}")
    for i in inventory_levels:
        actions_seq = [policy_alpha_df[(policy_alpha_df['a_prev']==a_prev) & 
                                       (policy_alpha_df['alpha']==alpha)]['actions'].values[0][i]
                       for alpha in alpha_grid]
        non_increasing = all(actions_seq[j] >= actions_seq[j+1] for j in range(len(actions_seq)-1))
        print(f"Inventory={i}, actions vs alpha: {actions_seq}, non-increasing={non_increasing}")



################Whittle Indices Computation#########################



def find_alpha_switch(alpha_grid, policy_alpha_df, i, a_prev):
    """
    Finds the alpha value (approximate Whittle index) where the optimal action switches.
    Returns None if no switch occurs in the alpha range.
    """
    # Extract actions across alpha values for this (i, a_prev)
    actions_seq = [policy_alpha_df[
        (policy_alpha_df["alpha"] == alpha) &
        (policy_alpha_df["a_prev"] == a_prev)
    ]["actions"].values[0][i] for alpha in alpha_grid]

    # Find switch: active (1) -> passive (0) as alpha increases
    for k in range(len(alpha_grid) - 1):
        if actions_seq[k] == 1 and actions_seq[k + 1] == 0:
            # Linear interpolation between α_k and α_{k+1}
            α_low, α_high = alpha_grid[k], alpha_grid[k + 1]
            return 0.5 * (α_low + α_high)

    # If no switch found, return boundary value (min or max)
    if all(a == 1 for a in actions_seq):
        return np.min(alpha_grid) - 0.1  # Always active (low index)
    elif all(a == 0 for a in actions_seq):
        return np.max(alpha_grid) + 0.1  # Always passive (high index)
    else:
        return None


# ----------------------------
# Compute Whittle Indices
# ----------------------------
whittle_indices = {}

for a_prev in [0, 1]:
    for i in inventory_levels:
        switch_alpha = find_alpha_switch(alpha_grid, policy_alpha_df, i, a_prev)
        whittle_indices[(i, a_prev)] = switch_alpha


# ----------------------------
# Organize into DataFrame
# ----------------------------
whittle_df = pd.DataFrame([
    {"inventory": i, "a_prev": a_prev, "whittle_index": whittle_indices[(i, a_prev)]}
    for i in inventory_levels for a_prev in [0, 1]
])

print("\nWhittle Indices:")
print(whittle_df.to_string(index=False))


# ----------------------------
# Plot Whittle Index Curves
# ----------------------------
plt.figure(figsize=(7, 4))
for a_prev in [0, 1]:
    subset = whittle_df[whittle_df["a_prev"] == a_prev]
    plt.plot(subset["inventory"], subset["whittle_index"], marker="o", label=f"a_prev={a_prev}")

plt.xlabel("Inventory Level (i)")
plt.ylabel("Whittle Index (α*)")
plt.title("Whittle Index vs Inventory Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Model parameters

max_inventory = 18
inventory_levels = list(range(max_inventory + 1))
actions = [0, 1]           # 0 = passive, 1 = active
a_prev_values = [0, 1]
reference_state = (0, 1)   # (i, a_prev) used for normalization; must be in state space

# Cost parameters 
h = 1.0            # holding cost (hk)
p = 2.0            # perish penalty (pk)
l = 5.0            # lost-sales penalty (lk)
gamma = 0.1        # perishability rate (γk)
lam = 0.6          # demand rate (λk)
mu = 0.8           # production rate
switch_cost = 3.0  # symmetric switching cost

# State space and index mapping
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]
state_index = {s: idx for idx, s in enumerate(states)}
nS = len(states)
ref_idx = state_index[reference_state]


# Transition model (CTMC-like discrete step)
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
    Outputs dict(next_state -> probability).
    next state's a_prev is set to the current action (since next step's prev action = current action).
    """
    i, a_prev = state
    trans = {}

    r_up, r_down = get_rates(i, action)
    R = r_up + r_down

    if R <= 1e-12: # no event possible (i=0, a=0)
        # no events; remain in same inventory, next a_prev becomes 'action'
        trans[(i, action)] = 1.0
        return trans

    p_event = 1 - np.exp(-R * tau)
    # conditional probabilities given an event:
    p_up = (r_up / R) * p_event
    p_down = (r_down / R) * p_event
    p_stay = 1.0 - p_event

    i_up = min(i + 1, max_inventory)
    i_down = max(i - 1, 0)

    # accumulate
    trans[(i_up, action)] = trans.get((i_up, action), 0.0) + p_up
    trans[(i_down, action)] = trans.get((i_down, action), 0.0) + p_down
    trans[(i, action)] = trans.get((i, action), 0.0) + p_stay

    return trans

#print("Example transitions from state (5,0) with action=1:", transitions_ctmc((5,0), 1, tau=1.0))
#print("Example transitions from state (5,0) with action=0:", transitions_ctmc((5,0), 0, tau=1.0))


# Cost primitives (w handled as scalar parameter)
def C_base(i, a_prev, w):
    """
    Base instantaneous cost for inventory i and previous action a_prev under penalty w.
    """
    cost = h * i + p * gamma * i
    if i == 0:
        cost += l * lam
    if a_prev == 1:
        cost += w
    return cost

def C_switch(i):
    return switch_cost

def one_step_cost(state, action, w):
    """Total instantaneous cost including switching (if action != a_prev)."""
    i, a_prev = state
    base = C_base(i, a_prev, w)
    if a_prev != action:
        base += C_switch(i)
    return base


# Build P and c for a given w
def build_model(w, tau=1.0):
    """
    Precompute P[(si,a)] = {sj: prob} and c[(si,a)] for all si,a under penalty w.
    Returns P,c
    """
    P = {}
    c = {}
    for s in states:
        si = state_index[s]
        for a in actions:
            trans = transitions_ctmc(s, a, tau=tau)
            P[(si, a)] = {state_index[s2]: prob for s2, prob in trans.items()}   # Convert state tuples to indices
            c[(si, a)] = one_step_cost(s, a, w)
    return P, c


# Relative Value Iteration (RVI)
def relative_value_iteration(P, c, ref_idx, tol=1e-8, max_iter=20000):
    nS = len(states)
    V = np.zeros(nS)
    rho = 0.0    # long-run average cost per stage (gain)

    for it in range(max_iter):
        V_old = V.copy()

        # compute Q(s,a) = c(s,a) + sum P(s'|s,a) * V_old[s']
        Q = np.full((nS, 2), np.inf)
        for si in range(nS):
            for a in actions:
                Q[si, a] = c[(si, a)] + sum(p * V_old[sj] for sj, p in P[(si, a)].items())

        V_new = np.min(Q, axis=1)
        policy = np.argmin(Q, axis=1)

        # normalize so V_new[ref_idx] = 0
        shift = V_new[ref_idx]
        V_new = V_new - shift

        V = V_new
        rho = shift  # approximate long-run average cost per stage (gain) as shift each iter; final value at convergence

        delta = np.max(np.abs(V - V_old))
        if delta < tol:
            break

    # final Q and policy (with converged V)
    Q = np.full((nS, 2), np.inf)
    for si in range(nS):
        for a in actions:
            Q[si, a] = c[(si, a)] + sum(p * V[sj] for sj, p in P[(si, a)].items())
    policy = np.argmin(Q, axis=1)

    return V, policy, Q, rho, it + 1

def extract_policy(w):
    P, c = build_model(w)
    V, policy, Q, rho, iterations = relative_value_iteration(P, c, ref_idx)
    policy_table = pd.DataFrame({
        "state": states,
        "optimal_action": policy
    })
    return policy_table

# Example Usage
w = -10   # choose penalty parameter
policy_df = extract_policy(w)
print("\nOptimal action for w:", w)
print(policy_df)


# Finite-horizon relative DP recursion
def finite_horizon_relative(P, c, T, ref_idx, minimize=True):
    """
    Returns a list of dicts for t=1..T:
      each dict contains 't','g_t','V_t','V_tilde','policy_t'
      - V_t is the standard finite-horizon value at horizon t
      - g_t = V_t[ref] - V_{t-1}[ref]
      - V_tilde = V_t - t*g_t (relative normalization as in your snippet)
    """
    nS = len(states)
    nA = len(actions)
    V_prev = np.zeros(nS)  # V_0 = 0
    results = []
    for t in range(1, T+1):
        Q = np.full((nS, nA), np.inf)
        for si in range(nS):
            for a in actions:
                Q[si, a] = c[(si, a)] + sum(p * V_prev[sj] for sj, p in P[(si, a)].items())
        if minimize:
            V_t = np.min(Q, axis=1)
            policy_t = np.argmin(Q, axis=1)
        else:
            V_t = np.max(Q, axis=1)
            policy_t = np.argmax(Q, axis=1)
        g_t = V_t[ref_idx] - V_prev[ref_idx]
        V_tilde = V_t - t * g_t
        results.append({"t": t, "g_t": g_t, "V_t": V_t.copy(), "V_tilde": V_tilde.copy(), "policy_t": policy_t.copy()})
        V_prev = V_t.copy()
    return results

# Indexability and monotonicity scan over w
def scan_over_w(w_values, tau=1.0, tol=1e-8):
    """
    For each w in w_values:
      - build model
      - run RVI -> obtain policy
      - record Pb(w) = set of states (si) where passive (action=0) is optimal

    Returns:
      Pb_by_w : dict w -> set(si)
      policy_by_w : dict w -> policy array
      Q_by_w : dict w -> Q matrix
      rho_by_w : dict w -> rho
    """
    Pb_by_w = {}
    policy_by_w = {}
    Q_by_w = {}
    rho_by_w = {}

    for w in w_values:
        P, c = build_model(w, tau=tau)
        V, policy, Q, rho, iters = relative_value_iteration(P, c, ref_idx, tol=tol)
        Pb = set(si for si in range(nS) if policy[si] == 0)
        Pb_by_w[w] = Pb
        policy_by_w[w] = policy.copy()
        Q_by_w[w] = Q.copy()
        rho_by_w[w] = rho

    return Pb_by_w, policy_by_w, Q_by_w, rho_by_w


w_values = np.linspace(-2, 2, 81)  # from -1000 to 1000 step 50

Pb_by_w, policy_by_w, Q_by_w, rho_by_w = scan_over_w(w_values)


print("Scanned Pb(w) for w in", w_values)
print("Example Pb(-1000):", Pb_by_w[w_values[0]])
print("policy at w=-1000:\n", policy_by_w[w_values[0]])
print("rho at w=-1000:", rho_by_w[w_values[0]])
print("Q at w=-1000:\n", Q_by_w[w_values[0]])


### Visualizes the Passive Set Pb(w) as a heatmap
def plot_passive_sets(Pb_by_w, n_states):
    sorted_w = sorted(Pb_by_w.keys())
    w_vals = np.array(sorted_w)
    
    grid = np.zeros((n_states, len(w_vals)))
    
    for col_idx, w in enumerate(sorted_w):
        passive_states = Pb_by_w[w]
        for s in range(n_states):
            if s in passive_states:
                grid[s, col_idx] = 1  # Passive
            else:
                grid[s, col_idx] = 0  # Active


    if len(w_vals) > 1:
        step = w_vals[1] - w_vals[0]
        w_edges = np.concatenate([w_vals - step/2, [w_vals[-1] + step/2]])
    else:
        w_edges = np.array([w_vals[0] - 0.5, w_vals[0] + 0.5])


    state_edges = np.arange(n_states + 1)

    X, Y = np.meshgrid(w_edges, state_edges)


    plt.figure(figsize=(12, 8))

    cmap = plt.get_cmap('viridis', 2) 
    

    im = plt.pcolormesh(X, Y, grid, cmap=cmap, shading='flat', edgecolors='none')
    

    cbar = plt.colorbar(im, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Active (Action 1)', 'Passive (Action 0)'])
    
    plt.title('Passive Set $P_b(w)$ vs. Subsidy $w$\n(Yellow = Passive, Blue = Active)', fontsize=14)
    plt.xlabel('Subsidy $w$', fontsize=12)
    plt.ylabel('State Index', fontsize=12)
    
    plt.tight_layout()
    print("Plot generated successfully.")
    plt.show()



# EXECUTION
if __name__ == "__main__":
    # Simulation parameters
    nS = 32
    
    # Use 41 points to match your snippet, or 100 for smoother res
    w_values = np.linspace(-2, 2, 81)
    
    Pb_by_w_demo = {}
    policy_by_w_demo = {}
    rho_by_w_demo = {}
    Q_by_w_demo = {}

    # Generate Mock Data (simulating scan_over_w)
    print(f"Generating mock data for {len(w_values)} w values...")
    for w in w_values:
        # At w=-1000 (high penalty for passive), nearly everyone is Active (Pb is empty)
        # At w=1000 (high subsidy), nearly everyone is Passive
        
        normalized_w = (w + 1000) / 2000  # 0.0 to 1.0
        
        # Threshold: States above this index are Passive
        # As w increases, threshold drops, so MORE states become passive
        threshold_state = int(nS * (1 - normalized_w))
        
        Pb = set(range(threshold_state, nS))
        Pb_by_w_demo[w] = Pb
        
        # Mock Policy/Rho/Q just to prevent errors if you access them
        policy_by_w_demo[w] = np.zeros(nS) 
        rho_by_w_demo[w] = 100.0
        Q_by_w_demo[w] = np.zeros((nS, 2))


    first_w = w_values[0] 
    
    print(f"\n--- Debugging for w = {first_w} ---")
    print(f"Pb set at w={first_w}:", Pb_by_w_demo[first_w])
    # print("Policy:", policy_by_w_demo[first_w]) # Uncomment to see arrays

 
    plot_passive_sets(Pb_by_w_demo, nS)


###################################

# Monotonicity / Indexability check
def check_monotonicity(Pb_by_w, w_values):
    """
    Check whether Pb(w_i) subset of Pb(w_{i+1}) for increasing w_values
    Returns boolean and list of violations (tuples)
    """
    is_indexable = True
    violations = []
    for i in range(len(w_values) - 1):
        w1 = w_values[i]
        w2 = w_values[i + 1]
        Pb1 = Pb_by_w[w1]
        Pb2 = Pb_by_w[w2]
        if not Pb1.issubset(Pb2):    ## Is every state inside Pb1 also inside Pb2
            is_indexable = False
            missing = Pb1 - Pb2
            violations.append((w1, w2, missing))
    return is_indexable, violations

def compute_whittle_indices(Pb_by_w, w_values):
    """
    For each state si, compute W(si) = inf { w : si in Pb(w) } using the scanned w_values
    If a state never becomes passive in the scanned range, set to np.inf
    Returns dict si -> W
    """
    whittle = {}
    sorted_w = sorted(w_values)
    for si in range(nS):
        found = False   # Assumed that we haven't found its index yet
        for w in sorted_w:
            if si in Pb_by_w[w]:
                whittle[si] = w
                found = True
                break
        if not found:
            whittle[si] = np.inf
    return whittle


# ---------------------------
# Main run: choose w grid, scan, check
# ---------------------------
if __name__ == "__main__":
    # choose w grid (extend range if needed)
    w_values = np.linspace(-2.0, 2.0, 81)  
    tau = 1.0
    Pb_by_w, policy_by_w, Q_by_w, rho_by_w = scan_over_w(w_values, tau=tau, tol=1e-8)

    # Check indexability (stateful: full (i,a_prev) states)
    indexable_stateful, violations_stateful = check_monotonicity(Pb_by_w, sorted(w_values))
    print("\nIndexable (stateful = (i,a_prev))?:", indexable_stateful)
    if not indexable_stateful:
        print("\nViolations (w1, w2, states in Pb(w1) but not in Pb(w2)):")
        for v in violations_stateful[:10]:
            w1, w2, missing = v
            # show missing as (i,a_prev) readable states
            missing_readable = [states[si] for si in missing]
            print(f"  between {w1} -> {w2}: {missing_readable}")

    # Compute Whittle index per full state
    whittle_state = compute_whittle_indices(Pb_by_w, w_values)
    whittle_state_readable = {states[si]: whittle_state[si] for si in range(nS)}
    print("\nWhittle index per state (inventory,i_prev):")
    for s, wval in sorted(whittle_state_readable.items()):
        print(f"  State {s}: W = {wval}")



# #################################################

#     # Inventory-level Pb and index (optional)
#     # Define inventory-level Pb_inv(w) as the set of inventory levels i
#     # for which BOTH previous-action variants prefer passive (a_prev=0 and a_prev=1).
#     # This is stricter — change to any-other definition if you prefer.

#     Pb_inv_by_w = {}
#     for w in w_values:
#         Pb_si = Pb_by_w[w]     #  Set of all specific states that are passive at this w
#         inv_set = set()
#         for i in inventory_levels:
#             # find indices for both a_prev values
#             si0 = state_index[(i, 0)]
#             si1 = state_index[(i, 1)]
#             if si0 in Pb_si and si1 in Pb_si:
#                 inv_set.add(i)
#         Pb_inv_by_w[w] = inv_set

#     indexable_inventory, violations_inventory = check_monotonicity(Pb_inv_by_w, sorted(w_values))
#     print("\nIndexable (inventory-level requiring both a_prev passive)?:", indexable_inventory)
#     if not indexable_inventory:
#         print("Inventory-level violations (example):")
#         for v in violations_inventory[:10]:
#             w1, w2, missing = v
#             print(f"  between {w1} -> {w2}: {missing}")

#     # Whittle index per inventory (first w where inventory i has passive for both a_prev)
#     whittle_inventory = {}
#     for i in inventory_levels:
#         found = False
#         for w in sorted(w_values):
#             if i in Pb_inv_by_w[w]:
#                 whittle_inventory[i] = w
#                 found = True
#                 break
#         if not found:
#             whittle_inventory[i] = np.inf

#     print("\nWhittle index per inventory-level (requiring passive for both a_prev):")
#     for i in sorted(whittle_inventory.keys()):
#         print(f"  Inventory {i}: W = {whittle_inventory[i]}")

#     # Optional: build a DataFrame summarizing policy at a few representative w values
#     sample_ws = [w_values[0], w_values[len(w_values)//2], w_values[-1]]
#     rows = []
#     for w in sample_ws:
#         pol = policy_by_w[w]
#         # Map policy for a_prev=0 and a_prev=1 separately across inventory levels
#         for i in inventory_levels:
#             rows.append({
#                 "w": w,
#                 "inventory": i,
#                 "opt_a_prev_0": pol[state_index[(i, 0)]],
#                 "opt_a_prev_1": pol[state_index[(i, 1)]]
#             })
#     policy_df = pd.DataFrame(rows)
#     print("\nPolicy sample (first rows):")
#     print(policy_df.head(10))

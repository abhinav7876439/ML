import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Model parameters
max_inventory = 15
inventory_levels = list(range(max_inventory + 1))
a_prev_values = [0, 1]      # previous action in state
actions = [0, 1]            # 0 = passive, 1 = active
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]
state_index = {s: idx for idx, s in enumerate(states)}
nS = len(states)


# reference state for normalization
reference_state = (0, 1)
if reference_state not in state_index:
    raise ValueError(f"reference_state {reference_state} not in state space.")
ref_idx = state_index[reference_state]


# Cost parameters (tune to your modeling intent)
h = 1.0            # holding cost per unit inventory per step
p = 2.0            # perish penalty scale (per unit lost item)
l = 5.0            # lost-sales penalty per lost demand occurrence
gamma = 0.1        # perishability rate (per-unit per time step)
lam = 0.6          # demand rate
mu = 0.8           # production rate when active
switch_cost = 3.0  # symmetric switching cost when action != a_prev



# Transition model (CTMC-like discrete step)
def get_rates(i, a):
    """
    Continuous-time rates for up and down events (per unit time).
    r_up: production arrival rate (one unit)
    r_down: demand + perish (per-unit) aggregated
    """
    r_up = mu if a == 1 else 0.0
    r_down = lam + gamma * i
    return r_up, r_down




def transitions_ctmc(state, action, tau=1.0):
    """
    Small-time-step CTMC approximation:
      - with probability p_event = 1 - exp(-R * tau) some event (up or down) happens;
      - conditional on an event: up with prob r_up/R, down with prob r_down/R.
    Returns a dict mapping next_state_tuple -> probability.
    next state's a_prev is set to the current action.
    """
    i, _ = state
    r_up, r_down = get_rates(i, action)
    R = r_up + r_down

    trans = {}
    if R <= 1e-12:
        trans[(i, action)] = 1.0
        return trans

    p_event = 1.0 - np.exp(-R * tau)
    p_up = (r_up / R) * p_event if R > 0 else 0.0
    p_down = (r_down / R) * p_event if R > 0 else 0.0
    p_stay = 1.0 - p_event

    i_up = min(i + 1, max_inventory)
    i_down = max(i - 1, 0)

    trans[(i_up, action)] = trans.get((i_up, action), 0.0) + p_up
    trans[(i_down, action)] = trans.get((i_down, action), 0.0) + p_down
    trans[(i, action)] = trans.get((i, action), 0.0) + p_stay

    return trans

print("Example transitions from state (5,0) with action=1:", transitions_ctmc((5,0), 1, tau=1.0))
print("Example transitions from state (5,0) with action=0:", transitions_ctmc((5,0), 0, tau=1.0))

# Cost
def C_base(i, a_prev, w):
    """
    Base instantaneous cost for state (i, a_prev) under subsidy/penalty w.
    Note: review modeling semantics for lost sales and perish penalty.
    """
    cost = h * i                   # holding cost
    cost += p * gamma * i          # expected perish penalty (per time-step; check model meaning)
    # lost-sales: currently charged deterministically when i == 0 (per-step)
    if i == 0:
        cost += l * lam            # Often lost-sales cost is l * demand or l * Prob(demand>0)
    # subsidy w applied when previous action was active (this is your modelling choice)
    if a_prev == 1:
        cost += w
    return cost

def C_switch(i):
    return switch_cost            # could be state-dependent; here constant switch_cost

# # If ON → OFF and OFF → ON have different costs
# def C_switch(i, a_prev, a):
#     if a_prev == 0 and a == 1:     # start production
#         return switch_cost * (1 + i)
#     elif a_prev == 1 and a == 0:   # stop production
#         return 0.5 * switch_cost
#     return 0.0



# def C_switch(i):
#     """
#     Smooth, bounded switching cost.
#     """
#     return switch_cost * (1 + i / max_inventory)



def one_step_cost(state, action, w):
    i, a_prev = state
    base = C_base(i, a_prev, w)
    if a_prev != action:
        base += C_switch(i)
    return base


# Build model: P and c
def build_model(w, tau=1.0):
    """
    Build P and c in index form:
      P[(si, a)] = {sj: prob}
      c[(si, a)] = cost scalar
    """
    P = {}
    c = {}
    for s in states:
        si = state_index[s]
        for a in actions:
            trans = transitions_ctmc(s, a, tau=tau)
            P[(si, a)] = {state_index[s2]: prob for s2, prob in trans.items()}
            c[(si, a)] = one_step_cost(s, a, w)
    return P, c


# Relative Value Iteration (RVI)
def relative_value_iteration(P, c, ref_idx, tol=1e-8, max_iter=20000, verbose=False):   #verbose: whether to print progress
    nS = len(states)
    V = np.zeros(nS)
    rho = 0.0             # long-run average cost per stage (gain)

    for it in range(1, max_iter + 1):
        V_old = V.copy()
        # Bellman (one-step)
        Q = np.full((nS, 2), np.inf)
        for si in range(nS):
            for a in actions:
                Q[si, a] = c[(si, a)] + sum(p * V_old[sj] for sj, p in P[(si, a)].items())

        V_new = np.min(Q, axis=1)
        # shift so reference state's value is zero
        shift = V_new[ref_idx]
        V_new = V_new - shift
        V = V_new
        rho = shift

        delta = np.max(np.abs(V - V_old))
        if verbose and (it % 500 == 0 or delta < tol):
            print(f"RVI iter {it}, delta={delta:.3e}, rho~={rho:.6f}")
        if delta < tol:
            break

    # recompute final Q and policy
    Q = np.full((nS, 2), np.inf)
    for si in range(nS):
        for a in actions:
            Q[si, a] = c[(si, a)] + sum(p * V[sj] for sj, p in P[(si, a)].items())
    policy = np.argmin(Q, axis=1)

    return V, policy, Q, rho, it

def extract_policy(w):
    P, c = build_model(w)
    V, policy, Q, rho, iterations = relative_value_iteration(P, c, ref_idx)
    policy_table = pd.DataFrame({
        "state": states,
        "optimal_action": policy
    })
    return policy_table


# Example Usage
w = 1.5   # choose penalty parameter
policy_df = extract_policy(w)
print("\nOptimal action for w:", w)
print(policy_df)



# Finite-horizon relative DP
def finite_horizon_relative(P, c, T, ref_idx, minimize=True):
    nS = len(states)
    nA = len(actions)
    V_prev = np.zeros(nS)
    results = []
    for t in range(1, T + 1):
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
def scan_over_w(w_values, tau=1.0, tol=1e-8, verbose=False, warm_start=True):   # warm_start: whether to reuse previous V as initial guess for next w
    """
    For each w compute RVI and record Pb(w) (states where passive is optimal).
    Optionally warm-start V from previous w to accelerate convergence.
    """
    Pb_by_w = {}
    policy_by_w = {}
    Q_by_w = {}
    rho_by_w = {}

    # warm-start V across w-values
    V_warm = None

    for idx, w in enumerate(sorted(w_values)):
        P, c = build_model(w, tau=tau)

        if warm_start and V_warm is not None:   # Warm start is used only except at first w
            # injecting warm-start via a tiny modification: we pass V_warm indirectly by doing
            # a few "policy-evaluation-like" steps is more involved. Here we simply call RVI
            V, policy, Q, rho, iters = relative_value_iteration(P, c, ref_idx, tol=tol, max_iter=20000, verbose=verbose)  # Whatever verbose the user gave to scan_over_w, pass the same value into relative_value_iteration
        else:
            V, policy, Q, rho, iters = relative_value_iteration(P, c, ref_idx, tol=tol, max_iter=20000, verbose=verbose)

        Pb = set(si for si in range(nS) if policy[si] == 0)
        Pb_by_w[w] = Pb
        policy_by_w[w] = policy.copy()
        Q_by_w[w] = Q.copy()
        rho_by_w[w] = rho
        V_warm = V.copy()
        if verbose:
            print(f"w={w:.4f}  iters={iters}  rho~={rho:.6f}  |Pb|={len(Pb)}")

    return Pb_by_w, policy_by_w, Q_by_w, rho_by_w

w_values = np.linspace(-2, 2, 81)  

Pb_by_w, policy_by_w, Q_by_w, rho_by_w = scan_over_w(w_values)


print("Scanned Pb(w) for w in", w_values)
print("Example Pb(-1000):", Pb_by_w[w_values[0]])
print("policy at w=-1000:\n", policy_by_w[w_values[0]])
print("rho at w=-1000:", rho_by_w[w_values[0]])
print("Q at w=-1000:\n", Q_by_w[w_values[0]])


# Monotonicity & Whittle extraction
def check_monotonicity(Pb_by_w, w_values):
    """
    Check whether Pb(w_i) subset of Pb(w_{i+1}) for increasing w_values
    Returns boolean and list of violations (tuples)
    """
    is_indexable = True
    violations = []
    sorted_w = sorted(w_values)
    for i in range(len(sorted_w) - 1):
        w1 = sorted_w[i]
        w2 = sorted_w[i + 1]
        Pb1 = Pb_by_w[w1]
        Pb2 = Pb_by_w[w2]
        if not Pb1.issubset(Pb2):
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
        found = False    # Assumed that we haven't found its index yet
        for w in sorted_w:
            if si in Pb_by_w[w]:
                whittle[si] = w
                found = True
                break
        if not found:
            whittle[si] = np.inf
    return whittle




# Plotting Pb(w)
def plot_passive_sets(Pb_by_w, n_states, cmap_colors=('tab:blue', 'gold')):
    sorted_w = sorted(Pb_by_w.keys())
    w_vals = np.array(sorted_w)
    grid = np.zeros((n_states, len(w_vals)))

    for col_idx, w in enumerate(sorted_w):
        passive_states = Pb_by_w[w]
        for s in range(n_states):
            grid[s, col_idx] = 1.0 if s in passive_states else 0.0

    # edges for pcolormesh
    if len(w_vals) > 1:
        step = w_vals[1] - w_vals[0]
        w_edges = np.concatenate([w_vals - step/2, [w_vals[-1] + step/2]])
    else:
        w_edges = np.array([w_vals[0] - 0.5, w_vals[0] + 0.5])
    state_edges = np.arange(n_states + 1)
    X, Y = np.meshgrid(w_edges, state_edges)

    plt.figure(figsize=(12, 8))
    cmap = ListedColormap([cmap_colors[0], cmap_colors[1]])  # 0: Active, 1: Passive
    im = plt.pcolormesh(X, Y, grid, cmap=cmap, shading='flat')
    cbar = plt.colorbar(im, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Active (1)', 'Passive (0)'])
    plt.title('Passive Set $P_b(w)$ vs. Subsidy $w$')
    plt.xlabel('Subsidy $w$')
    plt.ylabel('State Index')
    plt.tight_layout()
    plt.show()





# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # choose w grid for scan
    w_values = np.linspace(-2.0, 2.0, 41)  # moderate resolution; adjust as needed
    tau = 1.0
    # For exploratory scans, relax tol to speed up
    Pb_by_w, policy_by_w, Q_by_w, rho_by_w = scan_over_w(w_values, tau=tau, tol=1e-6, verbose=True, warm_start=True)

    # Check indexability
    indexable_stateful, violations_stateful = check_monotonicity(Pb_by_w, w_values)
    print("\nIndexable (stateful = (i,a_prev))?:", indexable_stateful)
    if not indexable_stateful:
        print("Violations (showing up to 10):")
        for v in violations_stateful[:10]:
            w1, w2, missing = v
            missing_readable = [states[si] for si in sorted(list(missing))]
            print(f"  between {w1:.4f} -> {w2:.4f}: {missing_readable}")

    # Compute Whittle indices
    whittle_state = compute_whittle_indices(Pb_by_w, w_values)
    whittle_readable = {states[si]: whittle_state[si] for si in range(nS)}
    print("\nWhittle index per state (inventory, a_prev):")
    for s, wval in sorted(whittle_readable.items()):
        print(f"  State {s}: W = {wval}")

    # Plot passive sets
    plot_passive_sets(Pb_by_w, nS)

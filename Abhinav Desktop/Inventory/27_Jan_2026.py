import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import product
import csv
import datetime
import markovianbandit as bandit
import os



# Model parameters
max_inventory = 15
inventory_levels = list(range(max_inventory + 1))
a_prev_values = [0, 1]      # previous action in state
actions = [0, 1]            # 0 = passive, 1 = active
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]
state_index = {s: idx for idx, s in enumerate(states)}
nS = len(states)

# reference state for normalization
reference_state = (0,0)
if reference_state not in state_index:
    raise ValueError(f"reference_state {reference_state} not in state space.")
ref_idx = state_index[reference_state]
print(f"Reference state {reference_state} has index {ref_idx}.")


# Cost parameters
h = 1.0            # holding cost per unit inventory per step
p = 2.0            # perish penalty scale (per unit lost item)
l = 5.0            # lost-sales penalty per lost demand occurrence
gamma = 0.1        # perishability rate (per-unit per time step)
lam = 0.3          # demand rate
mu = 0.8           # production rate when active
switch_cost = 0.0  # symmetric switching cost when action != a_prev
tau = 1






# --- PARAMETER GRIDS ---
h_values = [0.5, 1.0, 2.0]
p_values = [1.0, 2.0]
l_values = [2.0, 5.0]
gamma_values = [0.05, 0.1]
lam_values = [0.4, 0.6]
switch_cost_values = [0.0, 0.5]

mu = 0.8          # fixed
tau = 1.0         # fixed
max_inventory = 15





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

    p_event = 1.0 - np.exp(-R * tau)   # Probability at least one event happens in time τ
    p_up = (r_up / R) * p_event if R > 0 else 0.0
    p_down = (r_down / R) * p_event if R > 0 else 0.0
    p_stay = 1.0 - p_event

    i_up = min(i + 1, max_inventory)
    i_down = max(i - 1, 0)

    trans[(i_up, action)] = trans.get((i_up, action), 0.0) + p_up
    trans[(i_down, action)] = trans.get((i_down, action), 0.0) + p_down
    trans[(i, action)] = trans.get((i, action), 0.0) + p_stay

    return trans

# Quick sanity prints
print("Example transitions from state (15,0)) with action=1:", transitions_ctmc((15,0), 1, tau=1.0))
print("Example transitions from state (15,0)) with action=0:", transitions_ctmc((15,0), 0, tau=1.0))



def C_switch():
    return switch_cost            # could be state-dependent; here constant switch_cost



def C_base(i, a_prev, action, w):
    cost = h * i
    cost += p * gamma * i

    if i == 0:
        cost += l * lam * tau

    # switching cost (fine)
    if action != a_prev:
        cost += switch_cost

    # Whittle subsidy
    if action == 0:
        cost -= w

    return cost

def C_base_state_action(i, a_prev, action):
    cost = h * i
    cost += p * gamma * i

    if i == 0:
        cost += l * lam * tau

    # switching cost ONCE
    if action != a_prev:
        cost += switch_cost

    return cost



def set_model_parameters(h_, p_, l_, gamma_, lam_, switch_cost_):
    global h, p, l, gamma, lam, switch_cost
    h = h_
    p = p_
    l = l_
    gamma = gamma_
    lam = lam_
    switch_cost = switch_cost_







def build_model(tau):
    P = {}
    C_mat = np.zeros((nS, 2))

    for s_idx, (i, a_prev) in enumerate(states):
        for a in actions:
            trans = transitions_ctmc((i, a_prev), a, tau=tau)
            P[(s_idx, a)] = {
                state_index[s2]: prob for s2, prob in trans.items()
            }
            C_mat[s_idx, a] = C_base_state_action(i, a_prev, a)

    return P, C_mat



# Relative Value Iteration (RVI)
def relative_value_iteration(P, C, w, ref_idx,
                             tol=1e-8, max_iter=10000, V_init=None, verbose=False):
    """
    Solve subsidized average-cost MDP using RVI.
    Subsidy w is applied to passive action (a=0).
    """
    nS = len(states)
    if V_init is None:
        V = np.zeros(nS)
    else:
        V = np.array(V_init, dtype=float).copy()
        if V.shape[0] != nS:
            V = np.zeros(nS)

    # Apply subsidy
    C_sub = C.copy()
    C_sub[:, 1] += w

    for it in range(1, max_iter + 1):
        V_old = V.copy()
        Q = np.full((nS, 2), np.inf)


        for si in range(nS):
            for a in actions:
                Q[si, a] = C_sub[si, a] + sum(prob * V_old[sj] for sj, prob in P[(si, a)].items())

        # Bellman update
        V_tilde = np.min(Q, axis=1)

        # Relative normalization
        rho = V_tilde[ref_idx]
        V = V_tilde - rho

        delta = np.max(np.abs(V - V_old))
        if verbose and (it % 500 == 0 or delta < tol):
            print(f"RVI iter {it}, delta={delta:.3e}, rho~={rho:.6f}")
        if delta < tol:
            break

    # Recompute final Q and policy
    Q = np.zeros((nS, 2))
    for si in range(nS):
        for a in actions:
            Q[si, a] = C_sub[si, a] + sum(p * V[sj] for sj, p in P[(si, a)].items())
    policy = np.argmin(Q, axis=1)

    return V, policy, Q, rho, it



def extract_policy(w, tau=1, verbose=False):
    """
    Extract optimal stationary policy under subsidy w.
    """
    # Build base model (no subsidy inside)
    P, C_mat = build_model(tau)

    # Run RVI with subsidy w
    V, policy, Q, rho, iterations = relative_value_iteration(
        P=P,
        C=C_mat,
        w=w,
        ref_idx=ref_idx,
        verbose=verbose
    )

    policy_table = pd.DataFrame({
        "state": states,
        "optimal_action": policy,
        "inventory": [s[0] for s in states],
        "a_prev": [s[1] for s in states]
    })

    return policy_table, V, Q, rho





w_values = [-5, -2, 0, 2, 5]

for w in w_values:
    policy_table, _, _, rho = extract_policy(w)
    passive_states = policy_table[policy_table["optimal_action"] == 0]
    print(f"w={w}, #passive={len(passive_states)}, gain≈{rho:.4f}")







def passive_by_inventory(policy_table):
    inventories = sorted(policy_table["inventory"].unique())
    passive_frac = []

    for i in inventories:
        subset = policy_table[policy_table["inventory"] == i]
        passive_frac.append(np.mean(subset["optimal_action"] == 0))

    return np.array(inventories), np.array(passive_frac)


w_values = [-5, -2, 0, 2, 5]

for w in w_values:
    policy_table, _, _, rho = extract_policy(w)

    inv, passive_frac = passive_by_inventory(policy_table)

    plt.figure(figsize=(6, 4))
    plt.step(inv, passive_frac, where="post")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    plt.xlabel("Inventory level i")
    plt.ylabel("Fraction passive")
    plt.title(f"Passive region vs inventory (w = {w}, gain ≈ {rho:.2f})")

    # plt.show()


# Scanning over w (grid)
def scan_over_w(w_values, tau=1.0, tol=1e-8, verbose=False, warm_start=True):
    """
    For each w compute RVI and record Pb(w) (states where passive is optimal).
    Optionally warm-start V from previous w to accelerate convergence.
    """
    Pb_by_w = {}
    policy_by_w = {}
    Q_by_w = {}
    rho_by_w = {}

    V_warm = None

    # Build model 
    P, C_mat = build_model(tau)

    for w in sorted(w_values):
        V_init = V_warm if (warm_start and V_warm is not None) else None

        V, policy, Q, rho, iters = relative_value_iteration(
            P=P,
            C=C_mat,
            w=w,
            ref_idx=ref_idx,
            tol=tol,
            max_iter=20000,
            verbose=verbose,
            V_init=V_init
        )

        Pb = {si for si in range(nS) if policy[si] == 0}

        Pb_by_w[w] = Pb
        policy_by_w[w] = policy.copy()
        Q_by_w[w] = Q.copy()
        rho_by_w[w] = rho
        V_warm = V.copy()

        if verbose:
            print(f"w={w:.3f}, iters={iters}, rho={rho:.6f}, |Pb|={len(Pb)}")

    return Pb_by_w, policy_by_w, Q_by_w, rho_by_w



# To check indexability, for what value of w, Q(i,a_(t-1)=0, a_t=0) = Q(i,a_(t-1)=0, a_t=1) and Q(i,a_(t-1)=1, a_t=0) = Q(i,a_(t-1)=1, a_t=1)
w_values = np.linspace(-2, 2, 11)
Pb_by_w, policy_by_w, Q_by_w, rho_by_w = scan_over_w(w_values)

print(f"Scanned Pb(w) for w in {w_values}")

# for i, w_i in enumerate(w_values):
#     print(f"\n----- For w = {w_i} (idx {i}) -----")
#     print(f"Pb({w_i}): {Pb_by_w[w_i]}")
#     print(f"policy at w={w_i}:\n{policy_by_w[w_i]}")
#     print(f"rho at w={w_i}: {rho_by_w[w_i]}")
#     print(f"Q at w={w_i}:\n{Q_by_w[w_i]}")



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


is_indexable, violations = check_monotonicity(Pb_by_w, w_values)
print("Indexable?", is_indexable)
print("Violations:", violations)



def compute_whittle_indices_from_grid(Pb_by_w, w_values):
    """
    For each state si, compute W(si) = inf { w : si in Pb(w) } using the scanned w_values
    If a state never becomes passive in the scanned range, set to np.inf
    Returns dict si -> W
    """
    whittle = {}
    sorted_w = sorted(w_values)
    for si in range(nS):
        found = False
        for w in sorted_w:
            if si in Pb_by_w[w]:
                whittle[si] = w
                found = True
                break
        if not found:
            whittle[si] = np.inf
    return whittle


# Compute Whittle indices (grid-based fallback)
whittle_state_grid = compute_whittle_indices_from_grid(Pb_by_w, w_values)
whittle_readable_grid = {states[si]: whittle_state_grid[si] for si in range(nS)}
print("\nWhittle indices per state (grid approximation):")
for s, wval in sorted(whittle_readable_grid.items()):
    print(f"  State {s}: W_grid = {wval}")






# Finite-Horizon Relative DP
def finite_horizon_relative(P, C_base, T, ref_idx, minimize=True):
    """
    Perform finite-horizon relative dynamic programming.

    Parameters
    ----------
    P : dict
        Transition probabilities, P[(state_idx, action)] -> {next_state_idx: prob}
    C_base : ndarray
        Cost matrix, shape (nS, nA)
    T : int
        Horizon length
    ref_idx : int
        Reference state for relative normalization
    minimize : bool
        True for cost minimization, False for reward maximization

    Returns
    -------
    results : list of dicts
        Each dict contains:
            't' : time step
            'g_t' : average incremental cost
            'V_t' : value function at step t
            'V_tilde' : relative normalized value
            'policy_t' : optimal policy at step t
    """
    nS = len(states)
    nA = len(actions)
    V_prev = np.zeros(nS)  # initialize value function
    
    # # Apply subsidy
    # C_sub = C_base.copy()
    # C_sub[:, 1] += w


    results = []

    for t in range(1, T + 1):
        Q = np.full((nS, nA), np.inf)  # Q-matrix for current step

        # Compute one-step Q-values
        for si in range(nS):
            for a in actions:
                Q[si, a] = C_base[si, a] + sum(p * V_prev[sj] for sj, p in P[(si, a)].items())

        # Compute optimal value and policy
        if minimize:
            V_t = np.min(Q, axis=1)
            policy_t = np.argmin(Q, axis=1)
        else:
            V_t = np.max(Q, axis=1)
            policy_t = np.argmax(Q, axis=1)

        # Compute incremental cost (like rho in infinite-horizon RVI)
        g_t = V_t[ref_idx] - V_prev[ref_idx]

        # Relative normalization (span semi-norm idea)
        V_tilde = V_t - t * g_t

        # Store results
        results.append({
            "t": t,
            "g_t": g_t,
            "V_t": V_t.copy(),
            "V_tilde": V_tilde.copy(),
            "policy_t": policy_t.copy()
        })

        # Prepare for next step
        V_prev = V_t.copy()

    return results



def compare_rvi_vs_finite_horizon(P_base, C_base, w, ref_idx, T, verbose=True):
    """
    Compare RVI (infinite-horizon) vs Finite-Horizon Relative DP (fixed & corrected).
    
    Fixes:
      - Applies subsidy w to passive action in both RVI and finite-horizon DP.
      - Computes finite-horizon Q-values consistently.
      - Uses reference-state normalization like RVI.
    
    Returns
    -------
    comparison_df : DataFrame
        State-by-state comparison of policies and Q-values.
    summary : dict
        Summary statistics.
    """
    nS = len(states)
    
    # --- Apply subsidy to passive action ---
    C_sub = C_base.copy()
    C_sub[:, 0] -= w
    
    # --- RVI (infinite-horizon) ---
    V_inf, policy_inf, Q_inf, rho_inf, iter_inf = relative_value_iteration(
        P_base, C_sub, w=0.0, ref_idx=ref_idx, verbose=False
    )
    
    # --- Finite-Horizon DP ---
    results_fh = finite_horizon_relative(P_base, C_sub, T, ref_idx, minimize=True)
    
    # Extract last horizon value function
    V_last = results_fh[-1]['V_t']
    
    # Recompute Q-values at last step consistently
    Q_fh = np.zeros((nS, 2))
    for si in range(nS):
        for a in actions:
            Q_fh[si, a] = C_sub[si, a] + sum(p * V_last[sj] for sj, p in P_base[(si, a)].items())
    
    # Compute finite-horizon policy from last step Q
    policy_fh = np.argmin(Q_fh, axis=1)
    
    # Compute incremental cost at horizon T
    g_fh = V_last[ref_idx] - results_fh[-2]['V_t'][ref_idx] if T > 1 else V_last[ref_idx]
    
    # --- Create comparison DataFrame ---
    comparison_data = []
    for si in range(nS):
        s = states[si]
        i, ap = s
        
        policy_match = "✓" if policy_inf[si] == policy_fh[si] else "✗"
        
        comparison_data.append({
            "state": s,
            "i": i,
            "prev_action": ap,
            "policy_RVI": policy_inf[si],
            "policy_FH": policy_fh[si],
            "match": policy_match,
            "Q_RVI_passive": Q_inf[si, 0],
            "Q_RVI_active": Q_inf[si, 1],
            "Q_FH_passive": Q_fh[si, 0],
            "Q_FH_active": Q_fh[si, 1],
            "V_RVI": V_inf[si],
            "V_FH": V_last[si],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # --- Summary statistics ---
    policy_agreement = np.mean(policy_inf == policy_fh) * 100
    policy_diffs = np.sum(policy_inf != policy_fh)
    
    summary = {
        "w": w,
        "T_horizon": T,
        "rho_RVI (avg cost)": rho_inf,
        "g_T_FH": g_fh,
        "avg_cost_diff": abs(rho_inf - g_fh),
        "policy_agreement_%": policy_agreement,
        "policy_disagreements": policy_diffs,
        "RVI_iterations": iter_inf,
        "passive_states_RVI": np.sum(policy_inf == 0),
        "passive_states_FH": np.sum(policy_fh == 0),
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"COMPARISON: RVI vs Finite-Horizon DP (T={T})")
        print(f"{'='*80}")
        print(f"Subsidy w = {w}")
        print(f"\nAverage Cost (long-term):")
        print(f"  RVI (infinite-horizon):    rho = {rho_inf:.6f}")
        print(f"  Finite-Horizon (T={T}):     g_T = {g_fh:.6f}")
        print(f"  Difference:                 {abs(rho_inf - g_fh):.6f}")
        print(f"\nPolicy Comparison:")
        print(f"  Agreement:                  {policy_agreement:.2f}%")
        print(f"  Disagreements:              {policy_diffs} states")
        print(f"  Passive states (RVI):       {np.sum(policy_inf == 0)} / {nS}")
        print(f"  Passive states (FH):        {np.sum(policy_fh == 0)} / {nS}")
        print(f"\nAlgorithm Stats:")
        print(f"  RVI iterations:             {iter_inf}")
        print(f"\n{'State':<15} {'RVI':<8} {'FH':<8} {'Match':<8} {'Q_RVI_0':<12} {'Q_RVI_1':<12} {'Q_FH_0':<12} {'Q_FH_1':<12}")
        print(f"{'-'*100}")
        for _, row in comparison_df.iterrows():
            print(f"{str(row['state']):<15} {row['policy_RVI']:<8} {row['policy_FH']:<8} "
                  f"{row['match']:<8} {row['Q_RVI_passive']:<12.4f} {row['Q_RVI_active']:<12.4f} "
                  f"{row['Q_FH_passive']:<12.4f} {row['Q_FH_active']:<12.4f}")
    
    return comparison_df, summary


# Build model
P_base, C_base = build_model(tau=1.0)

# Test at different subsidies
test_w_values = [-1000, -2.0, -1.8, -1.6, 0.0, 0.2, 0.4, 0.5, 1.0, 1.2, 1.4, 2.0, 1000]
all_summaries = []

for w_test in test_w_values:
    comparison_df, summary = compare_rvi_vs_finite_horizon(P_base, C_base, w_test, ref_idx, T=50)
    all_summaries.append(summary)

# # Summary table
# print(f"\n\n{'='*80}")
# print("SUMMARY ACROSS DIFFERENT SUBSIDIES")
# print(f"{'='*80}")
# summary_table = pd.DataFrame(all_summaries)
# print(summary_table.to_string(index=False))





# Using Packages
def extract_P0_P1_R0_R1(tau=1.0):
    """
    Build P0, P1, R0, R1 for markovianbandit.
    State space = (inventory, a_prev).
    Switching cost = 0.
    Rewards = -costs.
    """

    # Build base (no subsidy)
    P_dict, C_mat = build_model(tau)

    P0 = np.zeros((nS, nS))
    P1 = np.zeros((nS, nS))
    R0 = np.zeros(nS)
    R1 = np.zeros(nS)

    for si in range(nS):
        # passive
        for sj, prob in P_dict[(si, 0)].items():
            P0[si, sj] = prob
        R0[si] = -C_mat[si, 0]   # reward = -cost

        # active
        for sj, prob in P_dict[(si, 1)].items():
            P1[si, sj] = prob
        R1[si] = -C_mat[si, 1]

    # normalize (CTMC discretization can introduce tiny drift)
    for i in range(nS):
        if P0[i].sum() > 0:
            P0[i] /= P0[i].sum()
        if P1[i].sum() > 0:
            P1[i] /= P1[i].sum()

    print("Shapes:", P0.shape, P1.shape, R0.shape, R1.shape)
    return P0, P1, R0, R1


# --- PACKAGE COMPARISON BLOCK  ---
P0, P1, R0, R1 = extract_P0_P1_R0_R1( )
model_pkg = bandit.restless_bandit_from_P0P1_R0R1(
        P0=P0, P1=P1, R0=R0, R1=R1
    )
print("Whittle indices (package):")
pkg_indices = model_pkg.whittle_indices()
print(pkg_indices)

pkg_indices_dict = {i: pkg_indices[i] for i in range(len(pkg_indices))}




def whittle_index_bisection(state_idx,
                            w_low=-10.0,
                            w_high=10.0,
                            tol_w=1e-6,
                            max_iter=40,
                            tau=1.0,
                            rvi_tol=1e-8,
                            rvi_max_iter=5000,
                            expansion_factor=2.0,
                            max_expansions=6):
    """
    Compute Whittle index for a specific state using bisection with automatic bracket expansion.
    
    Parameters
    ----------
    state_idx : int
        Index of the state for which Whittle index is computed.
    w_low, w_high : float
        Initial subsidy bracket.
    tol_w : float
        Tolerance for convergence of Whittle index.
    max_iter : int
        Maximum bisection iterations.
    tau : float
        Time step for CTMC approximation.
    rvi_tol, rvi_max_iter : float, int
        Tolerance and max iterations for RVI solver.
    expansion_factor : float
        Factor by which to expand bracket if both endpoints give same action.
    max_expansions : int
        Maximum number of progressive expansions to find a valid bracket.

    Returns
    -------
    W : float
        Whittle index for the given state. np.inf if bracket cannot be found.
    """

    # Build base model once (no w applied here)
    P, C_mat = build_model(tau)

    def optimal_action_at(w, V_init=None):
        V, policy, _, _, _ = relative_value_iteration(
            P=P,
            C=C_mat,
            w=w,
            ref_idx=ref_idx,
            tol=rvi_tol,
            max_iter=rvi_max_iter,
            V_init=V_init
        )
        return int(policy[state_idx]), V

    # Evaluate initial endpoints
    opt_low, V_low = optimal_action_at(w_low)
    opt_high, V_high = optimal_action_at(w_high)

    # Automatic bracket expansion if needed
    expansions = 0
    while opt_low == opt_high and expansions < max_expansions:
        w_low *= expansion_factor
        w_high *= expansion_factor
        opt_low, V_low = optimal_action_at(w_low)
        opt_high, V_high = optimal_action_at(w_high)
        expansions += 1

    if opt_low == opt_high:
        # Could not find valid bracket
        return np.inf

    # Bisection search with warm-start
    low, high = w_low, w_high
    V_warm = None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        best_action, V_mid = optimal_action_at(mid, V_warm)
        V_warm = V_mid.copy()

        if best_action == 0:
            # Passive optimal → index ≤ mid
            high = mid
        else:
            # Active optimal → index ≥ mid
            low = mid

        if abs(high - low) < tol_w:
            break

    return 0.5 * (low + high)


def compute_whittle_indices_bisection(w_low, w_high, tol_w=1e-6, tau=1.0):
    """
    Compute Whittle indices for all states using bisection method.

    Returns
    -------
    whittle : dict
        state_idx -> Whittle index
    """
    whittle = {}
    for s_idx in range(nS):
        W = whittle_index_bisection(state_idx=s_idx,
                                    w_low=w_low,
                                    w_high=w_high,
                                    tol_w=tol_w,
                                    tau=tau)
        whittle[s_idx] = W
        # print(f"State {states[s_idx]} → Whittle index = {W}")
    return whittle



print("\n--- Computing continuous Whittle indices with bisection ---")
whittle_state_cont = compute_whittle_indices_bisection(
    w_low=-10, w_high=10, tol_w=1e-6, tau=tau
)

# Map to readable states
whittle_readable_cont = {states[si]: whittle_state_cont[si] for si in range(nS)}
for s, wval in sorted(whittle_readable_cont.items()):
    print(f"  State {s}: Whittle index = {wval}")






# Newton Raphson-based Whittle index (continuous)
def whittle_index_newton(state_idx,
                         w_init=0.0,
                         tol_w=1e-6,
                         max_iter=40,
                         tau=1.0,
                         rvi_tol=1e-8,
                         rvi_max_iter=5000,
                         eps=1e-4):
    """
    Compute Whittle index for a specific state using Newton-Raphson method.
    g(w) = Q_passive(x; w) - Q_active(x; w)
    """

    # Build base model once (no subsidy)
    P, C_mat = build_model(tau)
    w = w_init
    V_warm = None

    for it in range(max_iter):
        # 1. Compute g(w)
        V, policy, Q, rho, _ = relative_value_iteration(
            P=P,
            C=C_mat,
            w=w,
            ref_idx=ref_idx,
            tol=rvi_tol,
            max_iter=rvi_max_iter,
            V_init=V_warm
        )
        g = Q[state_idx, 0] - Q[state_idx, 1]

        # 2. Compute finite-difference derivative g'(w)
        V_eps, _, Q_eps, _, _ = relative_value_iteration(
            P=P,
            C=C_mat,
            w=w + eps,
            ref_idx=ref_idx,
            tol=rvi_tol,
            max_iter=rvi_max_iter,
            V_init=V
        )
        V_epsm, _, Q_epsm, _, _ = relative_value_iteration(
            P=P,
            C=C_mat,
            w=w - eps,
            ref_idx=ref_idx,
            tol=rvi_tol,
            max_iter=rvi_max_iter,
            V_init=V
        )
        g_prime = (Q_eps[state_idx, 0] - Q_eps[state_idx, 1] -
                   (Q_epsm[state_idx, 0] - Q_epsm[state_idx, 1])) / (2 * eps)

        # Avoid division by zero
        if abs(g_prime) < 1e-12:
            print(f"Derivative too small at iteration {it}, stopping.")
            break

        # 3. Newton-Raphson update
        w_new = w - g / g_prime

        if abs(w_new - w) < tol_w:
            w = w_new
            break

        w = w_new
        V_warm = V.copy()  # warm-start next iteration

    return w


def compute_whittle_indices_newton(w_init=0.0, tol_w=1e-6, tau=1.0):
    """
    Compute Whittle indices for all states using Newton-Raphson method.
    """
    whittle = {}
    for s_idx in range(nS):
        W = whittle_index_newton(state_idx=s_idx, w_init=w_init, tol_w=tol_w, tau=tau)
        whittle[s_idx] = W
        # print(f"State {states[s_idx]} → Whittle index = {W}")
    return whittle


print("\n--- Computing Whittle indices using Newton-Raphson ---")
whittle_state_newton = compute_whittle_indices_newton(w_init=0.0, tol_w=1e-6, tau=tau)
# Map to readable states
whittle_readable_newton = {states[si]: whittle_state_newton[si] for si in range(nS)}
for s, wval in sorted(whittle_readable_newton.items()):
    print(f"  State {s}: Whittle index = {wval}")   





def run_single_experiment(w_scan=np.linspace(-2, 2, 11)):
    # Build model
    P_base, C_base = build_model(tau)

    # Scan for indexability
    Pb_by_w, policy_by_w, Q_by_w, rho_by_w = scan_over_w(w_scan)
    is_indexable, violations = check_monotonicity(Pb_by_w, w_scan)

    # Grid Whittle
    whittle_grid = compute_whittle_indices_from_grid(Pb_by_w, w_scan)

    # Bisection Whittle
    whittle_bisect = compute_whittle_indices_bisection(
        w_low=-10, w_high=10, tol_w=1e-6, tau=tau
    )

    # Package Whittle
    P0, P1, R0, R1 = extract_P0_P1_R0_R1(tau)
    model_pkg = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    whittle_pkg = model_pkg.whittle_indices()

    return {
        "indexable": is_indexable,
        "violations": violations,
        "rho_by_w": rho_by_w,
        "whittle_grid": whittle_grid,
        "whittle_bisect": whittle_bisect,
        "whittle_pkg": whittle_pkg
    }



def write_experiment_to_txt(f, params, results):
    f.write("="*90 + "\n")
    f.write("PARAMETERS\n")
    f.write("-"*90 + "\n")
    for k, v in params.items():
        f.write(f"{k:15s}: {v}\n")

    f.write("\nINDEXABILITY\n")
    f.write("-"*90 + "\n")
    f.write(f"Indexable: {results['indexable']}\n")
    if not results["indexable"]:
        f.write(f"Violations: {results['violations']}\n")

    f.write("\nAVERAGE COST (rho) vs w\n")
    f.write("-"*90 + "\n")
    for w, rho in results["rho_by_w"].items():
        f.write(f"w={w:6.3f}  rho={rho:10.6f}\n")

    f.write("\nWHITTLE INDICES (Grid)\n")
    f.write("-"*90 + "\n")
    for si, wval in results["whittle_grid"].items():
        f.write(f"State {states[si]} : {wval}\n")

    f.write("\nWHITTLE INDICES (Bisection)\n")
    f.write("-"*90 + "\n")
    for si, wval in results["whittle_bisect"].items():
        f.write(f"State {states[si]} : {wval}\n")

    f.write("\nWHITTLE INDICES (Package)\n")
    f.write("-"*90 + "\n")
    for si, wval in enumerate(results["whittle_pkg"]):
        f.write(f"State {states[si]} : {wval}\n")

    f.write("\n\n")



output_file = "whittle_parameter_sweep_report.txt"

with open(output_file, "w") as f:
    exp_id = 0

    for (h_, p_, l_, gamma_, lam_, switch_) in product(
        h_values, p_values, l_values, gamma_values, lam_values, switch_cost_values
    ):
        exp_id += 1
        print(f"Running experiment {exp_id}...")

        set_model_parameters(h_, p_, l_, gamma_, lam_, switch_)

        params = {
            "h": h_,
            "p": p_,
            "l": l_,
            "gamma": gamma_,
            "lambda": lam_,
            "mu": mu,
            "switch_cost": switch_,
            "tau": tau,
            "max_inventory": max_inventory
        }

        results = run_single_experiment()
        write_experiment_to_txt(f, params, results)

print(f"\nAll experiments saved to: {output_file}")

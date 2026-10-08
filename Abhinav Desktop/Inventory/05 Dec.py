import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import datetime
import markovianbandit as bandit  
import os

# -------------------------
# Model parameters
# -------------------------
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

# -------------------------
# Cost parameters (tune to your modeling intent)
# -------------------------
h = 1.0            # holding cost per unit inventory per step
p = 2.0            # perish penalty scale (per unit lost item)
l = 5.0            # lost-sales penalty per lost demand occurrence
gamma = 0.1        # perishability rate (per-unit per time step)
lam = 0.6          # demand rate
mu = 0.8           # production rate when active
switch_cost = 3.0  # symmetric switching cost when action != a_prev

# -------------------------
# Transition model (CTMC-like discrete step)
# -------------------------
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

# quick sanity prints (can remove if noisy)
print("Example transitions from state (5,0) with action=1:", transitions_ctmc((5,0), 1, tau=1.0))
print("Example transitions from state (5,0) with action=0:", transitions_ctmc((5,0), 0, tau=1.0))

# -------------------------
# Cost primitives
# -------------------------
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

def one_step_cost(state, action, w):
    i, a_prev = state
    base = C_base(i, a_prev, w)
    if a_prev != action:
        base += C_switch(i)
    return base

# -------------------------
# Build model: P and c
# -------------------------
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

# -------------------------
# Relative Value Iteration (RVI) with optional warm-start
# -------------------------
def relative_value_iteration(P, c, ref_idx, tol=1e-8, max_iter=20000, verbose=False, V_init=None):
    """
    Solve average-cost Bellman via RVI.
    - P: dict (si,a) -> {sj: prob}
    - c: dict (si,a) -> scalar cost
    - ref_idx: index to normalize V[ref_idx] = 0
    - V_init: optional initial guess for V (warm start)
    Returns: V, policy, Q, rho, iterations
    """
    nS = len(states)
    if V_init is None:
        V = np.zeros(nS)
    else:
        # ensure shape match
        V = np.array(V_init, dtype=float).copy()
        if V.shape[0] != nS:
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

# -------------------------
# Finite-horizon relative DP
# -------------------------
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

# Scanning over w (grid) with warm-start
def scan_over_w(w_values, tau=1.0, tol=1e-8, verbose=False, warm_start=True):
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

        # pass warm-start if available
        V_init = V_warm if (warm_start and V_warm is not None) else None
        V, policy, Q, rho, iters = relative_value_iteration(P, c, ref_idx, tol=tol, max_iter=20000, verbose=verbose, V_init=V_init)

        Pb = set(si for si in range(nS) if policy[si] == 0)
        Pb_by_w[w] = Pb
        policy_by_w[w] = policy.copy()
        Q_by_w[w] = Q.copy()
        rho_by_w[w] = rho
        V_warm = V.copy()
        if verbose:
            print(f"w={w:.4f}  iters={iters}  rho~={rho:.6f}  |Pb|={len(Pb)}")

    return Pb_by_w, policy_by_w, Q_by_w, rho_by_w

# -------------------------
# Monotonicity checks
# -------------------------
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

def check_delta_monotonicity(Q_by_w, w_values, tol=1e-10):
    """
    Checks if advantage Δ(s,w) = Q_passive - Q_active is monotone increasing in w.
    Returns:
        is_indexable: bool
        violations: list of (state, w1, w2, delta1, delta2)
    """
    sorted_w = sorted(w_values)
    is_indexable = True
    violations = []

    # number of states inferred from a sample Q
    sample_Q = next(iter(Q_by_w.values()))
    n_states = sample_Q.shape[0]

    for s in range(n_states):
        for i in range(len(sorted_w)-1):
            w1 = sorted_w[i]
            w2 = sorted_w[i+1]

            Q1 = Q_by_w[w1][s]
            Q2 = Q_by_w[w2][s]

            # Q[:,0] => passive, Q[:,1] => active (by our convention)
            delta1 = Q1[0] - Q1[1]   # passive - active
            delta2 = Q2[0] - Q2[1]

            if delta1 > delta2 + tol:  # violation of monotonicity
                is_indexable = False
                violations.append((s, w1, w2, delta1, delta2))

    return is_indexable, violations

def full_indexability_check(Pb_by_w, Q_by_w, w_values, tol=1e-10):
    pb_ok, pb_viol = check_monotonicity(Pb_by_w, w_values)
    delta_ok, delta_viol = check_delta_monotonicity(Q_by_w, w_values, tol=tol)

    if pb_ok and delta_ok:
        print("✓ Model is indexable on the grid and Δ is monotone (strong check).")
    else:
        print("✗ Indexability violated.")
        if not pb_ok:
            print("  - Pb subset violations (showing up to 20):")
            for v in pb_viol[:20]:
                w1, w2, missing = v
                missing_readable = [states[si] for si in sorted(list(missing))]
                print(f"    between {w1:.4f} -> {w2:.4f}: {missing_readable}")
        if not delta_ok:
            print("  - Δ monotonicity violations (showing up to 20):")
            for v in delta_viol[:20]:
                s, w1, w2, d1, d2 = v
                print(f"    state {states[s]} (idx {s}): Δ({w1:.4f})={d1:.6e} > Δ({w2:.4f})={d2:.6e}")

    return pb_ok, pb_viol, delta_ok, delta_viol

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

# -------------------------
# Plotting Pb(w)
# -------------------------
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
    cmap = ListedColormap([cmap_colors[0], cmap_colors[1]])  # 0: Active (0), 1: Passive (1)
    im = plt.pcolormesh(X, Y, grid, cmap=cmap, shading='flat')
    cbar = plt.colorbar(im, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Active (0)', 'Passive (1)'])
    plt.title('Passive Set $P_b(w)$ vs. Subsidy $w$')
    plt.xlabel('Subsidy $w$')
    plt.ylabel('State Index')
    plt.tight_layout()
    plt.show()

# -------------------------
# Bisection-based Whittle index (continuous)
# -------------------------
def whittle_index_bisection(state_idx,
                            w_low=-10.0,
                            w_high=10.0,
                            tol_w=1e-6,
                            max_iter=40,
                            tau=1.0,
                            rvi_tol=1e-8,
                            rvi_max_iter=5000,
                            verbose=False):
    """
    Compute Whittle index for a specific state using bisection.
    We assume indexability (Δ increasing in w). Warm-start RVI across iterations using last V.
    Returns midpoint when interval width < tol_w or np.inf if bracket cannot be found.
    """
    # Evaluate at endpoints to see if they are on different sides
    P_low, c_low = build_model(w_low, tau=tau)
    V_low, policy_low, Q_low, rho_low, _ = relative_value_iteration(P_low, c_low, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter)
    opt_low = int(policy_low[state_idx])
    if verbose:
        print(f"w_low={w_low}, optimal action={opt_low}, Δ={Q_low[state_idx,0]-Q_low[state_idx,1]:.6e}")

    P_high, c_high = build_model(w_high, tau=tau)
    V_high, policy_high, Q_high, rho_high, _ = relative_value_iteration(P_high, c_high, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter)
    opt_high = int(policy_high[state_idx])
    if verbose:
        print(f"w_high={w_high}, optimal action={opt_high}, Δ={Q_high[state_idx,0]-Q_high[state_idx,1]:.6e}")

    # If both endpoints result in same optimal action, we try to expand the bracket a bit
    if opt_low == opt_high:
        # expand outwards progressively
        factor = 2.0
        expanded = False
        for _ in range(6):
            w_low = w_low * factor
            w_high = w_high * factor
            P_low, c_low = build_model(w_low, tau=tau)
            V_low, policy_low, Q_low, rho_low, _ = relative_value_iteration(P_low, c_low, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter)
            P_high, c_high = build_model(w_high, tau=tau)
            V_high, policy_high, Q_high, rho_high, _ = relative_value_iteration(P_high, c_high, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter)
            if int(policy_low[state_idx]) != int(policy_high[state_idx]):
                expanded = True
                break
        if not expanded:
            # No bracket found; return np.inf indicating index not found in reasonable range
            if verbose:
                print("Could not find a bracket with differing optimal actions; returning inf")
            return np.inf

    # Now run bisection. Warm-start using last V computed for midpoints.
    V_warm = None
    low = w_low
    high = w_high

    for it in range(max_iter):
        mid = 0.5 * (low + high)
        P_mid, c_mid = build_model(mid, tau=tau)
        V_mid, policy_mid, Q_mid, rho_mid, _ = relative_value_iteration(P_mid, c_mid, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter, V_init=V_warm)
        best_action = int(policy_mid[state_idx])
        delta_mid = Q_mid[state_idx, 0] - Q_mid[state_idx, 1]

        if verbose:
            print(f"  it={it} mid={mid:.6f} action={best_action} Δ={delta_mid:.6e} interval=({low:.6e},{high:.6e})")

        # g(w)=Δ(w) is increasing with w; root w* s.t. g(w*) = 0
        # If at mid passive is optimal (Δ>0) => root is <= mid => move high = mid
        if best_action == 0:
            high = mid
        else:
            # active optimal (Δ<0) => root is >= mid => move low = mid
            low = mid

        V_warm = V_mid.copy()
        if abs(high - low) < tol_w:
            break

    return 0.5 * (low + high)

def compute_whittle_indices_bisection(w_low=-10, w_high=10, tol_w=1e-6, tau=1.0, verbose=False):
    whittle = {}
    for s_idx in range(nS):
        W = whittle_index_bisection(state_idx=s_idx, w_low=w_low, w_high=w_high, tol_w=tol_w, tau=tau, verbose=verbose)
        whittle[s_idx] = W
        print(f"State {states[s_idx]} → Whittle index = {W}")
    return whittle

# Newton Raphson-based Whittle index (continuous)
def whittle_index_newton(state_idx,
                         w_init=0.0,
                         tol_w=1e-6,
                         max_iter=40,
                         tau=1.0,
                         rvi_tol=1e-8,
                         rvi_max_iter=5000,
                         eps=1e-4,
                         verbose=False):
    """
    Compute Whittle index for a specific state using Newton-Raphson.
    g(w) = Q_passive(x; w) - Q_active(x; w)
    """
    w = float(w_init)
    V_warm = None

    for it in range(max_iter):
        # Compute g(w)
        P, c = build_model(w, tau=tau)
        V, policy, Q, rho, _ = relative_value_iteration(P, c, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter, V_init=V_warm)
        Q_pass = Q[state_idx, 0]
        Q_act  = Q[state_idx, 1]
        g = Q_pass - Q_act

        # finite-difference derivative
        P_eps, c_eps = build_model(w + eps, tau=tau)
        V_eps, policy_eps, Q_eps, rho_eps, _ = relative_value_iteration(P_eps, c_eps, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter, V_init=V)
        g_eps = Q_eps[state_idx, 0] - Q_eps[state_idx, 1]

        P_epsm, c_epsm = build_model(w - eps, tau=tau)
        V_epsm, policy_epsm, Q_epsm, rho_epsm, _ = relative_value_iteration(P_epsm, c_epsm, ref_idx, tol=rvi_tol, max_iter=rvi_max_iter, V_init=V)
        g_epsm = Q_epsm[state_idx, 0] - Q_epsm[state_idx, 1]

        g_prime = (g_eps - g_epsm) / (2 * eps)

        if abs(g_prime) < 1e-12:  # avoid division by zero
            if verbose:
                print("Derivative approx ~0; stopping Newton for state", state_idx)
            break

        # Newton-Raphson step
        w_new = w - g / g_prime

        if verbose:
            print(f"Newton it={it}: w={w:.6f}, g={g:.6e}, g'={g_prime:.6e}, w_new={w_new:.6f}")

        if abs(w_new - w) < tol_w:
            w = w_new
            break

        w = w_new
        V_warm = V.copy()  # warm-start next iteration

    return w

def compute_whittle_indices_newton(w_init=0.0, tol_w=1e-6, tau=1.0, verbose=False):
    whittle = {}
    for s_idx in range(nS):
        W = whittle_index_newton(state_idx=s_idx, w_init=w_init, tol_w=tol_w, tau=tau, verbose=verbose)
        whittle[s_idx] = W
        print(f"State {states[s_idx]} → Whittle index = {W}")
    return whittle

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # choose w grid for scan
    w_values = np.linspace(-2.0, 2.0, 41)  # moderate resolution; adjust as needed
    tau = 1.0
    # For exploratory scans, relax tol to speed up
    Pb_by_w, policy_by_w, Q_by_w, rho_by_w = scan_over_w(w_values, tau=tau, tol=1e-6, verbose=True, warm_start=True)

    # Debug prints for the first grid point
    first_w = sorted(w_values)[0]
    print("Scanned Pb(w) for w in", w_values)
    print("Example Pb(first w):", Pb_by_w[first_w])
    print("policy at first w:\n", policy_by_w[first_w])
    print("rho at first w:", rho_by_w[first_w])
    print("Q at first w:\n", Q_by_w[first_w])

    # Check indexability (both weak and strong)
    pb_ok, pb_viol, delta_ok, delta_viol = full_indexability_check(Pb_by_w, Q_by_w, w_values, tol=1e-10)

    # Compute Whittle indices (grid-based fallback)
    whittle_state_grid = compute_whittle_indices_from_grid(Pb_by_w, w_values)
    whittle_readable_grid = {states[si]: whittle_state_grid[si] for si in range(nS)}
    print("\nWhittle indices per state (grid approximation):")
    for s, wval in sorted(whittle_readable_grid.items()):
        print(f"  State {s}: W_grid = {wval}")

    # Write grid-based indices to CSV
    csv_grid_path = "whittle_grid.csv"
    with open(csv_grid_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "W_grid"])
        for si in range(nS):
            writer.writerow([states[si], whittle_state_grid[si]])
    print("Written grid-based Whittle indices to", csv_grid_path)

    # Compute continuous Whittle indices using bisection (only if indexable)
    print("\n--- Computing continuous Whittle indices with bisection ---")
    whittle_state_cont = compute_whittle_indices_bisection(w_low=-10, w_high=10, tol_w=1e-6, tau=tau, verbose=False)
    whittle_readable_cont = {states[si]: whittle_state_cont[si] for si in range(nS)}
    print("\nWhittle index per state (continuous, bisection):")
    for s, wval in sorted(whittle_readable_cont.items()):
        print(f"  State {s}: W = {wval}")

    # Write continuous (bisection) indices to CSV
    csv_cont_path = "whittle_bisection.csv"
    with open(csv_cont_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "W_bisection"])
        for si in range(nS):
            writer.writerow([states[si], whittle_state_cont[si]])
    print("Written bisection Whittle indices to", csv_cont_path)

    # Compute continuous Whittle indices using Newton-Raphson
    print("\n--- Computing continuous Whittle indices with Newton-Raphson ---")
    whittle_state_newton = compute_whittle_indices_newton(w_init=0.0, tol_w=1e-6, tau=tau, verbose=False)
    whittle_readable_newton = {states[si]: whittle_state_newton[si] for si in range(nS)}

    print("\nWhittle index per state (continuous, Newton-Raphson):")
    for s, wval in sorted(whittle_readable_newton.items()):
        print(f"  State {s}: W = {wval}")

    # Write Newton indices to CSV
    csv_newt_path = "whittle_newton.csv"
    with open(csv_newt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "W_newton"])
        for si in range(nS):
            writer.writerow([states[si], whittle_state_newton[si]])
    print("Written Newton Whittle indices to", csv_newt_path)

    # Plot passive sets (grid)
    plot_passive_sets(Pb_by_w, nS)

    # -------------------------
    # markovianbandit usage (commented placeholder)
    # -------------------------
    # The following is a placeholder example. Uncomment and fill proper P0,P1,R0,R1 or appropriate args
    # if you have the 'markovianbandit' package and correct matrices.
    #
    # model_restart = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    # print(model_restart.whittle_indices())
    # restart_whittle_indices = model_restart.whittle_indices()


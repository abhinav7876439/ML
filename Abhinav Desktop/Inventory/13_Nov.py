import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pprint import pprint

# -------------------- Model parameters --------------------
max_inventory = 15
inventory_levels = list(range(max_inventory + 1))
actions = [0, 1]           # 0 = passive, 1 = active
a_prev_values = [0, 1]
reference_state = (0, 1)   # used for normalization; must be in state space

# Cost parameters
h = 1.0            # holding cost
p = 2.0            # perish penalty
l = 5.0            # lost-sales penalty
gamma = 0.1        # perishability rate
lam = 0.6          # demand rate
mu = 0.8           # production rate
switch_cost = 3.0  # switching cost

# State space and index mapping
states = [(i, a_prev) for i in inventory_levels for a_prev in a_prev_values]
state_index = {s: idx for idx, s in enumerate(states)}
nS = len(states)
ref_idx = state_index[reference_state]

# -------------------- Transition & cost primitives --------------------
def get_rates(i, a):
    r_up = mu if a == 1 else 0.0
    r_down = lam + gamma * i
    return r_up, r_down

def transitions_ctmc(state, action, tau=1.0):
    """
    CTMC-inspired discrete-time transition approximation using tau
    Returns: dict(next_state_tuple -> probability)
    """
    i, a_prev = state
    trans = {}
    r_up, r_down = get_rates(i, action)
    R = r_up + r_down
    if R <= 1e-12:
        trans[(i, action)] = 1.0
        return trans
    p_event = 1 - np.exp(-R * tau)
    p_up = (r_up / R) * p_event if R > 0 else 0.0
    p_down = (r_down / R) * p_event if R > 0 else 0.0
    p_stay = 1.0 - p_event
    i_up = min(i + 1, max_inventory)
    i_down = max(i - 1, 0)
    trans[(i_up, action)] = trans.get((i_up, action), 0.0) + p_up
    trans[(i_down, action)] = trans.get((i_down, action), 0.0) + p_down
    trans[(i, action)] = trans.get((i, action), 0.0) + p_stay
    return trans

def C_base(i, a_prev, w):
    cost = h * i + p * gamma * i
    if i == 0:
        cost += l * lam
    if a_prev == 1:
        cost += w
    return cost

def C_switch(i):
    return switch_cost

def one_step_cost(state, action, w):
    i, a_prev = state
    base = C_base(i, a_prev, w)
    if a_prev != action:
        base += C_switch(i)
    return base

def build_model(w, tau=1.0):
    """
    Build P and c dictionaries keyed by (si, action),
      P[(si,a)] = dict(sj -> prob)
      c[(si,a)] = immediate cost (scalar)
    where si, sj are integer state indices.
    """
    P = {}
    c = {}
    for s in states:
        si = state_index[s]
        for a in actions:
            trans = transitions_ctmc(s, a, tau=tau)
            P[(si, a)] = { state_index[s2]: prob for s2, prob in trans.items() }
            c[(si, a)] = one_step_cost(s, a, w)
    return P, c

# -------------------- Finite-horizon relative DP recursion --------------------
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

# -------------------- Relative Value Iteration (RVI) --------------------
def relative_value_iteration(P, c, ref_idx, tol=1e-9, max_iter=20000):
    """
    RVI for average-cost MDP (minimization). Returns final:
       V (relative values, normalized so V[ref]=0),
       policy (greedy),
       Q_final, rho (estimated average cost), iterations performed
    """
    nS = len(states)
    V = np.zeros(nS)
    for it in range(max_iter):
        V_old = V.copy()
        Q = np.full((nS, 2), np.inf)
        for si in range(nS):
            for a in actions:
                Q[si, a] = c[(si, a)] + sum(p * V_old[sj] for sj, p in P[(si, a)].items())
        V_new = np.min(Q, axis=1)
        # normalize so ref state's value becomes 0
        shift = V_new[ref_idx]
        V_new = V_new - shift
        V = V_new
        if np.max(np.abs(V - V_old)) < tol:
            break
    # compute final Q and policy
    Q_final = np.full((nS, 2), np.inf)
    for si in range(nS):
        for a in actions:
            Q_final[si, a] = c[(si, a)] + sum(p * V[sj] for sj, p in P[(si, a)].items())
    policy = np.argmin(Q_final, axis=1)
    # since V[ref]=0, the average cost (gain) ρ is min_a Q_final[ref]
    rho = float(np.min(Q_final[ref_idx,:]))
    return V, policy, Q_final, rho, it+1

# -------------------- Scan over w --------------------
def scan_over_w(w_values, tau=1.0, tol=1e-9):
    summary = []
    policy_by_w = {}
    Vref_by_w = {}
    start = time.time()
    for w in w_values:
        P, c = build_model(w, tau=tau)
        V, policy, Q, rho, iters = relative_value_iteration(P, c, ref_idx, tol=tol)
        Pb = [si for si in range(nS) if policy[si] == 0]  # states where passive is optimal
        summary.append({"w": float(w), "rho": rho, "iters": iters, "n_passive": len(Pb)})
        policy_by_w[float(w)] = policy.copy()
        Vref_by_w[float(w)] = float(V[ref_idx])
    elapsed = time.time() - start
    print(f"Scan completed in {elapsed:.2f} s for {len(w_values)} w-values.")
    return pd.DataFrame(summary), policy_by_w, Vref_by_w

# -------------------- Main execution --------------------
if __name__ == "__main__":
    # choose w grid: -10000 .. 10000 (41 points)
    w_values = np.linspace(-10000, 10000, 41)
    df_summary, policy_by_w, Vref_by_w = scan_over_w(w_values, tau=1.0, tol=1e-9)

    # print summary head
    print("\nSummary (first 41 rows):")
    print(df_summary.head(41).to_string(index=False))

    # plot rho vs w
    plt.figure(figsize=(8,4))
    plt.plot(df_summary['w'], df_summary['rho'], marker='o')
    plt.xlabel("w (penalty when previous action = 1)")
    plt.ylabel("Estimated average cost rho")
    plt.title("rho vs w (RVI estimates)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save summary for inspection
    df_summary.to_csv("rvi_summary_w_scan.csv", index=False)
    print("Saved summary to rvi_summary_w_scan.csv")

    # Show example details for w=0
    w0 = 0.0
    P0, c0 = build_model(w0)
    V0, policy0, Q0, rho0, iters0 = relative_value_iteration(P0, c0, ref_idx)
    print("\nFor w = 0.0:")
    print("  rho =", rho0)
    print("  iterations =", iters0)
    # Present the policy mapping (small sample)
    sample_df = pd.DataFrame({
        "state_index": list(range(nS)),
        "state": states,
        "optimal_action": policy0
    })
    print("\nSample of policy (first 15 states):")
    print(sample_df.head(15).to_string(index=False))
    sample_df.to_csv("policy_w0.csv", index=False)
    print("Saved policy for w=0 to policy_w0.csv")

    # Finite horizon T example for w=0:
    fres = finite_horizon_relative(P0, c0, T=300, ref_idx=ref_idx)
    # extract g_t series
    gts = pd.DataFrame({"t": [r['t'] for r in fres], "g_t": [r['g_t'] for r in fres]})
    print("\nFinite-horizon g_t series (w=0):")
    print(gts.to_string(index=False))
    gts.to_csv("finite_horizon_g_t_w0.csv", index=False)
    print("Saved finite-horizon g_t series to finite_horizon_g_t_w0.csv")

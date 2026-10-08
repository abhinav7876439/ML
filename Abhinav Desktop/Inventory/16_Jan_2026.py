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
print(f"Reference state {reference_state} has index {ref_idx}.")

# -------------------------
# Cost parameters
# -------------------------
h = 1.0            # holding cost per unit inventory per step
p = 2.0            # perish penalty scale (per unit lost item)
l = 5.0            # lost-sales penalty per lost demand occurrence
gamma = 0.1        # perishability rate (per-unit per time step)
lam = 0.6          # demand rate
mu = 0.8           # production rate when active
switch_cost = 3.0  # symmetric switching cost when action != a_prev
tau = 1

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

# quick sanity prints (can remove if noisy)
# print("Example transitions from state (5,0) with action=1:", transitions_ctmc((5,0), 1, tau=1.0))
# print("Example transitions from state (5,0) with action=0:", transitions_ctmc((5,0), 0, tau=1.0))

# -------------------------
# Cost primitives
# -------------------------

def C_base(i):
    cost = h * i
    cost += p * gamma * i
    if i == 0:
        cost += l * lam * tau
    return cost


def C_switch():
    return switch_cost            # could be state-dependent; here constant switch_cost


# def C_switch(i):
#     """
#     Switching cost increasing in inventory
#     Models setup / disruption cost when inventory is high
#     """
#     return switch_cost * (1 + i)    # preserves monotonicity





def one_step_cost(state, action):
    i, a_prev = state
    cost = C_base(i)
    if a_prev != action:
        cost += C_switch()
    return cost

# -------------------------
# Build model: P and c
# -------------------------
def build_model(tau=1.0):
    P = {}
    c = {}
    for s in states:
        si = state_index[s]
        for a in actions:
            trans = transitions_ctmc(s, a, tau=tau)
            P[(si, a)] = {state_index[s2]: prob for s2, prob in trans.items()}
            c[(si, a)] = one_step_cost(s, a)
    return P, c



# -------------------------
# Relative Value Iteration (RVI)
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


P, c = build_model(tau=tau)

V, policy, Q, rho, iters = relative_value_iteration(
    P, c, ref_idx,
    tol=1e-8,
    max_iter=20000,
    verbose=True
)
print(f"RVI completed in {iters} iterations. Estimated average cost per step (rho) = {rho:.6f}")


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


def whittle_index(state, V, P):
    si = state_index[state]

    cont_passive = sum(
        prob * V[sj] for sj, prob in P[(si, 0)].items()
    )

    cont_active = sum(
        prob * V[sj] for sj, prob in P[(si, 1)].items()
    )

    return cont_passive - cont_active


W = {}
for s in states:
    W[s] = whittle_index(s, V, P)

# Example: only look at a_prev = 0
W_inventory = {i: W[(i, 0)] for i in inventory_levels}
print("Whittle index values (a_prev=0):")
for i in inventory_levels:
    print(f"  Inventory {i}: W = {W_inventory[i]:.4f}")

def extract_policy(w):
    P, c = build_model(tau=tau)
    V, policy, Q, rho, iterations = relative_value_iteration(P, c, ref_idx)
    policy_table = pd.DataFrame({
        "state": states,
        "optimal_action": policy
    })
    return policy_table

policy_table = extract_policy(w=0.0)
print("Optimal policy at w=0.0:")
print(policy_table)


policy_table = extract_policy(w=20000.0)
print("Optimal policy at w=20000.0:")
print(policy_table)
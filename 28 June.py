import numpy as np
import pandas as pd
from scipy.linalg import solve
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import os
import networkx as nx


# ----------------------------------------------------------------------
# markovianbandit package – graceful fallback
# ----------------------------------------------------------------------
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; package columns will be skipped.")


# ----------------------------------------------------------------------
# 1. Extended MDP builder
# ----------------------------------------------------------------------
def build_extended_mdp(N, p, C, K, C_switch):
    """
    N      : index of the terminal state (so p has length N+1, with p[N]=0)
    p      : array of length N+1, survival probabilities (terminal = 0.0)
    returns : P0, P1, R0, R1, state_labels
    """
    num_states = N + 2                     # (0,1), (0,0), (1,0), ..., (N,0)
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N+1)]

    # ---------- Active action ----------
    for s in range(num_states):
        if s == 0:                # (0,1)  – already active, only pay C
            P1[s, 0] = 1.0
            R1[s] = -C
        else:
            P1[s, 0] = 1.0        # switch to active and restart in (0,1)
            R1[s] = -(C + C_switch)

    # ---------- Passive action ----------
    for s in range(num_states):
        if s == 0:                # (0,1) – cannot be passive, but define anyway
            P0[s, 2] = p[0]       # jump to (1,0)
            P0[s, 1] = 1 - p[0]   # jump to (0,0)
            R0[s] = -(K * (1 - p[0]))
        elif s == 1:              # (0,0)
            P0[s, 2] = p[0]
            P0[s, 1] = 1 - p[0]
            R0[s] = -(K * (1 - p[0]))
        else:
            idx = s - 1           # p[idx] corresponds to state (idx,0)
            if idx < N:
                P0[s, s+1] = p[idx]      # move to next passive state
                P0[s, 1] = 1 - p[idx]    # jump back to (0,0)
                R0[s] = -(K * (1 - p[idx]))
            else:                 # terminal state (N,0), p[N]=0
                P0[s, 1] = 1.0
                R0[s] = -K

    return P0, P1, R0, R1, state_labels


def print_transition_matrix(P, state_labels, action_name):
    print("="*70)
    print(f"Transition matrix for action {action_name}")
    print("="*70)
    df = pd.DataFrame(P, index=state_labels, columns=state_labels)
    print(df)


def print_reward_vector(R, state_labels, action_name):
    print("="*70)
    print(f"Reward vector for action {action_name}")
    print("="*70)
    df = pd.DataFrame(R, index=state_labels, columns=[action_name])
    print(df)

# ----------------------------------------------------------------------
# 2. Discounted solvers (VI, PI)
# ----------------------------------------------------------------------
def solve_vi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=10000):
    S = len(R0)
    V = np.zeros(S)
    for _ in range(max_iter):
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        V_new = np.maximum(Q0, Q1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    Q0 = R0 + beta * (P0 @ V)
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1


def policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta):
    S = len(pi)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])
    I = np.eye(S)
    V = np.linalg.solve(I - beta * P_pi, r_pi)
    return V


def solve_pi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=1000):
    S = len(R0)
    pi = np.zeros(S, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta)
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + beta * (P0 @ V)
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1


def whittle_index_discounted(state, solver, beta, P0, P1, R0, R1,
                             w_min=-1000.0, w_max=1000.0,
                             tol_w=1e-5, tol_mdp=1e-8, verbose=False):
    """Bisection for one state (discounted)."""
    solve_f = solve_vi_discounted if solver == 'vi' else solve_pi_discounted

    def f(w):
        _, Q0, Q1 = solve_f(P0, P1, R0, R1, w, beta, tol_mdp)
        return Q0[state] - Q1[state]

    f_min = f(w_min)
    f_max = f(w_max)
    expand = 1.5
    while f_min * f_max > 0:
        if abs(f_min) < abs(f_max):
            w_min -= expand * abs(w_min - w_max) if abs(w_min - w_max) > 0 else 10.0
            f_min = f(w_min)
        else:
            w_max += expand * abs(w_max - w_min)
            f_max = f(w_max)
        if abs(w_min) > 1e6 or abs(w_max) > 1e6:
            raise ValueError("Cannot bracket root")
    for _ in range(160):
        w_mid = (w_min + w_max) / 2.0
        f_mid = f(w_mid)
        if abs(f_mid) < tol_w:
            return w_mid
        if f_mid * f_min > 0:
            w_min = w_mid
            f_min = f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0


# ----------------------------------------------------------------------
# 3. Closed‑form discounted Whittle indices (for extended MDP)
# ----------------------------------------------------------------------
# def compute_G_H(p, beta):
#     N = len(p)
#     H = np.zeros(N)
#     G = np.zeros(N+1)          # G[0] = 0
#     prod = 1.0
#     for k in range(N):
#         H[k] = beta**(k+1) * prod
#         G[k+1] = G[k] + (1 - p[k]) * H[k]
#         prod *= p[k]
#     return G, H


# def closed_form_indices(p_full, beta, K, C, C_switch):
#     """
#     p_full : length N+1 (includes terminal 0)
#     Returns: W_passive (length N+1), W_active (float)
#     """
#     G, H = compute_G_H(p_full, beta)
#     Nplus1 = len(p_full)
#     W_passive = np.zeros(Nplus1)
#     for k in range(Nplus1):
#         num = 1 - G[k+1] - p_full[k] * (1 - beta * G[k])
#         den = 1 - G[k+1] - beta * p_full[k] * (1 - G[k])
#         W_passive[k] = K * (num / den) - C - C_switch if den != 0 else np.inf
#     # active state (0,1) – use k=0 formula
#     if Nplus1 >= 2:
#         G1 = G[1]
#         H1 = H[1]
#     else:
#         G1 = G[1] if len(G) > 1 else 0
#         H1 = 0.0
#     numerator = (1 - beta) * (K * G1 + C_switch * H1)
#     denominator = beta * (1 - G1) - H1
#     W_active = (numerator / denominator) - C if denominator != 0 else np.inf
#     return W_passive, W_active



# =====================================================================
# Closed form Indices using exact derivations
# =====================================================================
def compute_G_H(p, beta):
    """
    G(k) = sum_{l=0}^{k-1} beta^{l+1} (1-p_l) prod_{j=0}^{l-1} p_j
    H(k) = beta^{k+1} prod_{l=0}^{k-1} p_l
    p : length N (no terminal zero)
    Returns:
      G : length N+2  (G[0]=0, G[1],...,G[N+1])
      H : length N+1  (H[0],...,H[N])
    """
    N = len(p)
    H = np.zeros(N+1)
    G = np.zeros(N+2)
    prod = 1.0
    for k in range(N+1):
        if k < N:
            H[k] = beta**(k+1) * prod
            prod *= p[k]
        else:
            H[k] = beta**(k+1) * prod   # prod after all p's
    G[0] = 0.0
    for k in range(N+1):
        p_k = p[k] if k < N else 0.0
        G[k+1] = G[k] + (1 - p_k) * H[k]
    return G, H

def closed_form_indices(p, beta, K, C, C_switch):
    """
    Compute closed-form Whittle indices for ALL states.
    p : length N (survival probabilities without terminal zero)
    Returns:
      w01        : index for (0,1)
      w_passive  : array of length N+1: indices for (0,0),(1,0),...,(N,0)
    """
    N = len(p)
    G, H = compute_G_H(p, beta)

    # --- (0,0) : w(0,0) = (1-p0)K - C - C_switch*(1-beta) ---
    p0 = p[0] if N > 0 else 0.0
    w00 = (1 - p0) * K - C - C_switch * (1 - beta)

    # --- (0,1) : W(0,1) with k=0 threshold ---
    if N >= 1:
        G1 = G[1]   # = G(1)
        H1 = H[1]   # = H(1)
        num = (1 - beta) * (K * G1 + C_switch * H1)
        den = beta * (1 - G1) - H1
        if abs(den) > 1e-15:
            w01 = (num / den) - C
        else:
            w01 = np.inf
    else:
        w01 = np.inf

    # --- (k,0) for k >= 1 : w(k,0) = K - C - C_switch - (K * pk * (1-beta)) / denom ---
    w_passive = np.zeros(N+1)
    w_passive[0] = w00
    for k in range(1, N+1):
        if k < N:
            pk = p[k]
        else:
            pk = 0.0
        Gk = G[k]   # = G(k)
        Hk = H[k]   # = H(k)
        denom = (1 - pk * beta) * (1 - Gk) - (1 - pk) * Hk
        if abs(denom) > 1e-15:
            w_passive[k] = K - C - C_switch - (K * pk * (1 - beta)) / denom
        else:
            w_passive[k] = np.inf
    return w01, w_passive

# ----------------------------------------------------------------------
# 4. Average‑reward solvers (RVI, PI)
# ----------------------------------------------------------------------
def solve_rvi(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=10000):
    n = len(R0)
    V = np.zeros(n)
    for _ in range(max_iter):
        Q0 = R0 + P0 @ V
        Q1 = R1 - w + P1 @ V
        V_tilde = np.maximum(Q0, Q1)
        delta = V_tilde[ref]
        V_new = V_tilde - delta
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q0 = R0 + P0 @ V
    Q1 = R1 - w + P1 @ V
    g = delta
    return V, g, Q0, Q1


def policy_evaluation_avg(pi, R0, R1, P0, P1, w, ref=0):
    n = len(R0)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(n)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(n)])
    A = np.zeros((n+1, n+1))
    A[:n, :n] = np.eye(n) - P_pi
    A[:n, -1] = 1.0
    A[n, ref] = 1.0
    b = np.zeros(n+1)
    b[:n] = r_pi
    sol = np.linalg.solve(A, b)
    h = sol[:n]
    g = sol[-1]
    return h, g


def solve_pi_avg(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=100):
    n = len(R0)
    pi = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        h, g = policy_evaluation_avg(pi, R0, R1, P0, P1, w, ref)
        Q0 = R0 + P0 @ h
        Q1 = R1 - w + P1 @ h
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + P0 @ h
    Q1 = R1 - w + P1 @ h
    return h, g, Q0, Q1


def whittle_index_bisection_avg(state, solver, R0, R1, P0, P1,
                                w_min=-2000.0, w_max=2000.0,
                                tol_w=1e-10, tol_mdp=1e-12, ref=0, verbose=False):
    solve_f = solve_rvi if solver == 'rvi' else solve_pi_avg
    def f(w):
        _, _, Q0, Q1 = solve_f(R0, R1, P0, P1, w, ref=ref, tol=tol_mdp)
        return Q0[state] - Q1[state]
    f_min = f(w_min)
    f_max = f(w_max)
    expand = 2.0
    while f_min * f_max > 0:
        if abs(f_min) < abs(f_max):
            w_min -= expand
            f_min = f(w_min)
        else:
            w_max += expand
            f_max = f(w_max)
        if abs(w_min) > 1e4 or abs(w_max) > 1e4:
            raise RuntimeError("Cannot bracket average index")
    for _ in range(160):
        w_mid = (w_min + w_max) / 2.0
        f_mid = f(w_mid)
        if abs(f_mid) < tol_w:
            return w_mid
        if f_mid * f_min > 0:
            w_min = w_mid
            f_min = f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0



# ----------------------------------------------------------------------
# Compute and print indices for both problems (Average Cost)
# ----------------------------------------------------------------------
def compute_indices_avg(R0, R1, P0, P1, problem_name, state_labels=None):
    """Compute Whittle indices using average cost methods."""
    print(f"\n=== {problem_name} ===")
    results = {}
    
    for method in ['rvi', 'pi']:
        print(f"\n--- {method.upper()} (Average Cost) ---")
        indices = []
        for s in range(len(R0)):
            try:
                lam = whittle_index_bisection_avg(s, method, R0, R1, P0, P1,
                                                  w_min=-1000.0, w_max=1000.0,
                                                  tol_w=1e-10, tol_mdp=1e-12,
                                                  ref=0, verbose=False)
                indices.append(lam)
                if state_labels:
                    print(f"State {state_labels[s]:8s}: λ = {lam:.10f}")
                else:
                    print(f"State {s+1}: λ = {lam:.10f}")
            except Exception as e:
                print(f"State {s+1}: Failed - {e}")
                indices.append(np.nan)
        results[method] = indices
    
    return results


# ----------------------------------------------------------------------
# Compute Discounted Indices
# ----------------------------------------------------------------------
def compute_indices_discounted(R0, R1, P0, P1, beta, problem_name, state_labels=None):
    """Compute Whittle indices using discounted cost methods."""
    print(f"\n=== {problem_name} ===")
    results = {}
    
    for method in ['vi', 'pi']:
        print(f"\n--- {method.upper()} (Discounted, β={beta}) ---")
        indices = []
        for s in range(len(R0)):
            try:
                lam = whittle_index_discounted(state=s, solver=method, beta=beta,
                                               w_min=-1000.0, w_max=1000.0,
                                               tol_w=1e-5, tol_mdp=1e-8,
                                               P0=P0, P1=P1, R0=R0, R1=R1,
                                               verbose=False)
                indices.append(lam)
                if state_labels:
                    print(f"State {state_labels[s]:8s}: λ = {lam:8.4f}")
                else:
                    print(f"State {s+1}: λ = {lam:.10f}")
            except Exception as e:
                print(f"State {s+1}: Failed - {e}")
                indices.append(np.nan)
        results[method] = indices
    
    return results


# ----------------------------------------------------------------------
# Lecture MDP
# ----------------------------------------------------------------------

def run_three_state_problem():
    """
    PART 3: 3‑state MDP with RED (passive) and BLUE (active) actions.
    Computes discounted (β=0.9,0.95,0.99) and average‑cost Whittle indices,
    shows optimal V*(s) and Q*(s,a) for w=0, and compares with package.
    """
    print("\n" + "=" * 70)
    print("PART 3: 3‑STATE RED/BLUE PROBLEM")
    print("=" * 70)

    # ---- Define raw transitions ---- 
    transitions = {
        # RED action (0) – passive
        (0, 0): [(0.5, 0, 0), (0.5, -1, 1)],      # s1
        (1, 0): [(0.25, -1, 0), (0.75, -2, 2)],    # s2
        (2, 0): [(0.5, 3, 1), (0.5, 3, 2)],        # s3
        # BLUE action (1) – active
        (0, 1): [(1.0, 1, 0)],
        (1, 1): [(1.0, 2, 0)],
        (2, 1): [(1.0, 1, 0)]
    }

    # Validate probability sums
    for (s, a), lst in transitions.items():
        total = sum(p for p, r, ns in lst)
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"Sum of probs for state {s}, action {a} is {total}, not 1")

    n_states = 3
    state_labels = ['s1', 's2', 's3']

    # Build matrices P0, P1, R0, R1
    P0 = np.zeros((n_states, n_states))
    P1 = np.zeros((n_states, n_states))
    R0 = np.zeros(n_states)
    R1 = np.zeros(n_states)

    for (s, a), lst in transitions.items():
        if a == 0:
            for (p, r, ns) in lst:
                P0[s, ns] += p
                R0[s] += p * r
        else:  # a == 1
            for (p, r, ns) in lst:
                P1[s, ns] += p
                R1[s] += p * r

    # Print matrices
    print("\nTransition matrix for RED (passive):")
    print(pd.DataFrame(P0, index=state_labels, columns=state_labels))
    print("\nReward vector for RED (passive):", R0)

    print("\nTransition matrix for BLUE (active):")
    print(pd.DataFrame(P1, index=state_labels, columns=state_labels))
    print("\nReward vector for BLUE (active):", R1)

    # ---- Discounted Whittle indices for β = 0.9, 0.95, 0.99 ----
    betas = [0.9, 0.95, 0.99]

    # Store all discounted results for final comparison table
    disc_table_data = {}
    for beta in betas:
        print(f"\n--- Discounted Whittle Indices (β = {beta}) ---")
        indices_vi = []
        indices_pi = []
        for s in range(n_states):
            try:
                lam_vi = whittle_index_discounted(s, 'vi', beta, P0, P1, R0, R1,
                                                  w_min=-1000, w_max=1000,
                                                  tol_w=1e-5, tol_mdp=1e-8, verbose=False)
            except Exception:
                lam_vi = np.nan
            try:
                lam_pi = whittle_index_discounted(s, 'pi', beta, P0, P1, R0, R1,
                                                  w_min=-1000, w_max=1000,
                                                  tol_w=1e-5, tol_mdp=1e-8, verbose=False)
            except Exception:
                lam_pi = np.nan
            indices_vi.append(lam_vi)
            indices_pi.append(lam_pi)
            print(f"  State {state_labels[s]:3s}:  VI = {lam_vi:8.4f}   PI = {lam_pi:8.4f}")

        # Show optimal V* and Q* for w=0 (illustrative)
        print(f"\n  Optimal value function V* and Q* for w=0 (β={beta}):")
        V_opt, Q0_opt, Q1_opt = solve_vi_discounted(P0, P1, R0, R1, w=0.0, beta=beta)
        for s, lbl in enumerate(state_labels):
            print(f"    {lbl}: V={V_opt[s]:8.4f}  Q0={Q0_opt[s]:8.4f}  Q1={Q1_opt[s]:8.4f}")

        # Package (discounted)
        pkg_idx = [np.nan]*n_states
        if PKG_AVAILABLE:
            try:
                model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                pkg_idx = model.whittle_indices(discount=beta)
                print(f"\n  Package indices: {pkg_idx}")
            except Exception:
                print("  Package failed.")

        disc_table_data[beta] = (indices_vi, indices_pi, pkg_idx)

    # ---- Average‑cost Whittle indices ----
    print("\n--- Average‑Cost Whittle Indices ---")
    avg_vi = []
    avg_pi = []
    for s in range(n_states):
        try:
            lam_vi = whittle_index_bisection_avg(s, 'rvi', R0, R1, P0, P1,
                                                 w_min=-100, w_max=100,
                                                 tol_w=1e-10, tol_mdp=1e-12, verbose=False)
        except Exception:
            lam_vi = np.nan
        try:
            lam_pi = whittle_index_bisection_avg(s, 'pi', R0, R1, P0, P1,
                                                 w_min=-100, w_max=100,
                                                 tol_w=1e-10, tol_mdp=1e-12, verbose=False)
        except Exception:
            lam_pi = np.nan
        avg_vi.append(lam_vi)
        avg_pi.append(lam_pi)
        print(f"  State {state_labels[s]:3s}:  RVI = {lam_vi:8.4f}   PI = {lam_pi:8.4f}")

    pkg_avg = [np.nan]*n_states
    if PKG_AVAILABLE:
        try:
            model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
            pkg_avg = model.whittle_indices()
            print(f"\n  Package indices (avg): {pkg_avg}")
        except Exception:
            print("  Package average failed.")

    # ---- Final Comparison Tables ----
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON – 3‑STATE RED/BLUE PROBLEM")
    print("=" * 70)

    # Discounted table for each beta
    for beta in betas:
        vi, pi, pkg = disc_table_data[beta]
        df = pd.DataFrame({
            'State': state_labels,
            'VI': vi,
            'PI': pi,
            'Package': pkg
        })
        print(f"\nDiscounted (β = {beta}):")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A"))

    # Average cost table
    df_avg = pd.DataFrame({
        'State': state_labels,
        'RVI': avg_vi,
        'PI': avg_pi,
        'Package': pkg_avg
    })
    print("\nAverage Cost:")
    print(df_avg.to_string(index=False, float_format=lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A"))

    print("\n3‑state problem finished.\n")


# ----------------------------------------------------------------------
# 5. Helper: string representations of matrices
# ----------------------------------------------------------------------
def transition_matrix_to_str(P, state_labels):
    df = pd.DataFrame(P, index=state_labels, columns=state_labels)
    return df.to_string(float_format=lambda x: f"{x:.6f}")


def reward_vector_to_str(R, state_labels, action_name):
    df = pd.DataFrame(R, index=state_labels, columns=[action_name])
    return df.to_string(float_format=lambda x: f"{x:.4f}")


# ----------------------------------------------------------------------
# 6. Main experimental loop
# ----------------------------------------------------------------------
def run_experiments():
    #----- Parameter ranges -----
    C_vals = [5.0, 10.0, 20.0, 400.0, 600.0, 1200.0]
    K_vals = [500.0, 1000.0, 2000.0]
    C_switch_vals = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 75.0, 100.0, 200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0]
    beta_vals = [0.9, 0.95, 0.9999]

    # C_vals = [5.0, 10.0, 20.0, 600.0]
    # K_vals = [500.0, 1000.0, 2000.0]
    # C_switch_vals = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 75.0, 100.0, 200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0]
    # beta_vals = [0.9, 0.95, 0.9999]

    # # ----- Transition matrices (survival probability arrays) -----
    # # N = 4, each array must be length 5 with terminal 0.
    # trans_matrices = [
    #     np.array([0.9507, 0.732, 0.5987, 0.3745, 0.0]),
    #     np.array([0.9, 0.8, 0.5, 0.3, 0.0]),
    #     np.array([0.99, 0.85, 0.7, 0.55, 0.0])
    # ]


    # ----- Transition matrices (randomly generated) -----
    N = 5                                  # number of passive states (excluding terminal)
    num_random_matrices = 3                # how many different matrices to generate
    np.random.seed(42)                     # for reproducibility (optional, change or remove as you like)

    trans_matrices = []
    for _ in range(num_random_matrices):
        p = np.sort(np.random.uniform(0.0, 1.0, N))[::-1]   # decreasing survival probs
        p_full = np.append(p, 0.0)                         # terminal state with 0 probability
        trans_matrices.append(p_full)

    total_combos = len(trans_matrices) * len(C_vals) * len(K_vals) * len(C_switch_vals) * len(beta_vals)
    instance_counter = 0

    # ----- Open output file -----


    # inside run_experiments():
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_filename = f"whittle_experiment_results_{now_str.replace(' ', '_').replace(':', '')}.txt"

    with open(out_filename, "w") as f:
        # ----- New header block -----
        f.write("=" * 80 + "\n")
        f.write("WHITTLE INDICES COMPARISON: CLOSED-FORM vs PACKAGE\n")
        #f.write(f"Generated on: {now_str}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")



        # ---- Header with parameter table ----
        f.write("Parameter ranges explored:\n")
        f.write(f"  C (activation cost): {C_vals}\n")
        f.write(f"  K (penalty cost): {K_vals}\n")
        f.write(f"  C_switch (switching cost): {C_switch_vals}\n")
        f.write(f"  beta (discount factor): {beta_vals}\n")
        f.write(f"  Number of transition matrices: {len(trans_matrices)}\n")
        f.write(f"  Total combinations: {total_combos}\n")
        f.write("\n" + "#" * 80 + "\n")

        # ---- Loop over transition matrices ----
        for mat_idx, p_full in enumerate(trans_matrices, start=1):
            N = len(p_full) - 1          # terminal index
            p_str = np.array2string(p_full, precision=4, separator=', ')
            f.write(f"TRANSITION MATRIX {mat_idx}\n")
            f.write(f"Survival probabilities: {p_str}\n")
            f.write("#" * 80 + "\n\n")

            # ---- Loop over parameter combinations ----
            for C in C_vals:
                for K in K_vals:
                    for C_switch in C_switch_vals:
                        for beta in beta_vals:
                            instance_counter += 1

                            # ---- Terminal progress ----
                            print(f"Working on instance {instance_counter}/{total_combos}: "
                                  f"C={C}, K={K}, C_switch={C_switch}, beta={beta}, matrix {mat_idx}",
                                  flush=True)

                            # ---- Build the extended MDP ----
                            P0, P1, R0, R1, state_labels = build_extended_mdp(N, p_full, C, K, C_switch)

                            # ---- Write instance header ----
                            f.write("=" * 80 + "\n")
                            f.write(f"INSTANCE {instance_counter}\n")
                            f.write("=" * 80 + "\n")
                            f.write("Parameters:\n")
                            f.write(f"  N = {N}\n")
                            f.write(f"  Survival probabilities p[0..{N}]: {p_str}\n")
                            f.write(f"  C = {C}\n")
                            f.write(f"  K = {K}\n")
                            f.write(f"  C_switch = {C_switch}\n")
                            f.write(f"  beta = {beta}\n")
                            f.write("-" * 80 + "\n\n")

                            # ---- Transition matrices & reward vectors ----
                            f.write("Transition matrix for action Passive (a=0):\n")
                            f.write(transition_matrix_to_str(P0, state_labels) + "\n\n")
                            f.write("Reward vector for action Passive (a=0):\n")
                            f.write(reward_vector_to_str(R0, state_labels, "Passive") + "\n\n")

                            f.write("Transition matrix for action Active (a=1):\n")
                            f.write(transition_matrix_to_str(P1, state_labels) + "\n\n")
                            f.write("Reward vector for action Active (a=1):\n")
                            f.write(reward_vector_to_str(R1, state_labels, "Active") + "\n\n")

                            # ---- Discounted Whittle indices ----
                            f.write(f"--- Discounted Whittle Indices (beta = {beta}) ---\n")
                            # Compute
                            disc_results = {}
                            for method in ['vi', 'pi']:
                                indices = []
                                for s in range(len(R0)):
                                    try:
                                        lam = whittle_index_discounted(
                                            s, method, beta, P0, P1, R0, R1,
                                            w_min=-2000.0, w_max=2000.0,
                                            tol_w=1e-5, tol_mdp=1e-8, verbose=False
                                        )
                                        indices.append(lam)
                                    except Exception as e:
                                        indices.append(np.nan)
                                disc_results[method] = indices

                            # Closed‑form
                            try:
                                W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
                                closed = [W_active] + list(W_passive)
                            except Exception:
                                closed = [np.nan] * len(state_labels)

                            # Package (discounted)
                            if PKG_AVAILABLE:
                                try:
                                    model_ext = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                                    pkg_disc = model_ext.whittle_indices(discount=beta)
                                except Exception:
                                    pkg_disc = [np.nan] * len(state_labels)
                            else:
                                pkg_disc = [np.nan] * len(state_labels)

                            # Build DataFrame and write
                            df_disc = pd.DataFrame({
                                'ClosedForm': closed,
                                'VI': disc_results['vi'],
                                'PI': disc_results['pi'],
                                'Package': pkg_disc
                            }, index=state_labels)
                            f.write(df_disc.to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")

                            # ---- Average‑cost Whittle indices ----
                            f.write("--- Average-Cost Whittle Indices ---\n")
                            avg_results = {}
                            for method in ['rvi', 'pi']:
                                indices = []
                                for s in range(len(R0)):
                                    try:
                                        lam = whittle_index_bisection_avg(
                                            s, method, R0, R1, P0, P1,
                                            w_min=-2000.0, w_max=2000.0,
                                            tol_w=1e-8, tol_mdp=1e-12, verbose=False
                                        )
                                        indices.append(lam)
                                    except Exception as e:
                                        indices.append(np.nan)
                                avg_results[method] = indices

                            if PKG_AVAILABLE:
                                try:
                                    model_ext = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                                    pkg_avg = model_ext.whittle_indices()
                                except Exception:
                                    pkg_avg = [np.nan] * len(state_labels)
                            else:
                                pkg_avg = [np.nan] * len(state_labels)

                            df_avg = pd.DataFrame({
                                'RVI': avg_results['rvi'],
                                'PI': avg_results['pi'],
                                'Package': pkg_avg
                            }, index=state_labels)
                            f.write(df_avg.to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")

    print("\nAll experiments completed. Results saved to 'whittle_experiment_results.txt'.")







# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # ===================================================================
    # PART 1: Standard Problems (Circulant, Restart, Non-indexable)
    # ===================================================================
    
    print("\n" + "="*70)
    print("PART 1: STANDARD PROBLEMS WITH AVERAGE COST")
    print("="*70)
    
    # ----------------------------------------------------------------------
    # Problem 1: Circulant dynamics (4 states)
    # ----------------------------------------------------------------------
    P0_circ = np.array([[0.5, 0.0, 0.0, 0.5],
                        [0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.5, 0.5, 0.0],
                        [0.0, 0.0, 0.5, 0.5]])

    P1_circ = np.array([[0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.5, 0.5, 0.0],
                        [0.0, 0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.0, 0.5]])

    R_circ = np.array([-1.0, 0.0, 0.0, 1.0])   # same for passive and active
    R0_circ = R_circ
    R1_circ = R_circ

    expected_circ = [-0.5, 0.5, 1.0, -1.0]

    # ----------------------------------------------------------------------
    # Problem 2: Restart problem (5 states)
    # ----------------------------------------------------------------------
    a = 0.9
    P0_rest = np.array([
        [0.1, 0.9, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.9, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.9, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.9],
        [0.1, 0.0, 0.0, 0.0, 0.9]
    ])

    P1_rest = np.ones((5, 5)) * 0
    P1_rest[:, 0] = 1.0   # restart to state 1 (index 0)

    R0_rest = np.array([a**(i+1) for i in range(5)])   # passive rewards
    R1_rest = np.zeros(5)                              # active rewards are zero

    expected_rest = [-0.9, -0.73, -0.5, -0.26, -0.01]

    # ----------------------------------------------------------------------
    # Problem 3: Non-Indexable problem (3 states)
    # ----------------------------------------------------------------------
    P0_non_indexable = np.array([[0.005, 0.793, 0.202],
                        [0.027, 0.558, 0.415],
                        [0.736, 0.269, 0.015]])

    P1_non_indexable = np.array([[0.718, 0.254, 0.028],
                                 [0.347, 0.097, 0.556],
                        [0.015, 0.956, 0.029]])

    R0_non_indexable = np.array([0.0, 0.0, 0.0])
    R1_non_indexable = np.array([0.699, 0.362, 0.715])

    # ----------------------------------------------------------------------
    # Problem 4: Non-Indexable problem (4 states)
    # ----------------------------------------------------------------------
    P0_non = np.array([[0.01, 0.11, 0.05, 0.83],
                        [0.48, 0.26, 0.22, 0.06],
                        [0.26, 0.21, 0.29, 0.26],
                        [0.41, 0.10, 0.34, 0.15]])

    P1_non = np.array([[0.18, 0.68, 0.10, 0.04],
                        [0.33, 0.19, 0.33, 0.15],
                        [0.26, 0.39, 0.03, 0.34],
                        [0.17, 0.02, 0.10, 0.71]])

    R0_non  = np.array([0.23, 0.93, 0.50, 0.57])
    R1_non = np.array([0.33, 0.60, 0.27, 0.72])

    # Compute average cost indices for all standard problems
    results_circ_avg = compute_indices_avg(R0_circ, R1_circ, P0_circ, P1_circ, 
                                           "Circulant dynamics (Average Cost)")
    
    results_rest_avg = compute_indices_avg(R0_rest, R1_rest, P0_rest, P1_rest, 
                                           "Restart problem (Average Cost)")
    
    results_non3_avg = compute_indices_avg(R0_non_indexable, R1_non_indexable, 
                                           P0_non_indexable, P1_non_indexable, 
                                           "Non-indexable problem - 3 states (Average Cost)")
    
    results_non4_avg = compute_indices_avg(R0_non, R1_non, P0_non, P1_non, 
                                           "Non-indexable problem - 4 states (Average Cost)")

    # Compare with expected values where available
    print("\n" + "="*70)
    print("COMPARISON WITH EXPECTED VALUES")
    print("="*70)
    
    for method in ['rvi', 'pi']:
        print(f"\n--- {method.upper()} ---")
        
        # Circulant
        diff_circ = np.abs(np.array(results_circ_avg[method]) - np.array(expected_circ))
        print(f"Circulant - Max error: {np.max(diff_circ):.2e}")
        print(f"  Computed: {[round(x, 6) for x in results_circ_avg[method]]}")
        print(f"  Expected: {expected_circ}")
        
        # Restart
        diff_rest = np.abs(np.array(results_rest_avg[method]) - np.array(expected_rest))
        print(f"Restart - Max error: {np.max(diff_rest):.2e}")
        print(f"  Computed: {[round(x, 6) for x in results_rest_avg[method]]}")
        print(f"  Expected: {expected_rest}")




    # Compare with markovianbandit package if available (Average Cost)
    if PKG_AVAILABLE:
        print("\n" + "="*70)
        print("PACKAGE COMPARISON (Average Cost)")
        print("="*70)
        
        


        # Create restless bandit models
        model_circ = bandit.restless_bandit_from_P0P1_R0R1(P0_circ, P1_circ, R0_circ, R1_circ)
        model_rest = bandit.restless_bandit_from_P0P1_R0R1(P0_rest, P1_rest, R0_rest, R1_rest)
        model_non_3_states = bandit.restless_bandit_from_P0P1_R0R1(P0_non_indexable, P1_non_indexable, R0_non_indexable, R1_non_indexable)
        model_non_4_states = bandit.restless_bandit_from_P0P1_R0R1(P0_non, P1_non, R0_non, R1_non)


        # Average-cost indices from package
        pkg_circ_avg = model_circ.whittle_indices()
        pkg_rest_avg = model_rest.whittle_indices()
        pkg_non_3_states_avg = model_non_3_states.whittle_indices()
        pkg_non_4_states_avg = model_non_4_states.whittle_indices()

        print("\nPackage Whittle indices (average cost):")
        
        print("\nCirculant problem:")
        for s, idx in enumerate(pkg_circ_avg):
            print(f"  State {s+1}: λ = {idx:.6f}")
            
        print("\nRestart problem:")
        for s, idx in enumerate(pkg_rest_avg):
            print(f"  State {s+1}: λ = {idx:.6f}")
            
        print("\nNon-indexable problem (3 states):")
        for s, idx in enumerate(pkg_non_3_states_avg):
            print(f"  State {s+1}: λ = {idx:.6f}")
            
        print("\nNon-indexable problem (4 states):")
        for s, idx in enumerate(pkg_non_4_states_avg):
            print(f"  State {s+1}: λ = {idx:.6f}")



    # ----------------------------------------------------------------------
    # Summary Comparison Tables
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY COMPARISON - Circulant Problem")
    print("="*70)
    
    # Create DataFrame for discounted indices
    df_disc_circulant = pd.DataFrame({
        'State': [f"State {i+1}" for i in range(len(R0_circ))],
        'RVI (Average Cost)': results_circ_avg['rvi'],
        'PI (Average Cost)': results_circ_avg['pi'],
        'Package (Average Cost)': pkg_circ_avg,
        'Expected': expected_circ,
    }) # if PKG_AVAILABLE else np.nan
     
    print("\n Average-Cost Indices:")
    print(df_disc_circulant)


    print("\n" + "="*70)
    print("SUMMARY COMPARISON - Restart Problem")
    print("="*70)
    
    # Create DataFrame for discounted indices
    df_disc_restart = pd.DataFrame({
        'State': [f"State {i+1}" for i in range(len(R0_rest))],  # <-- fixed length
        'RVI (Average Cost)': results_rest_avg['rvi'],
        'PI (Average Cost)': results_rest_avg['pi'],
        'Package (Average Cost)': pkg_rest_avg if PKG_AVAILABLE else np.nan,
        'Expected': expected_rest,
    })
     
    print("\n Average-Cost Indices:")
    print(df_disc_restart)





    print("\n" + "="*70)
    print("SUMMARY COMPARISON - Non-Indexable Problem (3 States)")
    print("="*70)
    
    # Create DataFrame for discounted indices
    df_disc_non_3_states = pd.DataFrame({
        'State': [f"State {i+1}" for i in range(len(R0_non_indexable))],
        'RVI (Average Cost)': results_non3_avg['rvi'],
        'PI (Average Cost)': results_non3_avg['pi'],
        'Package (Average Cost)': pkg_non_3_states_avg,
    }) # if PKG_AVAILABLE else np.nan
     
    print("\n Average-Cost Indices:")
    print(df_disc_non_3_states)


    print("\n" + "="*70)
    print("SUMMARY COMPARISON - Non-Indexable Problem (4 States)")
    print("="*70)
    
    # Create DataFrame for discounted indices
    df_disc_non_4_states = pd.DataFrame({
        'State': [f"State {i+1}" for i in range(len(R0_non))],
        'RVI (Average Cost)': results_non4_avg['rvi'],
        'PI (Average Cost)': results_non4_avg['pi'],
        'Package (Average Cost)': pkg_non_4_states_avg,
    }) # if PKG_AVAILABLE else np.nan
     
    print("\n Average-Cost Indices:")
    print(df_disc_non_4_states)

    # ===================================================================
    # PART 2: Extended MDP Problem
    # ===================================================================
    
    print("\n\n" + "="*70)
    print("PART 2: EXTENDED MDP PROBLEM")
    print("="*70)
    
    # Problem parameters
    N = 5
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)                       # terminal zero for the extended MDP
    C, K = 5.0, 500.0
    C_switch = 1.0
    beta = 0.9999                           # discount factor


    print(f"\nParameters:")
    print(f"  N = {N}")
    print(f"  p = {np.round(p, 3)}")
    print(f"  C = {C}, K = {K}, C_switch = {C_switch}")
    print(f"  β = {beta}")

    # Build Extended MDP
    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)

    # Print matrices
    print("\n" + "="*70)
    print("EXTENDED MDP MATRICES")
    print("="*70)
    print_transition_matrix(P0_ext, state_labels_ext, "Passive (a=0)")
    print_reward_vector(R0_ext, state_labels_ext, "Passive (a=0)")
    print_transition_matrix(P1_ext, state_labels_ext, "Active (a=1)")
    print_reward_vector(R1_ext, state_labels_ext, "Active (a=1)")

    # ----------------------------------------------------------------------
    # 2A. Discounted Whittle Indices
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("DISCOUNTED WHITTLE INDICES (β = {})".format(beta))
    print("="*70)
    
    results_ext_disc = compute_indices_discounted(R0_ext, R1_ext, P0_ext, P1_ext, beta, 
                                                   "Extended MDP (Discounted)", 
                                                   state_labels_ext)

    # Closed-form indices
    print("\n--- Closed Form (Discounted) ---")
    W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
    print(f"  Active state (0,1): W = {W_active:.6f}")
    for k in range(N+1):
        print(f"  Passive state ({k},0): W = {W_passive[k]:.6f}")

    # ----------------------------------------------------------------------
    # 2B. Average Cost Whittle Indices
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("AVERAGE COST WHITTLE INDICES")
    print("="*70)
    
    results_ext_avg = compute_indices_avg(R0_ext, R1_ext, P0_ext, P1_ext, 
                                          "Extended MDP (Average Cost)",
                                          state_labels_ext)

    # ----------------------------------------------------------------------
    # Package Comparison for Extended MDP
    # ----------------------------------------------------------------------
    if PKG_AVAILABLE:
        print("\n" + "="*70)
        print("PACKAGE COMPARISON FOR EXTENDED MDP")
        print("="*70)
        
        # Build the extended MDP model
        model_ext = bandit.restless_bandit_from_P0P1_R0R1(P0_ext, P1_ext, R0_ext, R1_ext)
        
        # Discounted indices from package
        pkg_ext_disc = model_ext.whittle_indices(discount=beta)
        
        print("\nPackage discounted indices (β = {}):".format(beta))
        for s, (label, idx) in enumerate(zip(state_labels_ext, pkg_ext_disc)):
            print(f"  {label:8s}: λ = {idx:.6f}")
        
        # Average cost indices from package
        pkg_ext_avg = model_ext.whittle_indices()
        
        print("\nPackage average cost indices:")
        for s, (label, idx) in enumerate(zip(state_labels_ext, pkg_ext_avg)):
            print(f"  {label:8s}: λ = {idx:.6f}")

    # ----------------------------------------------------------------------
    # Summary Comparison Tables
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY COMPARISON - EXTENDED MDP")
    print("="*70)
    
    # Create DataFrame for discounted indices
    df_disc = pd.DataFrame({
        'State': state_labels_ext,
        'VI (Disc)': results_ext_disc['vi'],
        'PI (Disc)': results_ext_disc['pi'],
        'Closed Form (Disc)': np.append(W_active, W_passive),
        'Package (Disc)': pkg_ext_disc if PKG_AVAILABLE else np.nan
    })
    print("\nDiscounted Indices:")
    print(df_disc)
    
    # Create DataFrame for average cost indices
    df_avg = pd.DataFrame({
        'State': state_labels_ext,
        'RVI (Avg)': results_ext_avg['rvi'],
        'PI (Avg)': results_ext_avg['pi'],
        'Package (Avg)': pkg_ext_avg if PKG_AVAILABLE else np.nan
    })
    print("\nAverage Cost Indices:")
    print(df_avg)

    print("\nAll computations completed successfully.")
    print("\n" + "##"*70 + "\n")


    # ----------------------------------------------------------------------
    # 3. Three-State Red/Blue Problem
    # ----------------------------------------------------------------------
    run_three_state_problem()


    # ----------------------------------------------------------------------
    # 4. Run full experiments with parameter sweeps and output to file
    # ----------------------------------------------------------------------
    run_experiments()
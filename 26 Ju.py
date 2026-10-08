import numpy as np
import pandas as pd
from typing import Tuple
from scipy.linalg import solve
from datetime import datetime
import sys

# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; package comparison will be skipped.")

# ----------------------------------------------------------------------
# 1. MDP builder (unchanged)
# ----------------------------------------------------------------------
def build_extended_mdp(N, p, C, K, C_switch):
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N+1)]

    # Active transitions
    for s in range(num_states):
        if s == 0:
            P1[s, 0] = 1.0
            R1[s] = -C
        else:
            P1[s, 0] = 1.0
            R1[s] = -(C + C_switch)

    # Passive transitions
    for s in range(num_states):
        if s == 0:
            P0[s, 2] = p[0]
            P0[s, 1] = 1 - p[0]
            R0[s] = -(K * (1 - p[0]))
        elif s == 1:
            P0[s, 2] = p[0]
            P0[s, 1] = 1 - p[0]
            R0[s] = -(K * (1 - p[0]))
        else:
            idx = s - 1
            if idx < N:
                P0[s, s+1] = p[idx]
                P0[s, 1] = 1 - p[idx]
                R0[s] = -(K * (1 - p[idx]))
            else:
                P0[s, 1] = 1.0
                R0[s] = -K

    return P0, P1, R0, R1, state_labels


# ----------------------------------------------------------------------
# 2. Discounted solvers
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
    return np.linalg.solve(I - beta * P_pi, r_pi)


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
    return V, Q0, Q1


# ----------------------------------------------------------------------
# 3. Discounted bisection
# ----------------------------------------------------------------------
def q_diff_discounted(state, w, solver, beta, P0, P1, R0, R1, tol=1e-8):
    if solver == 'vi':
        _, Q0, Q1 = solve_vi_discounted(P0, P1, R0, R1, w, beta, tol)
    elif solver == 'pi':
        _, Q0, Q1 = solve_pi_discounted(P0, P1, R0, R1, w, beta, tol)
    else:
        raise ValueError("solver must be 'vi' or 'pi'")
    return Q0[state] - Q1[state]


def whittle_index_discounted(state, solver, beta, w_min=-100.0, w_max=100.0,
                             tol_w=1e-4, tol_mdp=1e-8, P0=None, P1=None, R0=None, R1=None):
    f_min = q_diff_discounted(state, w_min, solver, beta, P0, P1, R0, R1, tol_mdp)
    f_max = q_diff_discounted(state, w_max, solver, beta, P0, P1, R0, R1, tol_mdp)

    # Expand bracket if necessary
    expand = 1.5
    while f_min * f_max > 0:
        if abs(f_min) < abs(f_max):
            w_min -= expand * abs(w_min - w_max) if abs(w_min - w_max) > 0 else 10.0
            f_min = q_diff_discounted(state, w_min, solver, beta, P0, P1, R0, R1, tol_mdp)
        else:
            w_max += expand * abs(w_max - w_min)
            f_max = q_diff_discounted(state, w_max, solver, beta, P0, P1, R0, R1, tol_mdp)
        if abs(w_min) > 1e6 or abs(w_max) > 1e6:
            raise ValueError(f"Could not bracket discounted index for state {state}")

    for _ in range(50):
        w_mid = (w_min + w_max) / 2.0
        f_mid = q_diff_discounted(state, w_mid, solver, beta, P0, P1, R0, R1, tol_mdp)
        if abs(f_mid) < tol_w:
            return w_mid
        if f_mid * f_min > 0:
            w_min = w_mid
            f_min = f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0


# ----------------------------------------------------------------------
# 4. Closed-form discounted indices
# ----------------------------------------------------------------------
def compute_G_H(p, beta):
    N = len(p)
    H = np.zeros(N)
    G = np.zeros(N+1)
    prod = 1.0
    for k in range(N):
        H[k] = beta**(k+1) * prod
        G[k+1] = G[k] + (1 - p[k]) * H[k]
        prod *= p[k]
    return G, H


def closed_form_indices(p, beta, K, C, C_switch):
    N = len(p)
    G, H = compute_G_H(p, beta)

    # Passive states
    W_passive = np.zeros(N)
    for k in range(N):
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        W_passive[k] = K * (num / den) - C - C_switch

    # Active special state (0,1)
    if N >= 2:
        G1 = G[1]
        H1 = H[1]
    else:
        G1 = G[1]
        H1 = 0.0
    numerator = (1 - beta) * (K * G1 + C_switch * H1)
    denominator = beta * (1 - G1) - H1
    W_active = (numerator / denominator) - C if denominator != 0 else np.inf

    return W_passive, W_active




# ----------------------------------------------------------------------
# Generic average‑reward solvers (accept separate R0, R1)
# ----------------------------------------------------------------------
def solve_rvi(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=10000):
    """Relative Value Iteration for given passive/active rewards."""
    n = len(R0)
    V = np.zeros(n)
    for _ in range(max_iter):
        Q0 = R0 + P0 @ V
        Q1 = R1 - w + P1 @ V      # active reward = R1 - subsidy
        V_tilde = np.maximum(Q0, Q1)
        delta = V_tilde[ref]
        V_new = V_tilde - delta
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q0 = R0 + P0 @ V
    Q1 = R1 - w + P1 @ V
    g = delta   # average gain
    return V, g, Q0, Q1

def policy_evaluation(pi, R0, R1, P0, P1, w, ref=0):
    """Solve (I - P_pi) h + g*1 = r_pi with h[ref]=0."""
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

def solve_pi(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=100):
    """Policy iteration for average reward."""
    n = len(R0)
    pi = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        h, g = policy_evaluation(pi, R0, R1, P0, P1, w, ref)
        Q0 = R0 + P0 @ h
        Q1 = R1 - w + P1 @ h
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + P0 @ h
    Q1 = R1 - w + P1 @ h
    return h, g, Q0, Q1

def whittle_index_bisection(R0, R1, P0, P1, s, solver='rvi',
                            w_min=-2.0, w_max=2.0, tol_w=1e-10,
                            tol_mdp=1e-12, ref=0, verbose=False):
    """Find w such that Q0[s] - Q1[s] = 0."""
    solve_func = solve_rvi if solver == 'rvi' else solve_pi
    def f(w):
        _, _, Q0, Q1 = solve_func(R0, R1, P0, P1, w, ref=ref, tol=tol_mdp)
        return Q0[s] - Q1[s]

    f_min = f(w_min)
    f_max = f(w_max)
    if f_min * f_max > 0:
        expand = 2.0
        while f_min * f_max > 0:
            if abs(f_min) < abs(f_max):
                w_min -= expand
                f_min = f(w_min)
            else:
                w_max += expand
                f_max = f(w_max)
            if abs(w_min) > 100 or abs(w_max) > 100:
                raise RuntimeError(f"Could not bracket index for state {s}")
    for _ in range(60):
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
# Compute and print indices for both problems
# ----------------------------------------------------------------------
def compute_indices(R0, R1, P0, P1, expected, problem_name):
    print(f"\n=== {problem_name} ===")
    for method in ['rvi', 'pi']:
        print(f"--- {method.upper()} ---")
        indices = []
        for s in range(len(R0)):
            lam = whittle_index_bisection(R0, R1, P0, P1, s,
                                          solver=method,
                                          w_min=-2.0, w_max=2.0,
                                          tol_w=1e-10, tol_mdp=1e-12,
                                          ref=0)
            indices.append(lam)
            print(f"State {s+1}: λ = {lam:.10f}")
        diff = np.abs(np.array(indices) - np.array(expected))
        print(f"Indices: {[round(x,10) for x in indices]}")
        print(f"Max error: {np.max(diff):.2e}")
        print()


# ----------------------------------------------------------------------
# 5. Average cost solvers
# ----------------------------------------------------------------------
# def solve_rvi(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=10000):
#     n = len(R0)
#     V = np.zeros(n)
#     for _ in range(max_iter):
#         Q0 = R0 + P0 @ V
#         Q1 = R1 - w + P1 @ V
#         V_tilde = np.maximum(Q0, Q1)
#         delta = V_tilde[ref]
#         V_new = V_tilde - delta
#         if np.max(np.abs(V_new - V)) < tol:
#             V = V_new
#             break
#         V = V_new
#     Q0 = R0 + P0 @ V
#     Q1 = R1 - w + P1 @ V
#     return V, delta, Q0, Q1


# def policy_evaluation(pi, R0, R1, P0, P1, w, ref=0):
#     n = len(R0)
#     P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(n)])
#     r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(n)])
#     A = np.zeros((n+1, n+1))
#     A[:n, :n] = np.eye(n) - P_pi
#     A[:n, -1] = 1.0
#     A[n, ref] = 1.0
#     b = np.zeros(n+1)
#     b[:n] = r_pi
#     sol = np.linalg.solve(A, b)
#     return sol[:n], sol[-1]  # h, g



# def policy_evaluation(pi, R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=10000):
#     """
#     Iterative policy evaluation for average reward (fixed policy pi).
#     Solves (I - P_pi)h + g*1 = r_pi with h[ref]=0 using relative value iteration.
#     """
#     n = len(R0)
#     P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(n)])
#     r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(n)])
    
#     V = np.zeros(n)
#     for _ in range(max_iter):
#         V_new = r_pi + P_pi @ V
#         delta = V_new[ref]
#         V_new = V_new - delta
#         if np.max(np.abs(V_new - V)) < tol:
#             V = V_new
#             break
#         V = V_new
#     h = V                     # relative values
#     g = r_pi[ref] + (P_pi @ h)[ref]   # average gain
#     return h, g



# def solve_pi(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=100):
#     n = len(R0)
#     pi = np.zeros(n, dtype=int)
#     for _ in range(max_iter):
#         h, g = policy_evaluation(pi, R0, R1, P0, P1, w, ref)
#         Q0 = R0 + P0 @ h
#         Q1 = R1 - w + P1 @ h
#         pi_new = (Q1 > Q0).astype(int)
#         if np.all(pi_new == pi):
#             break
#         pi = pi_new
#     Q0 = R0 + P0 @ h
#     Q1 = R1 - w + P1 @ h
#     return h, g, Q0, Q1


# def whittle_index_bisection(R0, R1, P0, P1, s, solver='rvi',
#                             w_min=-500.0, w_max=500.0, tol_w=1e-10,
#                             tol_mdp=1e-12, ref=0):
#     solve_func = solve_rvi if solver == 'rvi' else solve_pi
#     def f(w):
#         _, _, Q0, Q1 = solve_func(R0, R1, P0, P1, w, ref=ref, tol=tol_mdp)
#         return Q0[s] - Q1[s]

#     f_min = f(w_min)
#     f_max = f(w_max)
#     if f_min * f_max > 0:
#         expand = 2.0
#         while f_min * f_max > 0:
#             if abs(f_min) < abs(f_max):
#                 w_min -= expand
#                 f_min = f(w_min)
#             else:
#                 w_max += expand
#                 f_max = f(w_max)
#             if abs(w_min) > 1000 or abs(w_max) > 1000:
#                 raise RuntimeError(f"Could not bracket average index for state {s}")
#     for _ in range(60):
#         w_mid = (w_min + w_max) / 2.0
#         f_mid = f(w_mid)
#         if abs(f_mid) < tol_w:
#             return w_mid
#         if f_mid * f_min > 0:
#             w_min = w_mid
#             f_min = f_mid
#         else:
#             w_max = w_mid
#     return (w_min + w_max) / 2.0


# ----------------------------------------------------------------------
# 6. Main parametric study
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ===== Parameter ranges =====
    C_list = [5.0, 10.0, 20.0]
    K_list = [500.0, 1000.0, 2000.0]
    C_switch_list = [0.0, 5.0, 10.0, 20.0, 40.0, 200.0, 500.0]
    beta_list = [0.9, 0.95, 0.99]
    N = 4                           # number of non-terminal states
    num_p_matrices = 3              # number of different transition matrices

    # Generate multiple p vectors with different seeds
    p_matrices = []
    seeds = [42, 123, 999]
    for seed in seeds[:num_p_matrices]:
        np.random.seed(seed)
        p_vec = np.sort(np.random.uniform(0, 1.0, N))[::-1]
        p_vec = np.append(p_vec, 0.0)   # terminal zero
        p_matrices.append(p_vec)

    total_combinations = len(p_matrices) * len(C_list) * len(K_list) * len(C_switch_list) * len(beta_list)

    # ===== Open report file =====
    report_filename = "whittle_comparison_report.txt"
    with open(report_filename, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("WHITTLE INDICES COMPARISON: CLOSED-FORM vs PACKAGE\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write("Parameter ranges explored:\n")
        f.write(f"  C (activation cost): {C_list}\n")
        f.write(f"  K (penalty cost): {K_list}\n")
        f.write(f"  C_switch (switching cost): {C_switch_list}\n")
        f.write(f"  beta (discount factor): {beta_list}\n")
        f.write(f"  Number of transition matrices: {num_p_matrices}\n")
        f.write(f"  Total combinations: {total_combinations}\n\n")

        instance_counter = 0

        for t_idx, p_full in enumerate(p_matrices):
            f.write("#"*80 + "\n")
            f.write(f"TRANSITION MATRIX {t_idx+1}\n")
            f.write(f"Survival probabilities: {np.round(p_full, 4)}\n")
            f.write("#"*80 + "\n\n")

            for C in C_list:
                for K in K_list:
                    for C_switch in C_switch_list:
                        for beta in beta_list:
                            instance_counter += 1
                            print(f"Processing instance {instance_counter}/{total_combinations}...", end='\r')
                            sys.stdout.flush()

                            # Build MDP
                            P0, P1, R0, R1, state_labels = build_extended_mdp(N, p_full, C, K, C_switch)

                            # ===== Write instance header =====
                            f.write("="*80 + "\n")
                            f.write(f"INSTANCE {instance_counter}\n")
                            f.write("="*80 + "\n")
                            f.write("Parameters:\n")
                            f.write(f"  N = {N}\n")
                            f.write(f"  Survival probabilities p[0..{N}]: {np.round(p_full, 4)}\n")
                            f.write(f"  C = {C}\n")
                            f.write(f"  K = {K}\n")
                            f.write(f"  C_switch = {C_switch}\n")
                            f.write(f"  beta = {beta}\n")
                            f.write("-"*80 + "\n")

                            # Print transition matrices (compact)
                            f.write("\nTransition matrix for action Passive (a=0):\n")
                            df_p0 = pd.DataFrame(P0, index=state_labels, columns=state_labels)
                            f.write(df_p0.to_string())
                            f.write(f"\n\nReward vector for action Passive (a=0):\n{R0}\n")

                            f.write("\nTransition matrix for action Active (a=1):\n")
                            df_p1 = pd.DataFrame(P1, index=state_labels, columns=state_labels)
                            f.write(df_p1.to_string())
                            f.write(f"\n\nReward vector for action Active (a=1):\n{R1}\n\n")

                            # ---- Discounted indices ----
                            # Closed-form
                            W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
                            closed_all = np.concatenate(([W_active], W_passive))

                            # VI & PI discounted
                            vi_disc = []
                            pi_disc = []
                            for s in range(len(R0)):
                                lam_vi = whittle_index_discounted(s, solver='vi', beta=beta,
                                                                  w_min=-200.0, w_max=200.0,
                                                                  tol_w=1e-5, tol_mdp=1e-8,
                                                                  P0=P0, P1=P1, R0=R0, R1=R1)
                                lam_pi = whittle_index_discounted(s, solver='pi', beta=beta,
                                                                  w_min=-200.0, w_max=200.0,
                                                                  tol_w=1e-5, tol_mdp=1e-8,
                                                                  P0=P0, P1=P1, R0=R0, R1=R1)
                                vi_disc.append(lam_vi)
                                pi_disc.append(lam_pi)

                            # Package discounted
                            if PKG_AVAILABLE:
                                model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                                pkg_disc = model.whittle_indices(discount=beta)
                            else:
                                pkg_disc = None

                            # Write discounted comparison
                            f.write("Closed-form Whittle indices (discounted, beta = {}):\n".format(beta))
                            f.write(f"  Active state (0,1): W = {W_active:.6f}\n")
                            for k in range(N+1):
                                f.write(f"  Passive state ({k},0): W = {W_passive[k]:.6f}\n")

                            f.write("\nVI discounted indices:\n")
                            for lbl, lam in zip(state_labels, vi_disc):
                                f.write(f"  {lbl}: {lam:.6f}\n")
                            f.write("\nPI discounted indices:\n")
                            for lbl, lam in zip(state_labels, pi_disc):
                                f.write(f"  {lbl}: {lam:.6f}\n")

                            if pkg_disc is not None:
                                f.write("\nPackage discounted indices (beta = {}):\n".format(beta))
                                for lbl, lam in zip(state_labels, pkg_disc):
                                    f.write(f"  {lbl}: {lam:.6f}\n")
                                f.write("\nDifferences (closed-form - package):\n")
                                f.write(f"  Active (0,1): {closed_all[0] - pkg_disc[0]:.2e}\n")
                                for k in range(1, len(state_labels)):
                                    diff = closed_all[k] - pkg_disc[k]
                                    f.write(f"  Passive ({k-1},0): {diff:.2e}\n")

                            # ---- Average cost indices ----
                            f.write("\n" + "-"*80 + "\n")
                            f.write("Average-cost indices:\n")
                            # RVI
                            rvi_avg = []
                            for s in range(len(R0)):
                                try:
                                    lam = whittle_index_bisection(R0, R1, P0, P1, s, solver='rvi',
                                                                  w_min=-500.0, w_max=500.0,
                                                                  tol_w=1e-10, tol_mdp=1e-12, ref=0)
                                except RuntimeError as e:
                                    lam = np.nan
                                rvi_avg.append(lam)
                            # PI
                            pi_avg = []
                            for s in range(len(R0)):
                                try:
                                    lam = whittle_index_bisection(R0, R1, P0, P1, s, solver='pi',
                                                                  w_min=-500.0, w_max=500.0,
                                                                  tol_w=1e-10, tol_mdp=1e-12, ref=0)
                                except RuntimeError:
                                    lam = np.nan
                                pi_avg.append(lam)

                            # Package average
                            if PKG_AVAILABLE:
                                pkg_avg = model.whittle_indices(discount=0.0)
                            else:
                                pkg_avg = None

                            f.write("\nRVI average indices:\n")
                            for lbl, lam in zip(state_labels, rvi_avg):
                                f.write(f"  {lbl}: {lam:.6f}\n")
                            f.write("\nPI average indices:\n")
                            for lbl, lam in zip(state_labels, pi_avg):
                                f.write(f"  {lbl}: {lam:.6f}\n")
                            if pkg_avg is not None:
                                f.write("\nPackage average-cost indices:\n")
                                for lbl, lam in zip(state_labels, pkg_avg):
                                    f.write(f"  {lbl}: {lam:.6f}\n")
                                f.write("\nDifferences (RVI - package):\n")
                                for lbl, rvi, pkg in zip(state_labels, rvi_avg, pkg_avg):
                                    if not np.isnan(rvi):
                                        f.write(f"  {lbl}: {rvi - pkg:.2e}\n")
                                    else:
                                        f.write(f"  {lbl}: RVI failed\n")

                            f.write("\n\n")   # blank line between instances

        print("\nDone. Report saved to", report_filename)
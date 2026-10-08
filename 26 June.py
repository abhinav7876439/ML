import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import bisect
import networkx as nx
from scipy.optimize import brentq
# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")

# ----------------------------------------------------------------------
# 1. MDP builder
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
# 2. Discounted Value Iteration
# ----------------------------------------------------------------------
def solve_vi_discounted(P0: np.ndarray, P1: np.ndarray,
                        R0: np.ndarray, R1: np.ndarray,
                        w: float, beta: float,
                        tol: float = 1e-8, max_iter: int = 10000
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ----------------------------------------------------------------------
# 3. Discounted Policy Iteration
# ----------------------------------------------------------------------
def policy_evaluation_discounted(pi: np.ndarray,
                                 P0: np.ndarray, P1: np.ndarray,
                                 R0: np.ndarray, R1: np.ndarray,
                                 w: float, beta: float
                                 ) -> np.ndarray:
    S = len(pi)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])

    I = np.eye(S)
    V = np.linalg.solve(I - beta * P_pi, r_pi)
    return V


def solve_pi_discounted(P0: np.ndarray, P1: np.ndarray,
                        R0: np.ndarray, R1: np.ndarray,
                        w: float, beta: float,
                        tol: float = 1e-8, max_iter: int = 1000
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# 4. Binary search for Whittle index (discounted)
# ----------------------------------------------------------------------
def q_diff_discounted(state: int, w: float,
                      solver: str = 'vi',
                      beta: float = 0.9,
                      P0=None, P1=None, R0=None, R1=None,
                      tol: float = 1e-8) -> float:
    if solver == 'vi':
        _, Q0, Q1 = solve_vi_discounted(P0, P1, R0, R1, w, beta, tol)
    elif solver == 'pi':
        _, Q0, Q1 = solve_pi_discounted(P0, P1, R0, R1, w, beta, tol)
    else:
        raise ValueError("solver must be 'vi' or 'pi'")
    return Q0[state] - Q1[state]


def whittle_index_discounted(state: int,
                             solver: str = 'vi',
                             beta: float = 0.9,
                             w_min: float = -100.0, w_max: float = 100.0,
                             tol_w: float = 1e-4, tol_mdp: float = 1e-8,
                             P0=None, P1=None, R0=None, R1=None,
                             verbose: bool = False) -> float:
    f_min = q_diff_discounted(state, w_min, solver, beta, P0, P1, R0, R1, tol_mdp)
    f_max = q_diff_discounted(state, w_max, solver, beta, P0, P1, R0, R1, tol_mdp)

    expand = 1.5
    while f_min * f_max > 0:
        if verbose:
            print(f"Expanding bracket: f({w_min})={f_min:.4f}, f({w_max})={f_max:.4f}")
        if abs(f_min) < abs(f_max):
            w_min -= expand * abs(w_min - w_max) if abs(w_min - w_max) > 0 else 10.0
            f_min = q_diff_discounted(state, w_min, solver, beta, P0, P1, R0, R1, tol_mdp)
        else:
            w_max += expand * abs(w_max - w_min)
            f_max = q_diff_discounted(state, w_max, solver, beta, P0, P1, R0, R1, tol_mdp)
        if abs(w_min) > 1e6 or abs(w_max) > 1e6:
            raise ValueError(f"Could not find bracket for state {state}")

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
            f_max = f_mid

    return (w_min + w_max) / 2.0


# ----------------------------------------------------------------------
# Closed Form Whittle Indices (Discounted)
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
# Average reward solvers
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


def policy_evaluation(pi, R0, R1, P0, P1, w, ref=0):
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
# Main comparison
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Problem parameters
    N = 5
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)                       # terminal zero for the extended MDP
    C, K = 5.0, 500.0
    C_switch = 1.0               ## 1.0, 10.0 (Discounted correct) whereas 100.0 (Last 3 states matches) 
    beta = 0.9999                           # discount factor

    # C, K = 5.0, 10.0
    # C_switch = 5.0
    # beta = 0.9999                           # discount factor

    print("=" * 70)
    print(f"Survival probabilities p[0..{N}]: {np.round(p_full, 3)}")
    print("=" * 70)

    # Build MDP
    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)

    # Print matrices
    print_transition_matrix(P0_ext, state_labels_ext, "Passive (a=0)")
    print_reward_vector(R0_ext, state_labels_ext, "Passive (a=0)")
    print_transition_matrix(P1_ext, state_labels_ext, "Active (a=1)")
    print_reward_vector(R1_ext, state_labels_ext, "Active (a=1)")

    # ----- Discounted Indices -----
    print("\n" + "="*70)
    print("DISCOUNTED WHITTLE INDICES (beta = {})".format(beta))
    print("="*70)

    # Closed form
    W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
    closed_form_arr = np.concatenate(([W_active], W_passive))  # length N+2

    # Value Iteration
    print("Computing VI discounted indices...")
    vi_disc = []
    for s in range(len(R0_ext)):
        lam = whittle_index_discounted(state=s, solver='vi', beta=beta,
                                       w_min=-50.0, w_max=50.0,
                                       tol_w=1e-5, tol_mdp=1e-8,
                                       P0=P0_ext, P1=P1_ext, R0=R0_ext, R1=R1_ext,
                                       verbose=False)
        vi_disc.append(lam)
        print(f"  State {state_labels_ext[s]:8s}: λ = {lam:8.4f}")

    # Policy Iteration
    print("Computing PI discounted indices...")
    pi_disc = []
    for s in range(len(R0_ext)):
        lam = whittle_index_discounted(state=s, solver='pi', beta=beta,
                                       w_min=-50.0, w_max=50.0,
                                       tol_w=1e-5, tol_mdp=1e-8,
                                       P0=P0_ext, P1=P1_ext, R0=R0_ext, R1=R1_ext,
                                       verbose=False)
        pi_disc.append(lam)
        print(f"  State {state_labels_ext[s]:8s}: λ = {lam:8.4f}")

    # Package discounted
    if PKG_AVAILABLE:
        model = bandit.restless_bandit_from_P0P1_R0R1(P0_ext, P1_ext, R0_ext, R1_ext)
        pkg_disc = model.whittle_indices(discount=beta)
        print("\nPackage discounted indices computed.")
    else:
        pkg_disc = None

    # # ----- Average Cost Indices -----
    # print("\n" + "="*70)
    # print("AVERAGE COST WHITTLE INDICES")
    # print("="*70)

    # # Value Iteration (RVI)
    # print("Computing RVI average indices...")
    # rvi_avg = []
    # for s in range(len(R0_ext)):
    #     lam = whittle_index_bisection(R0_ext, R1_ext, P0_ext, P1_ext, s,
    #                                   solver='rvi',
    #                                   w_min=-2.0, w_max=2.0,
    #                                   tol_w=1e-10, tol_mdp=1e-12,
    #                                   ref=0)
    #     rvi_avg.append(lam)
    #     print(f"  State {state_labels_ext[s]:8s}: λ = {lam:.6f}")

    # # Policy Iteration (PI)
    # print("Computing PI average indices...")
    # pi_avg = []
    # for s in range(len(R0_ext)):
    #     lam = whittle_index_bisection(R0_ext, R1_ext, P0_ext, P1_ext, s,
    #                                   solver='pi',
    #                                   w_min=-2.0, w_max=2.0,
    #                                   tol_w=1e-10, tol_mdp=1e-12,
    #                                   ref=0)
    #     pi_avg.append(lam)
    #     print(f"  State {state_labels_ext[s]:8s}: λ = {lam:.6f}")


    # ----- Average Cost Indices -----
    print("\n" + "="*70)
    print("AVERAGE COST WHITTLE INDICES")
    print("="*70)

    # Value Iteration (RVI)
    print("Computing RVI average indices...")
    rvi_avg = []
    for s in range(len(R0_ext)):
        lam = whittle_index_bisection(R0_ext, R1_ext, P0_ext, P1_ext, s,
                                      solver='rvi',
                                      w_min=-500.0, w_max=500.0,   # wide bracket
                                      tol_w=1e-10, tol_mdp=1e-12,
                                      ref=0)
        rvi_avg.append(lam)
        print(f"  State {state_labels_ext[s]:8s}: λ = {lam:.6f}")

    # Policy Iteration (PI)
    print("Computing PI average indices...")
    pi_avg = []
    for s in range(len(R0_ext)):
        lam = whittle_index_bisection(R0_ext, R1_ext, P0_ext, P1_ext, s,
                                      solver='pi',
                                      w_min=-500.0, w_max=500.0,   # wide bracket
                                      tol_w=1e-10, tol_mdp=1e-12,
                                      ref=0)
        pi_avg.append(lam)
        print(f"  State {state_labels_ext[s]:8s}: λ = {lam:.6f}")

    # Package average
    if PKG_AVAILABLE:
        pkg_avg = model.whittle_indices(discount=0.0)
        print("\nPackage average‑cost indices computed.")
    else:
        pkg_avg = None

    # ----- Comparison Tables -----
    # Discounted table
    discounted_data = {
        'State': state_labels_ext,
        'ClosedForm': closed_form_arr,
        'VI': vi_disc,
        'PI': pi_disc,
    }
    if PKG_AVAILABLE and pkg_disc is not None:
        discounted_data['Package'] = pkg_disc
    df_disc = pd.DataFrame(discounted_data)
    df_disc = df_disc.set_index('State')
    print("\n" + "="*70)
    print("DISCOUNTED INDICES COMPARISON")
    print("="*70)
    print(df_disc)

    # Average cost table
    average_data = {
        'State': state_labels_ext,
        'RVI': rvi_avg,
        'PI': pi_avg,
    }
    if PKG_AVAILABLE and pkg_avg is not None:
        average_data['Package'] = pkg_avg
    df_avg = pd.DataFrame(average_data)
    df_avg = df_avg.set_index('State')
    print("\n" + "="*70)
    print("AVERAGE COST INDICES COMPARISON")
    print("="*70)
    print(df_avg)


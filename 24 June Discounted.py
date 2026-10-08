


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
# 1. Your original MDP builder (unchanged)
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
    """
    Discounted infinite‑horizon value iteration for a given subsidy w.
    Returns optimal value function V, and Q‑factors Q0, Q1.
    """
    S = len(R0)
    V = np.zeros(S)

    for _ in range(max_iter):
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        V_new = np.maximum(Q0, Q1)

        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new

    # Final Q‑values with the converged V
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
    """
    Evaluate a deterministic policy pi (0/1) for the discounted case.
    Returns value function V (no differential/gain needed).
    """
    S = len(pi)
    # Build transition and reward for current policy
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])

    # Solve (I - beta*P_pi) * V = r_pi
    I = np.eye(S)
    V = np.linalg.solve(I - beta * P_pi, r_pi)
    return V


def solve_pi_discounted(P0: np.ndarray, P1: np.ndarray,
                        R0: np.ndarray, R1: np.ndarray,
                        w: float, beta: float,
                        tol: float = 1e-8, max_iter: int = 1000
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discounted policy iteration. Returns V, Q0, Q1.
    """
    S = len(R0)
    pi = np.zeros(S, dtype=int)  # start with passive everywhere

    for _ in range(max_iter):
        # 1. Policy evaluation
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta)

        # 2. Policy improvement
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        pi_new = (Q1 > Q0).astype(int)   # active if Q1 strictly larger

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
    """
    Returns Q0(state) - Q1(state) for a given w.
    solver = 'vi' (value iteration) or 'pi' (policy iteration).
    """
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
    """
    Discounted Whittle index for a given state via bisection.
    """
    # Ensure the bracket contains the root
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

    # Bisection
    for _ in range(50):   # 50 iterations -> 2^-50 precision
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

# def compute_G_H(p, beta):
#     """
#     Compute the G and H sequences defined in the paper.
    
#     Parameters:
#     p : array of length N, survival probabilities p[0]..p[N-1]
#     beta : discount factor (0 < beta < 1)
    
#     Returns:
#     G : array of length N+1, G[0]=0, G[k+1] = G[k] + (1-p[k]) * H[k]
#     H : array of length N, H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
#     """
#     N = len(p)
#     H = np.zeros(N)
#     G = np.zeros(N+1)          # G[0] = 0
#     prod = 1.0
#     for k in range(N):
#         # H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
#         H[k] = beta**(k+1) * prod
#         G[k+1] = G[k] + (1 - p[k]) * H[k]
#         prod *= p[k]           # update product for next iteration
#     return G, H

# def closed_form_indices(p, beta, K, C, C_switch):
#     """
#     Compute closed-form Whittle indices for passive states (k,0) and the
#     special active state (0,1).
    
#     Returns:
#     W_passive : array of length N, indices for k = 0..N-1
#     W_active_special : float, index for state (0,1)
#     """
#     N = len(p)  
#     G, H = compute_G_H(p, beta)
    
#     # Passive states (k,0)
#     W_passive = np.zeros(N)
#     for k in range(N):
#         # Using formulas from the paper
#         num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
#         den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
#         # Avoid division by zero (should not happen for valid parameters)
#         W_passive[k] = K * (num / den) - C - C_switch
    
#     # Special active state (0,1) – using k = 0 in the active-state formula
#     # i.e., G[1] and H[1] (note H[1] exists only if N >= 2; for N=1 we treat H[1]=0)
#     if N >= 2:
#         G1 = G[1]
#         H1 = H[1]
#     else:
#         # For N=1 there is only p[0]; H[1] would be beta^2 * p[0] but not defined.
#         # In the paper, the sum for the active state uses H(2) if needed.
#         # For simplicity we assume N>=2; otherwise adjust accordingly.
#         # Here we set H1 = 0 to avoid errors, but the formula may not be accurate.
#         G1 = G[1]
#         H1 = 0.0
    
#     numerator = (1 - beta) * (K * G1 + C_switch * H1)
#     denominator = beta * (1 - G1) - H1
#     W_active = (numerator / denominator) - C if denominator != 0 else np.inf
    
#     return W_passive, W_active



def compute_G_H(p, beta):
    """
    Compute G and H with length extended to include the terminal state.
    p : array of length N (survival probabilities, no terminal zero)
    Returns:
      G : length N+2 (G[0]..G[N+1])
      H : length N+1 (H[0]..H[N])
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
    Compute closed-form Whittle indices for ALL states of the extended MDP.
    p : length N (survival probabilities without the trailing zero)
    Returns:
      W_active   : index for state (0,1)
      W_passive  : array length N+1, indices for states (0,0),(1,0),...,(N,0)
    """
    N = len(p)
    G, H = compute_G_H(p, beta)

    # Special active state (0,1)
    G1 = G[1]
    H1 = H[1] if N >= 1 else 0.0   # H[1] defined for N>=1
    numerator = (1 - beta) * (K * G1 + C_switch * H1)
    denominator = beta * (1 - G1) - H1
    W_active = (numerator / denominator) - C if denominator != 0 else np.inf

    # Passive states (k,0) for k = 0..N
    W_passive = np.zeros(N+1)
    for k in range(N+1):
        if k < N:
            p_k = p[k]
        else:
            p_k = 0.0
        num = 1 - G[k+1] - p_k * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p_k * (1 - G[k])
        W_passive[k] = K * (num / den) - C - C_switch
    return W_active, W_passive



## Average Cost Whittle Indices (Average Reward)

# ----------------------------------------------------------------------
# Generic average‑reward solvers
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
# 5. Example usage
# ----------------------------------------------------------------------




if __name__ == "__main__":
    # Problem parameters
    N = 5
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)                       # terminal zero for the extended MDP
    C, K = 5.0, 500.0
    C_switch = 1.0
    beta = 0.9999                           # discount factor

    print("=" * 70)
    print(f"Survival probabilities p[0..{N}]: {np.round(p_full, 3)}")
    print("=" * 70)

    # Build MDP
    # P0, P1, R0, R1, state_labels = build_extended_mdp(N, p_full, C, K, C_switch)
    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)


    # Print matrices with state labels both vertically and horizontally
    print_transition_matrix(P0_ext, state_labels_ext, "Passive (a=0)")
    print_reward_vector(R0_ext, state_labels_ext, "Passive (a=0)")

    print_transition_matrix(P1_ext, state_labels_ext, "Active (a=1)")
    print_reward_vector(R1_ext, state_labels_ext, "Active (a=1)")

    print("Discounted Whittle Index Computation")
    print(f"beta = {beta}")
    print("States:", state_labels_ext)
    print()

    # Compute with Value Iteration
    print("--- Using Value Iteration (Discounted) ---")
    indices_vi = []
    for s in range(len(R0_ext)):
        lam = whittle_index_discounted(state=s, solver='vi', beta=beta,
                                       w_min=-50.0, w_max=50.0,
                                       tol_w=1e-5, tol_mdp=1e-8,
                                       P0=P0_ext, P1=P1_ext, R0=R0_ext, R1=R1_ext,
                                       verbose=False)
        indices_vi.append(lam)
        print(f"State {state_labels_ext[s]:8s}: λ = {lam:8.4f}")

    print()

    # Compute with Policy Iteration
    print("--- Using Policy Iteration (Discounted) ---")
    indices_pi = []
    for s in range(len(R0_ext)):
        lam = whittle_index_discounted(state=s, solver='pi', beta=beta,
                                       w_min=-50.0, w_max=50.0,
                                       tol_w=1e-5, tol_mdp=1e-8,
                                       P0=P0_ext, P1=P1_ext, R0=R0_ext, R1=R1_ext,
                                       verbose=False)
        indices_pi.append(lam)
        print(f"State {state_labels_ext[s]:8s}: λ = {lam:8.4f}")

    print()


    # Closed‑form indices
    #W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
    W_passive, W_active = closed_form_indices(p, beta, K, C, C_switch)
    
    print("\nClosed‑form Whittle indices (discounted, beta = {}):".format(beta))
    print(f"  Active state (0,1): W = {W_active:.6f}")
    for k in range(N+1):
        print(f"  Passive state ({k},0): W = {W_passive[k]:.6f}")


    # Compute with Value Iteration
    print("--- Using Value Iteration (Average) ---")
    indices_vi = []
    # for s in range(len(R0_ext)):
    #     lam = whittle_index_discounted(state=s, solver='vi', beta=beta,
    #                                    w_min=-50.0, w_max=50.0,
    #                                    tol_w=1e-5, tol_mdp=1e-8,
    #                                    P0=P0_ext, P1=P1_ext, R0=R0_ext, R1=R1_ext,
    #                                    verbose=False)
    #     indices_vi.append(lam)
    #     print(f"State {state_labels_ext[s]:8s}: λ = {lam:8.4f}")

    # print()

    compute_indices(R0_ext, R1_ext, P0_ext, P1_ext, [0.0]*(N+2), "Extended MDP (no expected values)")
    
    # Compare with markovianbandit package if available
    if PKG_AVAILABLE:
        # Build the extended MDP using the full p (including terminal zero)
        P0, P1, R0, R1, _ = build_extended_mdp(N, p_full, C, K, C_switch)
        
        # Create restless bandit model
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        
        # Discounted indices from the package
        pkg_disc = model.whittle_indices(discount=beta)
        pkg_passive = pkg_disc[1:N+2]          # states (0,0) ... (N-1,0)
        pkg_active_special = pkg_disc[0]       # state (0,1)
        
        print("\nPackage discounted indices (beta = {}):".format(beta))
        print(f"  Active state (0,1): W = {pkg_active_special:.6f}")
        for k in range(N+1):
            print(f"  Passive state ({k},0): W = {pkg_passive[k]:.6f}")
        
        # Differences
        print("\nDifferences (closed‑form - package):")
        print(f"  Active (0,1): {W_active - pkg_active_special:.2e}")
        for k in range(N+1):
            diff = W_passive[k] - pkg_passive[k]
            print(f"  Passive ({k},0): {diff:.2e}")
        
        # (Optional) Average‑cost indices (discount=0)
        pkg_avg = model.whittle_indices(discount=0.0)
        pkg_avg_passive = pkg_avg[1:N+2]
        pkg_avg_active_special = pkg_avg[0]
        
        print("\nPackage average‑cost indices:")
        print(f"  Active state (0,1): W = {pkg_avg_active_special:.4f}")
        for k, val in enumerate(pkg_avg_passive):
            print(f"  Passive state ({k},0): W = {val:.4f}")
        print("=" * 60)


    # Comparison table
    df = pd.DataFrame({'State': state_labels_ext,
                       'VI_index': indices_vi,
                       'PI_index': indices_pi})
    df['Abs_diff'] = np.abs(df['VI_index'] - df['PI_index'])
    print("--- Comparison ---")
    print(df)
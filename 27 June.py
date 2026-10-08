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

def build_extended_mdp(N, p, C, K, C_switch):
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    # State labels
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
            P0[s, 1] = 1-p[0]
            R0[s] = -(K*(1-p[0]))
        elif s == 1:
            P0[s, 2] = p[0]
            P0[s, 1] = 1-p[0]
            R0[s] = -(K*(1-p[0]))
        else:
            idx = s-1
            if idx < N:
                P0[s, s+1] = p[idx]
                P0[s, 1] = 1-p[idx]
                R0[s] = -(K*(1-p[idx]))
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

def compute_G_H(p, beta):
    """
    Compute the G and H sequences defined in the paper.
    
    Parameters:
    p : array of length N, survival probabilities p[0]..p[N-1]
    beta : discount factor (0 < beta < 1)
    
    Returns:
    G : array of length N+1, G[0]=0, G[k+1] = G[k] + (1-p[k]) * H[k]
    H : array of length N, H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
    """
    N = len(p)
    H = np.zeros(N)
    G = np.zeros(N+1)          # G[0] = 0
    prod = 1.0
    for k in range(N):
        # H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
        H[k] = beta**(k+1) * prod
        G[k+1] = G[k] + (1 - p[k]) * H[k]
        prod *= p[k]           # update product for next iteration
    return G, H

def closed_form_indices(p, beta, K, C, C_switch):
    """
    Compute closed-form Whittle indices for passive states (k,0) and the
    special active state (0,1).
    
    Returns:
    W_passive : array of length N, indices for k = 0..N-1
    W_active_special : float, index for state (0,1)
    """
    N = len(p)  
    G, H = compute_G_H(p, beta)
    
    # Passive states (k,0)
    W_passive = np.zeros(N)
    for k in range(N):
        # Using formulas from the paper
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        # Avoid division by zero (should not happen for valid parameters)
        W_passive[k] = K * (num / den) - C - C_switch
    
    # Special active state (0,1) – using k = 0 in the active-state formula
    # i.e., G[1] and H[1] (note H[1] exists only if N >= 2; for N=1 we treat H[1]=0)
    if N >= 2:
        G1 = G[1]
        H1 = H[1]
    else:
        # For N=1 there is only p[0]; H[1] would be beta^2 * p[0] but not defined.
        # In the paper, the sum for the active state uses H(2) if needed.
        # For simplicity we assume N>=2; otherwise adjust accordingly.
        # Here we set H1 = 0 to avoid errors, but the formula may not be accurate.
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
                lam = whittle_index_bisection(R0, R1, P0, P1, s,
                                              solver=method,
                                              w_min=-1000.0, w_max=1000.0,
                                              tol_w=1e-10, tol_mdp=1e-12,
                                              ref=0)
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

   

# # ===================================================================
# # CORRECTED PARAMETER SWEEP AND FILE OUTPUT
# # ===================================================================

# import datetime


# # ----------------------------------------------------------------------
# # Helper: write a matrix or vector to file as a formatted table
# # ----------------------------------------------------------------------
# def print_matrix_to_file(f, matrix, labels, title):
#     """Write a labelled matrix to file."""
#     n = len(labels)
#     header = "       " + "".join(f"{l:>10s}" for l in labels)
#     f.write(title + "\n")
#     f.write(header + "\n")
#     for i, row in enumerate(matrix):
#         f.write(f"{labels[i]:>6s}  " + "".join(f"{x:10.6f}" for x in row) + "\n")
#     f.write("\n")

# def print_vector_to_file(f, vector, labels, title):
#     """Write a labelled vector to file."""
#     f.write(title + "\n")
#     for i, (lbl, val) in enumerate(zip(labels, vector)):
#         f.write(f"  {lbl:6s} : {val:12.6f}\n")
#     f.write("\n")

# def print_table_to_file(f, df, title):
#     """Write a pandas DataFrame to file with a title."""
#     f.write(title + "\n")
#     f.write(df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if not np.isnan(x) else "     nan") + "\n\n")


# # ----------------------------------------------------------------------
# # Parameter ranges
# # ----------------------------------------------------------------------
# C_vals       = [5.0, 10.0, 20.0]
# K_vals       = [500.0, 1000.0, 2000.0]
# C_switch_vals = [0.0, 5.0, 10.0, 20.0, 40.0, 200.0, 500.0]
# beta_vals    = [0.9, 0.95, 0.99]

# N_live = 5                     # number of live stages (p[0..4])
# N_states = N_live + 2           # total states in extended MDP

# seeds = [42, 123, 456]
# p_matrices = {}
# for i, seed in enumerate(seeds):
#     np.random.seed(seed)
#     p_matrices[i] = np.sort(np.random.uniform(0.1, 0.99, N_live))[::-1]  # decreasing survival
#     print(f"\n Transition matrix {i+1} (seed {seed}): p = {np.round(p_matrices[i], 4)}")

# # ----------------------------------------------------------------------
# # Open output file
# # ----------------------------------------------------------------------
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# filename = f"whittle_results_{timestamp}.txt"
# f = open(filename, "w", encoding="utf-8")

# # Header
# f.write("="*80 + "\n")
# f.write("WHITTLE INDICES COMPARISON: CLOSED-FORM vs PACKAGE\n")
# f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
# f.write("="*80 + "\n\n")
# f.write("Parameter ranges explored:\n")
# f.write(f"  C (activation cost): {C_vals}\n")
# f.write(f"  K (penalty cost): {K_vals}\n")
# f.write(f"  C_switch (switching cost): {C_switch_vals}\n")
# f.write(f"  beta (discount factor): {beta_vals}\n")
# f.write(f"  Number of transition matrices: {len(seeds)}\n")
# f.write(f"  Total discounted combinations: {len(seeds)*len(C_vals)*len(K_vals)*len(C_switch_vals)*len(beta_vals)}\n")
# f.write(f"  Total average-cost combinations: {len(seeds)*len(C_vals)*len(K_vals)*len(C_switch_vals)}\n\n")

# # ----------------------------------------------------------------------
# # Loop over all instances
# # ----------------------------------------------------------------------
# disc_total = len(seeds) * len(C_vals) * len(K_vals) * len(C_switch_vals) * len(beta_vals)
# avg_total = len(seeds) * len(C_vals) * len(K_vals) * len(C_switch_vals)
# disc_count = 0
# avg_count = 0

# for p_id in range(len(seeds)):
#     p_full = np.append(p_matrices[p_id], 0.0)
#     # State labels for extended MDP
#     labels = [f"(0,1)", f"(0,0)"] + [f"({i},0)" for i in range(1, N_live+1)]

#     # ---- Write transition matrix section ----
#     f.write(f"\n{'#'*80}\n")
#     f.write(f"TRANSITION MATRIX {p_id+1}\n")
#     f.write(f"Survival probabilities: {np.round(p_full, 4)}\n")
#     f.write(f"{'#'*80}\n")

#     # ---- DISCOUNTED COST ----
#     for C in C_vals:
#         for K in K_vals:
#             for C_switch in C_switch_vals:
#                 for beta in beta_vals:
#                     disc_count += 1
#                     print(f"[Discounted {disc_count}/{disc_total}] "
#                           f"Matrix {p_id+1}, C={C}, K={K}, C_switch={C_switch}, beta={beta}")

#                     # Build MDP
#                     P0, P1, R0, R1, _ = build_extended_mdp(N_live, p_full, C, K, C_switch)

#                     # Write instance header
#                     f.write(f"\n{'='*80}\n")
#                     f.write(f"INSTANCE (Discounted) – Matrix {p_id+1}, C={C}, K={K}, C_switch={C_switch}, β={beta}\n")
#                     f.write(f"{'='*80}\n")
#                     f.write(f"Parameters:\n")
#                     f.write(f"  N = {N_live}\n")
#                     f.write(f"  Survival probabilities p[0..{N_live}]: {np.round(p_full, 4)}\n")
#                     f.write(f"  C = {C}\n")
#                     f.write(f"  K = {K}\n")
#                     f.write(f"  C_switch = {C_switch}\n")
#                     f.write(f"  beta = {beta}\n")
#                     f.write(f"{'-'*80}\n")

#                     # Print transition matrices and rewards
#                     print_matrix_to_file(f, P0, labels, "Transition matrix for action Passive (a=0):")
#                     print_vector_to_file(f, R0, labels, "Reward vector for action Passive (a=0):")
#                     print_matrix_to_file(f, P1, labels, "Transition matrix for action Active (a=1):")
#                     print_vector_to_file(f, R1, labels, "Reward vector for action Active (a=1):")

#                     # --- Discounted Whittle Indices ---
#                     f.write(f"--- Discounted Whittle Indices (beta = {beta}) ---\n")

#                     # Compute all indices
#                     disc_vi = []
#                     disc_pi = []
#                     for s in range(len(labels)):
#                         try:
#                             disc_vi.append(whittle_index_discounted(s, solver='vi', beta=beta, P0=P0, P1=P1, R0=R0, R1=R1))
#                         except:
#                             disc_vi.append(np.nan)
#                         try:
#                             disc_pi.append(whittle_index_discounted(s, solver='pi', beta=beta, P0=P0, P1=P1, R0=R0, R1=R1))
#                         except:
#                             disc_pi.append(np.nan)

#                     # Closed form (using corrected function with p_full)
#                     try:
#                         W_active, W_passive = closed_form_indices(p_full, beta, K, C, C_switch)
#                         disc_closed = [W_active] + list(W_passive)
#                     except:
#                         disc_closed = [np.nan] * len(labels)

#                     # Package (if available)
#                     disc_pkg = [np.nan] * len(labels)
#                     if PKG_AVAILABLE:
#                         try:
#                             model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#                             disc_pkg = model.whittle_indices(discount=beta)
#                         except Exception as e:
#                             print(f"  Package error: {e}")

#                     # Build DataFrame and write
#                     df_disc = pd.DataFrame({
#                         'State': labels,
#                         'ClosedForm': disc_closed,
#                         'VI': disc_vi,
#                         'PI': disc_pi,
#                         'Package': disc_pkg
#                     })
#                     print_table_to_file(f, df_disc, "Discounted Indices:")

#                     # --- Average-Cost Whittle Indices (same MDP, no beta) ---
#                     f.write(f"--- Average-Cost Whittle Indices ---\n")
#                     avg_vi = []
#                     avg_pi = []
#                     for s in range(len(labels)):
#                         try:
#                             avg_vi.append(whittle_index_bisection(R0, R1, P0, P1, s, solver='rvi', ref=0))
#                         except:
#                             avg_vi.append(np.nan)
#                         try:
#                             avg_pi.append(whittle_index_bisection(R0, R1, P0, P1, s, solver='pi', ref=0))
#                         except:
#                             avg_pi.append(np.nan)

#                     avg_pkg = [np.nan] * len(labels)
#                     if PKG_AVAILABLE:
#                         try:
#                             avg_pkg = model.whittle_indices()  # average-cost
#                         except:
#                             pass

#                     df_avg = pd.DataFrame({
#                         'State': labels,
#                         'RVI': avg_vi,
#                         'PI': avg_pi,
#                         'Package': avg_pkg
#                     })
#                     print_table_to_file(f, df_avg, "Average Cost Indices:")

#     # ---- AVERAGE COST ONLY (no beta loop) ----
#     for C in C_vals:
#         for K in K_vals:
#             for C_switch in C_switch_vals:
#                 avg_count += 1
#                 print(f"[Average {avg_count}/{avg_total}] "
#                       f"Matrix {p_id+1}, C={C}, K={K}, C_switch={C_switch}")

#                 P0, P1, R0, R1, _ = build_extended_mdp(N_live, p_full, C, K, C_switch)

#                 f.write(f"\n{'='*80}\n")
#                 f.write(f"INSTANCE (Average Cost) – Matrix {p_id+1}, C={C}, K={K}, C_switch={C_switch}\n")
#                 f.write(f"{'='*80}\n")
#                 f.write(f"Parameters:\n")
#                 f.write(f"  N = {N_live}\n")
#                 f.write(f"  Survival probabilities p[0..{N_live}]: {np.round(p_full, 4)}\n")
#                 f.write(f"  C = {C}\n")
#                 f.write(f"  K = {K}\n")
#                 f.write(f"  C_switch = {C_switch}\n")
#                 f.write(f"{'-'*80}\n")

#                 print_matrix_to_file(f, P0, labels, "Transition matrix for action Passive (a=0):")
#                 print_vector_to_file(f, R0, labels, "Reward vector for action Passive (a=0):")
#                 print_matrix_to_file(f, P1, labels, "Transition matrix for action Active (a=1):")
#                 print_vector_to_file(f, R1, labels, "Reward vector for action Active (a=1):")

#                 # Average cost indices
#                 f.write("--- Average-Cost Whittle Indices ---\n")
#                 avg_vi = []
#                 avg_pi = []
#                 for s in range(len(labels)):
#                     try:
#                         avg_vi.append(whittle_index_bisection(R0, R1, P0, P1, s, solver='rvi', ref=0))
#                     except:
#                         avg_vi.append(np.nan)
#                     try:
#                         avg_pi.append(whittle_index_bisection(R0, R1, P0, P1, s, solver='pi', ref=0))
#                     except:
#                         avg_pi.append(np.nan)

#                 avg_pkg = [np.nan] * len(labels)
#                 if PKG_AVAILABLE:
#                     try:
#                         model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#                         avg_pkg = model.whittle_indices()
#                     except:
#                         pass

#                 df_avg = pd.DataFrame({
#                     'State': labels,
#                     'RVI': avg_vi,
#                     'PI': avg_pi,
#                     'Package': avg_pkg
#                 })
#                 print_table_to_file(f, df_avg, "Average Cost Indices:")

# # ----------------------------------------------------------------------
# # Close file and finish
# # ----------------------------------------------------------------------
# f.close()
# print(f"\nAll results written to {filename}")
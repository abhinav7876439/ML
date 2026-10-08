
import numpy as np
from scipy import linalg

# =====================================================================
# MDP builder and solvers (your existing code)
# =====================================================================
def build_extended_mdp(N, p, C, K, C_switch):
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)
    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N+1)]

    for s in range(num_states):
        if s == 0:                # (0,1)
            P1[s, 0] = 1.0
            R1[s] = -C
        else:
            P1[s, 0] = 1.0
            R1[s] = -(C + C_switch)

    for s in range(num_states):
        if s == 0:                # (0,1)
            P0[s, 2] = p[0]
            P0[s, 1] = 1 - p[0]
            R0[s] = -(K * (1 - p[0]))
        elif s == 1:              # (0,0)
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


# # ----------------------------------------------------------------------
# # 2. Discounted solvers (VI, PI)
# # ----------------------------------------------------------------------

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

# =====================================================================
# Closed form Indices using exact derivations
# =====================================================================
def compute_G_H(p, beta):
    """
    G(k) = sum_{l=0}^{k-1} beta^{l+1} (1-p_l) prod_{j=0}^{l-1} p_j
    H(k) = beta^{k+1} prod_{l=0}^{k-1} p_l
    G(k+1) = G(k) + (1-p_k) H(k)
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
    Compute closed-form Whittle indices for All states.
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
            w_active = (num / den) - C
        else:
            w_active = np.inf
    else:
        w_active = np.inf

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
    return w_active, w_passive

# =====================================================================
# Verification utilities
# =====================================================================
def check_index_monotonicity(indices, state_labels, ordering=None):
    S = len(indices)
    if ordering is None:
        ordered_states = list(range(S))
    else:
        ordered_states = ordering
    monotone = True
    for i in range(1, len(ordered_states)):
        s_prev = ordered_states[i-1]
        s_curr = ordered_states[i]
        if indices[s_curr] <= indices[s_prev]:
            monotone = False
            print(f"  Monotonicity broken: {state_labels[s_prev]} ({indices[s_prev]:.4f}) >= {state_labels[s_curr]} ({indices[s_curr]:.4f})")
    if monotone:
        print("  Indices are strictly increasing in the state ordering.")
    return monotone



def verify_threshold_policies(indices, P0, P1, R0, R1, beta, state_labels,
                              solver='pi', tol=1e-6, w_eps=1e-6):
    S = len(indices)
    unique_indices = sorted(set(indices))
    test_Ws = []
    for lam in unique_indices:
        test_Ws.append(lam - w_eps)
        test_Ws.append(lam)
        test_Ws.append(lam + w_eps)
    if unique_indices:
        test_Ws.append(unique_indices[0] - 10.0)
    test_Ws = sorted(set(test_Ws))

    all_ok = True
    for W in test_Ws:
        pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, W, beta)
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - W) + beta * (P1 @ V)
        for s in range(S):
            if pi[s] == 0:
                if Q1[s] > Q0[s] + tol:    # active would be better
                    print(f"  FAIL at W={W:.6f}, state {state_labels[s]}: passive but Q1 > Q0")
                    all_ok = False
                if abs(V[s] - Q0[s]) > tol:
                    print(f"  Value mismatch at W={W:.6f}, state {state_labels[s]}: V != Q0")
                    all_ok = False
            else:
                if Q0[s] > Q1[s] + tol:    # passive would be better
                    print(f"  FAIL at W={W:.6f}, state {state_labels[s]}: active but Q0 > Q1")
                    all_ok = False
                if abs(V[s] - Q1[s]) > tol:
                    print(f"  Value mismatch at W={W:.6f}, state {state_labels[s]}: V != Q1")
                    all_ok = False
    if all_ok:
        print(f"  All {len(test_Ws)} test W values passed optimality check. Threshold policy is optimal.")
    return all_ok

# def verify_threshold_policies(indices, P0, P1, R0, R1, beta, state_labels,
#                               solver='pi', tol=1e-6, w_eps=1e-6):
#     S = len(indices)
#     unique_indices = sorted(set(indices))
#     test_Ws = []
#     for lam in unique_indices:
#         test_Ws.append(lam - w_eps)
#         test_Ws.append(lam)
#         test_Ws.append(lam + w_eps)
#     if unique_indices:
#         test_Ws.append(unique_indices[0] - 10.0)
#     test_Ws = sorted(set(test_Ws))

#     all_ok = True
#     for W in test_Ws:
#         pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
#         V = policy_evaluation_discounted(pi, P0, P1, R0, R1, W, beta)
#         Q0 = R0 + beta * (P0 @ V)
#         Q1 = (R1 - W) + beta * (P1 @ V)
#         for s in range(S):
#             if pi[s] == 0:
#                 if Q0[s] > Q1[s] + tol:
#                     print(f"  FAIL at W={W:.6f}, state {state_labels[s]}: passive but Q0({Q0[s]:.6f}) > Q1({Q1[s]:.6f})")
#                     all_ok = False
#                 if abs(V[s] - Q0[s]) > tol:
#                     print(f"  Value mismatch at W={W:.6f}, state {state_labels[s]}: V={V[s]:.6f} != Q0={Q0[s]:.6f}")
#                     all_ok = False
#             else:
#                 if Q1[s] > Q0[s] + tol:
#                     print(f"  FAIL at W={W:.6f}, state {state_labels[s]}: active but Q1({Q1[s]:.6f}) > Q0({Q0[s]:.6f})")
#                     all_ok = False
#                 if abs(V[s] - Q1[s]) > tol:
#                     print(f"  Value mismatch at W={W:.6f}, state {state_labels[s]}: V={V[s]:.6f} != Q1={Q1[s]:.6f}")
#                     all_ok = False
#     if all_ok:
#         print(f"  All {len(test_Ws)} test W values passed optimality check. Threshold policy is optimal.")
#     return all_ok

# =====================================================================
# Main demonstration
# =====================================================================
if __name__ == "__main__":
    N = 5
    np.random.seed(42)
    p_raw = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probs
    p_full = np.append(p_raw, 0.0)                        # terminal zero for MDP builder
    C, K = 5.0, 500.0
    C_switch = 1.0
    beta = 0.9999

    print("=" * 70)
    print(f"Survival probabilities (non-terminal): {np.round(p_raw, 3)}")
    print("=" * 70)

    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)

    # Compute closed-form indices using your exact formulas
    w01, w_passive_arr = closed_form_indices(p_raw, beta, K, C, C_switch)
    closed_indices = np.zeros(len(R0_ext))
    closed_indices[0] = w01               # (0,1)
    closed_indices[1] = w_passive_arr[0] # (0,0)
    for k in range(1, N+1):
        closed_indices[1 + k] = w_passive_arr[k]   # (k,0)

    print("\n--- Closed-form indices (derived formulas) ---")
    for s, lab in enumerate(state_labels_ext):
        print(f"State {lab:8s}: λ = {closed_indices[s]:8.4f}")

    # Numeric indices via bisection (for comparison)
    print("\n--- Numeric indices (Policy Iteration bisection) ---")
    numeric_indices = []
    for s in range(len(R0_ext)):
        lam = whittle_index_discounted(state=s, solver='pi', beta=beta,
                                       w_min=-100.0, w_max=1000.0,
                                       tol_w=1e-6, tol_mdp=1e-8,
                                       P0=P0_ext, P1=P1_ext, R0=R0_ext, R1=R1_ext)
        numeric_indices.append(lam)
        print(f"State {state_labels_ext[s]:8s}: λ = {lam:8.4f}")

    diff = np.array(closed_indices) - np.array(numeric_indices)
    print("\n--- Difference (closed - numeric) ---")
    for s, lab in enumerate(state_labels_ext):
        print(f"State {lab:8s}: diff = {diff[s]:.2e}")
    if np.max(np.abs(diff)) < 1e-5:
        print("Closed-form and numeric indices match within tolerance.")
    else:
        print("WARNING: Discrepancy between closed-form and numeric indices!")

    # Monotonicity check
    print("\n=== Monotonicity of closed-form indices ===")
    expected_order = [1, 0] + list(range(2, len(closed_indices)))  # (0,0) < (0,1) < (1,0) < ...
    print(f"Expected Order : {[state_labels_ext[i] for i in expected_order]}")
    check_index_monotonicity(closed_indices, state_labels_ext, ordering=expected_order)

    # Indexability verification
    print("\n=== Verification of threshold policies (indexability) ===")
    result = verify_threshold_policies(closed_indices, P0_ext, P1_ext, R0_ext, R1_ext,
                                       beta, state_labels_ext, solver='pi', tol=1e-5)
    if result:
        print("Indexability verified: threshold policies based on closed-form indices are optimal for all W.")
    else:
        print("Indexability verification FAILED.")

    # Indifference at each state's index
    print("\n--- Indifference check at each state's closed-form index ---")
    for s in range(len(closed_indices)):
        W = closed_indices[s]
        V, Q0, Q1 = solve_pi_discounted(P0_ext, P1_ext, R0_ext, R1_ext, W, beta, tol=1e-8)
        diff_val = abs(Q0[s] - Q1[s])
        if diff_val < 1e-5:
            print(f"  State {state_labels_ext[s]}: indifference holds (|Q0-Q1| = {diff_val:.2e})")
        else:
            print(f"  State {state_labels_ext[s]}: indifference VIOLATED (|Q0-Q1| = {diff_val:.6f})")

    print("\nDone.")
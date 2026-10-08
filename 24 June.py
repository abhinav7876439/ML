# import numpy as np

# # ----------------------------------------------------------------------
# # MDP definition (states 0..3 represent original states 1..4)
# # ----------------------------------------------------------------------
# P0 = np.array([[0.5, 0.0, 0.0, 0.5],
#                [0.5, 0.5, 0.0, 0.0],
#                [0.0, 0.5, 0.5, 0.0],
#                [0.0, 0.0, 0.5, 0.5]])

# P1 = np.array([[0.5, 0.5, 0.0, 0.0],
#                [0.0, 0.5, 0.5, 0.0],
#                [0.0, 0.0, 0.5, 0.5],
#                [0.5, 0.0, 0.0, 0.5]])

# R = np.array([-1.0, 0.0, 0.0, 1.0])   # state‑dependent reward

# num_states = len(R)

# # ----------------------------------------------------------------------
# # Solver 1: Relative Value Iteration (average reward)
# # ----------------------------------------------------------------------
# def solve_rvi(w, ref=0, tol=1e-12, max_iter=10000):
#     """Return differential values h (h[ref]=0) and gain g for subsidy w."""
#     V = np.zeros(num_states)
#     for _ in range(max_iter):
#         Q0 = R + P0 @ V
#         Q1 = R - w + P1 @ V
#         V_tilde = np.maximum(Q0, Q1)
#         delta = V_tilde[ref]
#         V_new = V_tilde - delta
#         if np.max(np.abs(V_new - V)) < tol:
#             V = V_new
#             break
#         V = V_new
#     # final Q-values
#     Q0 = R + P0 @ V
#     Q1 = R - w + P1 @ V
#     # gain = delta (the subtracted value at convergence)
#     g = delta
#     return V, g, Q0, Q1

# # ----------------------------------------------------------------------
# # Solver 2: Policy Iteration (average reward)
# # ----------------------------------------------------------------------
# def policy_evaluation(pi, w, ref=0):
#     """Solve (I - P_pi) h + g * 1 = r_pi with h[ref]=0."""
#     P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(num_states)])
#     r_pi = np.array([R[s] if pi[s] == 0 else R[s] - w for s in range(num_states)])
#     S = num_states
#     # Build matrix: [ I-P_pi   ones ] [h; g] = r_pi  and  h[ref]=0
#     A = np.zeros((S+1, S+1))
#     A[:S, :S] = np.eye(S) - P_pi
#     A[:S, -1] = 1.0           # gain column
#     A[S, ref] = 1.0           # constraint h[ref]=0
#     b = np.zeros(S+1)
#     b[:S] = r_pi
#     # b[S] = 0 already
#     sol = np.linalg.solve(A, b)
#     h = sol[:S]
#     g = sol[-1]
#     return h, g

# def solve_pi(w, ref=0, tol=1e-12, max_iter=100):
#     """Policy iteration for subsidy w."""
#     pi = np.zeros(num_states, dtype=int)  # start passive
#     for _ in range(max_iter):
#         h, g = policy_evaluation(pi, w, ref)
#         Q0 = R + P0 @ h
#         Q1 = R - w + P1 @ h
#         pi_new = (Q1 > Q0).astype(int)  # active if Q1 strictly larger
#         if np.all(pi_new == pi):
#             break
#         pi = pi_new
#     # final Q-values
#     Q0 = R + P0 @ h
#     Q1 = R - w + P1 @ h
#     return h, g, Q0, Q1

# # ----------------------------------------------------------------------
# # Bisection for Whittle index of a state s
# # ----------------------------------------------------------------------
# def whittle_index_bisection(s, solver='rvi', w_min=-5.0, w_max=5.0,
#                             tol_w=1e-10, tol_mdp=1e-12, ref=0, verbose=False):
#     """Find w such that Q0[s] - Q1[s] = 0 using binary search."""
#     solve_func = solve_rvi if solver == 'rvi' else solve_pi
#     # function f(w) = Q0[s] - Q1[s]
#     def f(w):
#         _, _, Q0, Q1 = solve_func(w, ref=ref, tol=tol_mdp)
#         return Q0[s] - Q1[s]

#     # ensure bracket contains root
#     f_min = f(w_min)
#     f_max = f(w_max)
#     if f_min * f_max > 0:
#         # expand bracket aggressively
#         expand = 2.0
#         while f_min * f_max > 0:
#             if abs(f_min) < abs(f_max):
#                 w_min -= expand
#                 f_min = f(w_min)
#             else:
#                 w_max += expand
#                 f_max = f(w_max)
#             if abs(w_min) > 100 or abs(w_max) > 100:
#                 raise RuntimeError(f"Could not bracket index for state {s}")
#         if verbose:
#             print(f"Bracket expanded to [{w_min}, {w_max}]")
#     # bisection
#     for _ in range(60):   # plenty for double precision
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

# # ----------------------------------------------------------------------
# # Compute indices with both methods and compare
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     print("Computing Whittle indices for states 1..4 (0-indexed as 0..3)")
#     print("Expected: λ(1)=-0.5, λ(2)=0.5, λ(3)=1.0, λ(4)=-1.0\n")

#     for method in ['rvi', 'pi']:
#         print(f"--- Using {method.upper()} ---")
#         indices = []
#         for s in range(num_states):
#             lam = whittle_index_bisection(s, solver=method, w_min=-2.0, w_max=2.0,
#                                          tol_w=1e-10, tol_mdp=1e-12, ref=0)
#             indices.append(lam)
#             print(f"State {s+1} (orig {s+1}): λ = {lam:.10f}")
#         print("Indices:", indices)
#         # check against exact
#         exact = [-0.5, 0.5, 1.0, -1.0]
#         diff = np.abs(np.array(indices) - exact)
#         print("Max error from exact:", np.max(diff))
#         print()



import numpy as np
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")

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

print("P0_circ:\n", P0_circ)
print("P1_circ:\n", P1_circ)    
print("R0_circ:", R0_circ)
print("R1_circ:", R1_circ)

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

print("P0_rest:\n", P0_rest)

P1_rest = np.ones((5, 5)) * 0
P1_rest[:, 0] = 1.0   # restart to state 1 (index 0)

print("P1_rest:\n", P1_rest)

R0_rest = np.array([a**(i+1) for i in range(5)])   # passive rewards
R1_rest = np.zeros(5)                              # active rewards are zero

print("R0_rest:", R0_rest)
print("R1_rest:", R1_rest)


# ----------------------------------------------------------------------
# Problem 3 : Non-Indexable problem (3 states)
# ----------------------------------------------------------------------


P0_non_indexable = np.array([[0.005, 0.793, 0.202],
                    [0.027, 0.558, 0.415],
                    [0.736, 0.269, 0.015]])

P1_non_indexable = np.array([[0.718, 0.254, 0.028],
                             [0.347, 0.097, 0.556],
                    [0.015, 0.956, 0.029]])

R0_non_indexable = np.array([0.0, 0.0, 0.0])          # or np.zeros(3)
R1_non_indexable = np.array([0.699, 0.362, 0.715])


# ----------------------------------------------------------------------
# Problem 4 : Non-Indexable problem (4 states)
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



# Expected values
expected_circ = [-0.5, 0.5, 1.0, -1.0]
expected_rest = [-0.9, -0.73, -0.5, -0.26, -0.01]

compute_indices(R0_circ, R1_circ, P0_circ, P1_circ,
                expected_circ, "Circulant dynamics")
compute_indices(R0_rest, R1_rest, P0_rest, P1_rest,
                expected_rest, "Restart problem")
compute_indices(R0_non_indexable, R1_non_indexable, P0_non_indexable, P1_non_indexable, [0.0, 0.0, 0.0], "Non-indexable problem (4 states, no expected values)")
compute_indices(R0_non, R1_non, P0_non, P1_non, [0.0, 0.0, 0.0, 0.0], "Non-indexable problem (4 states, no expected values)")

# Compare with markovianbandit package if available
if PKG_AVAILABLE:
    # Create restless bandit models
    model_circ = bandit.restless_bandit_from_P0P1_R0R1(P0_circ, P1_circ, R0_circ, R1_circ)
    model_rest = bandit.restless_bandit_from_P0P1_R0R1(P0_rest, P1_rest, R0_rest, R1_rest)
    model_non_indexable = bandit.restless_bandit_from_P0P1_R0R1(P0_non_indexable, P1_non_indexable, R0_non_indexable, R1_non_indexable)
    model_non = bandit.restless_bandit_from_P0P1_R0R1(P0_non, P1_non, R0_non, R1_non)

    # Compute Whittle indices (one per original state)
    # The method may use a default discount factor; adjust if needed.
    indices_circ = model_circ.whittle_indices()
    indices_rest = model_rest.whittle_indices()
    indices_non_indexable = model_non_indexable.whittle_indices()
    indices_non = model_non.whittle_indices()

    print("\nPackage Whittle indices (discount factor default):")
    print("Circulant problem:")
    for s, idx in enumerate(indices_circ):
        print(f"  State {s+1}: λ = {idx:.6f}")
    print("Restart problem:")
    for s, idx in enumerate(indices_rest):
        print(f"  State {s+1}: λ = {idx:.6f}")
    print("Non-indexable problem(3 States):")
    for s, idx in enumerate(indices_non_indexable):
        print(f"  State {s+1}: λ = {idx:.6f}")
    print("Non-indexable problem(4 States):")
    for s, idx in enumerate(indices_non):
        print(f"  State {s+1}: λ = {idx:.6f}")

    

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import solve

# ------------------------------------------------------------
# 1. Build MDP (same as before)
# ------------------------------------------------------------
def build_mdp(N, p, C, K, c_switch=0):
    n_states = (N + 1) * 2
    P0 = np.zeros((n_states, n_states))
    P1 = np.zeros((n_states, n_states))
    R0 = np.zeros(n_states)
    R1 = np.zeros(n_states)

    for x in range(N + 1):
        idx_p = x
        idx_a = x + (N + 1)
        # Passive
        P0[idx_p, 0] += (1 - p[x])
        P0[idx_a, 0] += (1 - p[x])
        if x < N:
            P0[idx_p, x + 1] += p[x]
            P0[idx_a, x + 1] += p[x]
        else:
            P0[idx_p, 0] += p[x]
            P0[idx_a, 0] += p[x]
        R0[idx_p] = -K * (1 - p[x])
        R0[idx_a] = -K * (1 - p[x])
        # Active
        P1[idx_p, N + 1] = 1.0
        P1[idx_a, N + 1] = 1.0
        R1[idx_p] = -(C + c_switch)
        R1[idx_a] = -C
    return P0, P1, R0, R1

# ------------------------------------------------------------
# 2. Discounted Policy Iteration for a given w
# ------------------------------------------------------------
def discounted_policy_iteration(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=100):
    """
    Policy iteration for discounted MDP with subsidy w.
    Returns optimal value function V and optimal policy (as boolean array: True=active).
    """
    n = len(R0)
    R1_mod = R1 - w
    # Initial policy: all passive (action 0)
    policy = np.zeros(n, dtype=bool)   # False = passive, True = active

    for _ in range(max_iter):
        # --- Policy evaluation ---
        # Build R_pi and P_pi
        R_pi = np.zeros(n)
        P_pi = np.zeros((n, n))
        for i in range(n):
            if policy[i]:   # active
                R_pi[i] = R1_mod[i]
                P_pi[i, :] = P1[i, :]
            else:           # passive
                R_pi[i] = R0[i]
                P_pi[i, :] = P0[i, :]
        # Solve (I - beta * P_pi) V = R_pi
        A = np.eye(n) - beta * P_pi
        V = solve(A, R_pi)

        # --- Policy improvement ---
        Q0 = R0 + beta * (P0 @ V)
        Q1 = R1_mod + beta * (P1 @ V)
        new_policy = (Q1 < Q0)   # active if Q1 < Q0

        if np.all(policy == new_policy):
            break
        policy = new_policy

    return V, policy

# ------------------------------------------------------------
# 3. Whittle index using policy iteration inside bisection
# ------------------------------------------------------------
def whittle_index_policy(state_idx, P0, P1, R0, R1, beta,
                         w_init=0.0, w_span=100.0, tol=1e-6):
    """
    Compute discounted Whittle index using policy iteration for each w.
    """
    def advantage(w):
        V, _ = discounted_policy_iteration(P0, P1, R0, R1, w, beta)
        Q_pass = R0[state_idx] + beta * (P0[state_idx, :] @ V)
        Q_act = (R1[state_idx] - w) + beta * (P1[state_idx, :] @ V)
        return Q_pass - Q_act

    # Dynamic bracket expansion (same as before)
    w_low, w_high = w_init - w_span, w_init + w_span
    f_low = advantage(w_low)
    f_high = advantage(w_high)
    for _ in range(20):
        if f_low * f_high <= 0:
            break
        if abs(f_low) < abs(f_high):
            w_low -= (w_high - w_low)
            f_low = advantage(w_low)
        else:
            w_high += (w_high - w_low)
            f_high = advantage(w_high)
    else:
        return w_low if abs(f_low) < abs(f_high) else w_high

    try:
        return brentq(advantage, w_low, w_high, xtol=tol)
    except ValueError:
        return w_low if abs(f_low) < abs(f_high) else w_high

# ------------------------------------------------------------
# 4. Value iteration (synchronous) for comparison
# ------------------------------------------------------------
def discounted_sync_vi(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=20000):
    n = len(R0)
    R1_mod = R1 - w
    V = np.zeros(n)
    for _ in range(max_iter):
        V_old = V
        Q0 = R0 + beta * (P0 @ V_old)
        Q1 = R1_mod + beta * (P1 @ V_old)
        V = np.minimum(Q0, Q1)
        if np.max(np.abs(V - V_old)) < tol:
            break
    return V

def whittle_index_vi(state_idx, P0, P1, R0, R1, beta,
                     w_init=0.0, w_span=100.0, tol=1e-6):
    def advantage(w):
        V = discounted_sync_vi(P0, P1, R0, R1, w, beta)
        Q_pass = R0[state_idx] + beta * (P0[state_idx, :] @ V)
        Q_act = (R1[state_idx] - w) + beta * (P1[state_idx, :] @ V)
        return Q_pass - Q_act

    w_low, w_high = w_init - w_span, w_init + w_span
    f_low = advantage(w_low)
    f_high = advantage(w_high)
    for _ in range(20):
        if f_low * f_high <= 0:
            break
        if abs(f_low) < abs(f_high):
            w_low -= (w_high - w_low)
            f_low = advantage(w_low)
        else:
            w_high += (w_high - w_low)
            f_high = advantage(w_high)
    else:
        return w_low if abs(f_low) < abs(f_high) else w_high

    try:
        return brentq(advantage, w_low, w_high, xtol=tol)
    except ValueError:
        return w_low if abs(f_low) < abs(f_high) else w_high

# ------------------------------------------------------------
# 5. Closed‑form expressions (same as before)
# ------------------------------------------------------------
def closed_form_discounted(p, C, K, beta):
    N = len(p)
    G = np.zeros(N + 1)
    for k in range(N):
        prod = 1.0
        for i in range(k):
            prod *= p[i]
        Hk = beta**(k+1) * prod
        G[k+1] = G[k] + (1 - p[k]) * Hk
    W = np.zeros(N)
    for k in range(N):
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        W[k] = K * num / den - C
    return W

# ------------------------------------------------------------
# 6. Main: compare value iteration vs policy iteration
# ------------------------------------------------------------
def main():
    np.random.seed(42)
    N = 7
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]
    p = np.append(p, 0.0)
    print("Survival probabilities p_x:", np.round(p, 3))

    C, K, c_switch = 5.0, 500.0, 0.0
    beta = 0.95

    P0, P1, R0, R1 = build_mdp(N, p, C, K, c_switch)

    # Closed‑form
    closed = closed_form_discounted(p, C, K, beta)

    # Value iteration indices
    print("\nComputing discounted indices using VALUE ITERATION...")
    ind_vi = np.zeros(2*(N+1))
    for s in range(2*(N+1)):
        ind_vi[s] = whittle_index_vi(s, P0, P1, R0, R1, beta,
                                     w_init=0.0, w_span=200.0, tol=1e-6)
    active_vi = ind_vi[N+1:2*(N+1)]

    # Policy iteration indices
    print("\nComputing discounted indices using POLICY ITERATION...")
    ind_pi = np.zeros(2*(N+1))
    for s in range(2*(N+1)):
        ind_pi[s] = whittle_index_policy(s, P0, P1, R0, R1, beta,
                                         w_init=0.0, w_span=200.0, tol=1e-6)
    active_pi = ind_pi[N+1:2*(N+1)]

    # Comparison table
    print("\n" + "="*90)
    print(f"Discounted Whittle indices (beta = {beta})")
    print("-"*90)
    print(f"{'x':>3} | {'Closed-form':>12} | {'Value Iter':>12} | {'Policy Iter':>12} | {'Diff(VI-CF)':>12} | {'Diff(PI-CF)':>12}")
    print("-"*90)
    for x in range(N+1):
        diff_vi = active_vi[x] - closed[x]
        diff_pi = active_pi[x] - closed[x]
        print(f"{x:3} | {closed[x]:12.4f} | {active_vi[x]:12.4f} | {active_pi[x]:12.4f} | {diff_vi:12.4f} | {diff_pi:12.4f}")

    # Plot comparison
    plt.figure(figsize=(8,5))
    plt.plot(range(N+1), closed, 'o-', label='Closed-form')
    plt.plot(range(N+1), active_vi, 's--', label='Value Iteration')
    plt.plot(range(N+1), active_pi, '^:', label='Policy Iteration')
    plt.xlabel('Deterioration state x')
    plt.ylabel('Whittle index W(x,1)')
    plt.title('Comparison: Value Iteration vs Policy Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
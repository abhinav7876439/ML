import numpy as np
from scipy.optimize import brentq


def discounted_value_iter(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=20000):
    """
    Solve the discounted Bellman equation for the w-subsidy problem.
    Returns the value function V, the optimal action, and the number of iterations until convergence.
    """
    n = len(R0)
    R1_mod = R1 - w
    V = np.zeros(n)
    converged_iter = max_iter
    for it in range(max_iter):
        V_old = V.copy()
        Q0 = R0 + beta * (P0 @ V_old)
        Q1 = R1_mod + beta * (P1 @ V_old)
        V = np.minimum(Q0, Q1)          # no normalisation!
        if np.max(np.abs(V - V_old)) < tol:
            converged_iter = it + 1
            break

    Q0 = R0 + beta * (P0 @ V)
    Q1 = R1_mod + beta * (P1 @ V)
    opt_action = (Q1 < Q0).astype(int)
    return V, opt_action, converged_iter


def discounted_value_iteration(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=20000):
    n = len(R0)
    R1_mod = R1 - w
    V = np.zeros(n)
    for it in range(max_iter):
        V_old = V
        Q0 = R0 + beta * (P0 @ V_old)
        Q1 = R1_mod + beta * (P1 @ V_old)
        V = np.minimum(Q0, Q1)
        if np.linalg.norm(V - V_old, ord=np.inf) < tol:
            break
    return V   


def discounted_whittle_bisection(state_idx, P0, P1, R0, R1, beta, w_min, w_max, tol=1e-6):
    def advantage(w):
        V, _, iters = discounted_value_iteration(P0, P1, R0, R1, w, beta)
        print(f"State {state_idx}, w={w:.4f}, converged in {iters} iterations")
        Q_pass = R0[state_idx] + beta * (P0[state_idx, :] @ V)
        Q_act = (R1[state_idx] - w) + beta * (P1[state_idx, :] @ V)
        return Q_pass - Q_act

    # Expand bounds if needed
    w_low, w_high = w_min, w_max
    f_low = advantage(w_low)
    f_high = advantage(w_high)
    while f_low * f_high > 0 and (w_high - w_low) < 1e6:
        if f_low > 0:
            w_low -= (w_high - w_low)
        else:
            w_high += (w_high - w_low)
        f_low = advantage(w_low)
        f_high = advantage(w_high)

    try:
        return brentq(advantage, w_low, w_high, xtol=tol)
    except ValueError:
        # fallback: choose the bound where advantage is closer to zero
        return w_low if abs(f_low) < abs(f_high) else w_high


def compute_all_indices_discounted(N, p, C, K, c_switch, beta, w_range=(-100, 500)):
    P0, P1, R0, R1 = build_mdp(N, p, C, K, c_switch)
    indices = np.zeros(2*(N+1))
    for s in range(2*(N+1)):
        indices[s] = discounted_whittle_bisection(s, P0, P1, R0, R1, beta, w_range[0], w_range[1])
    return indices
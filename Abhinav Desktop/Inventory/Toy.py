import numpy as np

# -----------------------------------
# MDP specification
# -----------------------------------
states = [0, 1]
actions = [0, 1]

c = {
    (0,0): 1.0,
    (0,1): 1.5,
    (1,0): 0.5,
    (1,1): 0.8
}

P = {
    (0,0): np.array([0.5, 0.5]),
    (0,1): np.array([0.2, 0.8]),
    (1,0): np.array([0.7, 0.3]),
    (1,1): np.array([0.4, 0.6])
}


# ===============================================================
# (A) FINITE-HORIZON RELATIVE DP RECURSION
# ===============================================================
def finite_horizon_relative(T, ref_state=0):
    """
    Computes finite-horizon DP with relative normalization:
    V_tilde_{t}(s) = V_t(s) - g_t * t
    g_t = average incremental gain
    """
    nS = len(states)
    nA = len(actions)

    V_next = np.zeros(nS)   # V_{t-1}, initialized at horizon 0 = 0
    results = []

    for t in range(1, T+1):
        Q = np.zeros((nS, nA))

        # Compute Q_t(s,a)
        for s in states:
            for a in actions:
                Q[s, a] = c[(s,a)] + np.sum(P[(s,a)] * V_next)

        # Compute V_t(s)
        V_t = np.max(Q, axis=1)

        # Compute gain g_t = V_t(ref_state) - V_{t-1}(ref)
        g_t = V_t[ref_state] - V_next[ref_state]

        # Create relative value function
        V_tilde = V_t - t*g_t

        results.append((t, g_t, V_t.copy(), V_tilde.copy()))

        # Prepare for next iteration
        V_next = V_t.copy()

    return results


# ===============================================================
# (B) RELATIVE VALUE ITERATION (RVI) FOR AVERAGE COST
# ===============================================================
def RVI(ref_state=0, tol=1e-10, max_iter=50000):
    V = np.zeros(len(states))
    rho = 0.0

    for it in range(max_iter):
        V_old = V.copy()
        Q = np.zeros((2,2))

        # Compute Q(s,a)
        for s in states:
            for a in actions:
                Q[s,a] = c[(s,a)] + np.sum(P[(s,a)] * V_old)

        # Update V_new(s) = max_a Q(s,a)
        V_new = np.max(Q, axis=1)

        # Gain update (shift)
        shift = V_new[ref_state] - V_old[ref_state]
        rho += shift

        # Relative normalization
        V_new = V_new - V_new[ref_state]

        V = V_new

        if np.max(np.abs(V - V_old)) < tol:
            break

    # Compute greedy policy
    policy = {}
    for s in states:
        Q_s = [c[(s,a)] + np.sum(P[(s,a)] * V) for a in actions]
        policy[s] = int(np.argmin(Q_s))  # minimizing cost

    return rho, V, policy
    


# ===============================================================
# RUN BOTH ALGORITHMS
# ===============================================================

# ---- Finite horizon ----
T = 20
results = finite_horizon_relative(T)

print("FINITE-HORIZON RELATIVE RECURSION RESULTS")
print("t   g_t         V_t               V_tilde")
for (t, g_t, Vt, Vtil) in results:
    print(f"{t:2d}  {g_t:.6f}   {Vt}   {Vtil}")

# ---- RVI ----
rho, V, policy = RVI()
print("\nRVI RESULTS")
print("Average cost rho =", rho)
print("Relative value V =", V)
print("Optimal policy =", policy)

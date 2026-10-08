import numpy as np

# -----------------------------------------------------------
# Toy MDP definition
# -----------------------------------------------------------

states = [0, 1]
actions = [0, 1]

# Costs: c[s][a]
c = {
    0: {0: 1, 1: 3},
    1: {0: 4, 1: 2}
}

# Transition probabilities: deterministic
# P[s][a] = next state
P = {
    0: {0: 0, 1: 1},
    1: {0: 1, 1: 0}
}

# Reference state for relative value
ref_state = 0


# -----------------------------------------------------------
# Finite-horizon relative recursion
# -----------------------------------------------------------
def finite_horizon_relative(T):
    """
    Implements the relative-value dynamic programming:
        g_t = V_t(ref) - V_{t-1}(ref)
        tilde_V_t = V_t - g_t   (with V_t(ref)=0 normalization)
    """

    # Store the raw finite horizon V_t (before normalizing)
    Vt = np.zeros((T+1, 2))
    
    # Initialize terminal condition: V_0 = 0
    Vt[0, :] = 0.0

    # Store incremental gains g_t
    g_vals = np.zeros(T+1)

    # Store tilde V_t (relative values)
    tilde_V = np.zeros((T+1, 2))

    # By definition tilde_V_0 = 0
    tilde_V[0, :] = 0.0

    for t in range(1, T+1):
        V_prev = Vt[t-1].copy()
        V_new = np.zeros(2)

        # Bellman update (finite horizon)
        for s in states:
            Qa = []
            for a in actions:
                s_next = P[s][a]
                Qa.append(c[s][a] + V_prev[s_next])
            V_new[s] = min(Qa)

        # Compute incremental gain:
        # g_t = V_t(ref) - V_{t-1}(ref)
        g_vals[t] = V_new[ref_state] - V_prev[ref_state]

        # Compute relative values:
        # tilde_V_t(s) = V_t(s) - V_t(ref)
        tilde_V[t, :] = V_new - V_new[ref_state]

        # Store raw V_t
        Vt[t, :] = V_new

    return Vt, tilde_V, g_vals


# -----------------------------------------------------------
# Run finite-horizon relative recursion
# -----------------------------------------------------------
T = 20
Vt, tilde_V, g_vals = finite_horizon_relative(T)

print("Raw V_t values (finite horizon):\n", Vt)
print("\nRelative tilde V_t values:\n", tilde_V)
print("\nIncremental gains g_t:\n", g_vals)





def RVI(max_iter=2000, tol=1e-10):
    """
    Implements the average-cost Relative Value Iteration:

        V_{new}(s) = min_a [ c(s,a) + sum P * V_old ]
        shift = V_new(ref_state)
        V_new := V_new - shift
        rho := shift   (est of average cost)

    Returns:
        V   = relative value function
        rho = average cost
        it  = iterations
    """

    V = np.zeros(2)     # initial guess
    rho = 0.0

    for it in range(max_iter):
        V_old = V.copy()
        V_new = np.zeros(2)

        # Bellman update
        for s in states:
            Qa = []
            for a in actions:
                s_next = P[s][a]
                Qa.append(c[s][a] + V_old[s_next])
            V_new[s] = min(Qa)

        # RVI normalization: subtract value at reference state
        shift = V_new[ref_state]
        V_new = V_new - shift

        rho = shift     # estimate of g (average cost)

        # convergence check
        if np.max(np.abs(V_new - V_old)) < tol:
            break

        V = V_new

    return V, rho, it+1


# -----------------------------------------------------------
# Run RVI and print results
# -----------------------------------------------------------
V_star, rho_star, iters = RVI()

print("RVI relative value function V(s):")
print(" V(0) = {:.6f}".format(V_star[0]))
print(" V(1) = {:.6f}".format(V_star[1]))

print("\nEstimated average cost rho =", rho_star)
print("Converged in", iters, "iterations")

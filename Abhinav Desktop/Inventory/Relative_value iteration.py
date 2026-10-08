import numpy as np

states = [0, 1]  # Example state space
actions = [0, 1]      # Example action space: 0 - passive, 1 - active





P = {(0, 0): {0: 0.5, 1: 0.5},
    (0, 1): {0: 0.2, 1: 0.8},
    (1, 0): {0: 0.7, 1: 0.3},
    (1, 1): {0: 0.4, 1: 0.6},
}

C = {(0,0): 1, (0,1): 1.5, (1,0): 0.5, (1,1): 0.8}  # Costs for each state-action pair


def relative_value_iteration(P, C, w, ref_idx,
                             tol=1e-8, max_iter=10000, V_init=None, verbose=False):

    nS = len(states)
    state_to_idx = {s:i for i,s in enumerate(states)}

    if V_init is None:
        V = np.zeros(nS)
    else:
        V = np.array(V_init, dtype=float)

    # Apply subsidy to passive action a=0
    C_sub = C.copy()
    for s in states:
        C_sub[(s,0)] += w

    for it in range(1, max_iter+1):

        V_old = V.copy()
        Q = np.zeros((nS, len(actions)))

        for s in states:
            si = state_to_idx[s]
            for a in actions:
                Q[si,a] = C_sub[(s,a)] + sum(
                    prob * V_old[state_to_idx[sj]]
                    for sj, prob in P[(s,a)].items()
                )

        V_tilde = np.min(Q, axis=1)
        rho = V_tilde[ref_idx]
        V = V_tilde - rho

        if np.max(np.abs(V - V_old)) < tol:
            break

    policy = np.argmin(Q, axis=1)
    return V, policy, Q, rho, it




ref_idx = 0
verbose = True


w = 0
# Run RVI with subsidy w
V, policy, Q, rho, iterations = relative_value_iteration(
        P=P,
        C=C,
        w=w,
        ref_idx=ref_idx,
        verbose=verbose
    )

print("Optimal Relative Value Function:", V)
print("Optimal Policy:", policy)    
print("Optimal Q-Function:", Q)
print("Optimal Average Cost (rho):", rho)
print("Iterations:", iterations)

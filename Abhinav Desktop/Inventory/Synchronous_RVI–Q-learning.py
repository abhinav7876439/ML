import numpy as np
import random
from collections import defaultdict

# -------------------------
# Toy MDP (user's MDP)
# -------------------------
states = [0, 1]
actions = [0, 1]
nS = len(states)
nA = len(actions)

# Transition probabilities P[(s,a)] = np.array([P(s'=0), P(s'=1)])
P = {
    (0, 0): np.array([0.5, 0.5]),
    (0, 1): np.array([0.2, 0.8]),
    (1, 0): np.array([0.7, 0.3]),
    (1, 1): np.array([0.4, 0.6]),
}

# Immediate costs c[(s,a)]
c = {
    (0, 0): 1.0,
    (0, 1): 1.5,
    (1, 0): 0.5,
    (1, 1): 0.8,
}

# -------------------------
# Utility: sample next state index given distribution (array)
# -------------------------
def sample_next_state(pdist):
    return np.searchsorted(np.cumsum(pdist), random.random())

# -------------------------
# Offset function f(Q)
# examples:
#  - f(Q) = Q[i0,a0]
#  - f(Q) = min_b Q[i0,b]   <- we use this
#  - f(Q) = np.mean(Q)
# Must satisfy translation-invariance f(Q + c*1) = f(Q) + c
# -------------------------
def f_offset_min_at_state(Q, i0):
    # returns min_b Q[i0, b]
    return float(np.min(Q[i0, :]))


# -------------------------
# Synchronous RVI-Q-learning
# -------------------------
def rvi_q_learning_synchronous(
    P, c,
    states, actions,
    i0=0,               # reference state for f
    f_fn=None,          # offset function f(Q)
    alpha=1.0,          # gamma(n) = alpha/(n+1)
    max_iter=20000,
    seed=0,
    verbose_interval=2000
):
    """
    Synchronous RVI-Q-learning.

    Algorithm (synchronous):
      For n = 0,1,2,...
        For every (s,a):
          sample s' ~ P[(s,a)]
          sample reward g = c[(s,a)]
          target = g + min_b Q[s',b] - f(Q) - Q[s,a]
          Q[s,a] <- Q[s,a] + gamma(n) * target

    Returns:
      Q: final Q-table (nS x nA)
      rho_trace: list of f(Q) estimates over iterations
      policy_trace: greedy policy from Q periodically (optional)
    """
    random.seed(seed)
    np.random.seed(seed)

    nS = len(states)
    nA = len(actions)
    Q = np.zeros((nS, nA))  # initialize Q arbitrarily, zeros ok

    if f_fn is None:
        # default: min over actions at reference state i0
        f_fn = lambda Qmat: f_offset_min_at_state(Qmat, i0)

    rho_trace = []
    policy_trace = []
    # For reproducibility, we'll draw a single sample s' for each (s,a) per iteration.
    # This is synchronous: all components see their own sampled next-state each iter.

    for n in range(max_iter):
        gamma_n = alpha / (n + 1.0)  # stepsize schedule (1/(n+1) when alpha=1)
        fQ = f_fn(Q)                 # offset evaluated at current Q (synchronous)

        # we'll collect updates in a delta array to apply synchronously
        delta = np.zeros_like(Q)

        for s in states:
            for a in actions:
                # sample next state according to P[(s,a)]
                p = P[(s, a)]
                s_next = sample_next_state(p)
                g_sample = c[(s, a)]  # immediate cost (no noise here)
                # compute sample target
                min_Q_next = float(np.min(Q[s_next, :]))
                target = g_sample + min_Q_next - fQ - Q[s, a]
                delta[s, a] = gamma_n * target

        # synchronous update
        Q += delta

        # bookkeeping
        rho_trace.append(fQ)
        if (n % verbose_interval) == 0 or n == max_iter - 1:
            # greedy policy (minimizing Q for costs)
            greedy_policy = {s: int(np.argmin(Q[s, :])) for s in states}
            policy_trace.append((n, greedy_policy, fQ))
            if (n % verbose_interval) == 0:
                print(f"iter {n:6d}  f(Q)={fQ:.6f}  greedy_policy={greedy_policy}")

    return Q, rho_trace, policy_trace


# -------------------------
# Run the algorithm
# -------------------------
if __name__ == "__main__":
    Q_final, rhos, policy_hist = rvi_q_learning_synchronous(
        P=P, c=c, states=states, actions=actions,
        i0=0, f_fn=None, alpha=1.0,
        max_iter=12000, seed=123, verbose_interval=2000
    )

    print("\nFinal Q-values:")
    for s in states:
        print(f"s={s}:  Q = {Q_final[s, :]}")

    print("\nEstimated rho trace (last 10 values):")
    print(np.round(rhos[-10:], 6))

    print("\nFinal greedy policy (minimizing Q):")
    final_policy = {s: int(np.argmin(Q_final[s, :])) for s in states}
    print(final_policy)









import numpy as np
import random

# ============================================================
#  Toy MDP (your example)
# ============================================================

states = [0, 1]
actions = [0, 1]
nS = len(states)
nA = len(actions)

# Transition probabilities P[(s,a)] = array([P(s'=0), P(s'=1)])
P = {
    (0, 0): np.array([0.5, 0.5]),
    (0, 1): np.array([0.2, 0.8]),
    (1, 0): np.array([0.7, 0.3]),
    (1, 1): np.array([0.4, 0.6]),
}

# Immediate costs
c = {
    (0, 0): 1.0,
    (0, 1): 1.5,
    (1, 0): 0.5,
    (1, 1): 0.8,
}

# ------------------------------------------------------------
# Utility: sample next-state index from distribution
# ------------------------------------------------------------
def sample_state(pdist):
    return np.searchsorted(np.cumsum(pdist), random.random())

# ------------------------------------------------------------
# Offset function f(Q)
# Must satisfy:
#   f(Q + c*1) = f(Q) + c
# Here we use: f(Q) = min_b Q(i0, b), reference state i0
# ------------------------------------------------------------
def offset_function(Q, i0=0):
    return float(np.min(Q[i0, :]))

# ------------------------------------------------------------
# Asynchronous RVI–Q-learning
# Only one (state,action) is updated each iteration.
# ------------------------------------------------------------
def rvi_q_learning_asynchronous(
    max_iter=40000,
    alpha=1.0,          # learning rate parameter
    i0=0,               # reference state for f(Q)
    seed=0,
    verbose_interval=5000
):
    random.seed(seed)
    np.random.seed(seed)

    Q = np.zeros((nS, nA))           # initialize Q-table
    visit_count = np.zeros((nS, nA)) # ν(n,i,a)

    rho_trace = []

    for n in range(max_iter):

        # (1) Choose ONE random (s,a) to update — true asynchronous update
        s = random.choice(states)
        a = random.choice(actions)

        # (2) Generate next state sample ξ
        s_next = sample_state(P[(s,a)])

        # (3) Compute step-size using visit count γ(ν(n,i,a)) = alpha/(1+count)
        visit_count[s,a] += 1
        gamma = alpha / visit_count[s,a]

        # (4) Offset f(Q)
        fQ = offset_function(Q, i0)

        # (5) Q-learning Bellman target
        g_sample = c[(s,a)]
        min_Q_next = float(np.min(Q[s_next, :]))

        target = g_sample + min_Q_next - fQ - Q[s,a]

        # (6) Update
        Q[s,a] += gamma * target

        # Track estimated rho = f(Q)
        rho_trace.append(fQ)

        # Optional printing
        if (n % verbose_interval) == 0 and n > 0:
            policy = {s0: int(np.argmin(Q[s0, :])) for s0 in states}
            print(f"iter {n:6d}  f(Q)={fQ:.6f}  policy={policy}")

    # Return results
    final_policy = {s: int(np.argmin(Q[s,:])) for s in states}
    return Q, rho_trace, final_policy


# ============================================================
# Run Asynchronous RVI–Q-learning
# ============================================================
if __name__ == "__main__":
    Q_final, rho_hist, policy = rvi_q_learning_asynchronous(
        max_iter=40000,
        alpha=1.0,
        i0=0,
        seed=123,
        verbose_interval=5000
    )

    print("\nFinal Q-table:")
    for s in states:
        print(f"State {s}: Q = {Q_final[s]}")

    print("\nFinal greedy policy:")
    print(policy)

    print("\nLast 10 rho estimates:")
    print(np.round(rho_hist[-10:], 6))

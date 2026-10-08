import numpy as np
import math
import random
from typing import List, Tuple

# -----------------------------
# Single-arm MDP solver (uniformization + relative value iteration)
# -----------------------------
def solve_single_arm_policy(lam: float, mu: float, h: float, D: float, W: float, N: int,
                            tol: float = 1e-8, max_iter: int = 20000) -> Tuple[List[int], List[float], float]:
    """
    Solve single-arm average-cost MDP (states 0..N) with passive subsidy W using uniformization
    and relative value iteration.

    - lam: demand rate λ
    - mu: service rate µ
    - h: holding cost per unit-time
    - D: lost-sale cost per lost demand
    - W: passive subsidy (added as W*(1-a) per unit time)
    - N: truncation state (0..N). births at N keep you at N (reflecting/truncation).
    Returns:
     - policy: list of ints (0 passive, 1 active) for states 0..N
     - v: relative value function (length N+1)
     - g: long-run average cost (approx) under this W (scalar)
    """
    # uniformization rate
    q = lam + mu
    if q <= 0:
        q = 1.0

    # immediate per-step cost under action a:
    # - holding cost per unit-time: h * x -> per-step: (h * x) / q
    # - lost-sale expected per-step (when x==0): probability of demand in a uniformized step is lam/q,
    #   so expected lost-sale cost per-step is D * (lam / q) when x == 0.
    # - subsidy: W*(1-a) per unit time -> per-step: W*(1-a)/q

    def immediate_cost(x: int, a: int):
        base = (h * x) / q
        if x == 0:
            base += (D * lam) / q
        base += (W * (1 - a)) / q
        return base

    # transition probabilities for each (x,a) for one uniformized step (discrete-time)
    # for action a=1 (active): birth at rate mu -> prob mu/q (to x+1 or stay at N), death at rate lam -> prob lam/q (if x>0 -> x-1, if x==0 -> stays at 0 but we accounted lost sale in cost)
    # for a=0 (passive): no birth, only death prob lam/q (if x>0 -> x-1)
    def trans_probs(x: int, a: int):
        probs = np.zeros(N + 1)
        if a == 1:
            # birth
            if x < N:
                probs[x + 1] += mu / q
            else:
                probs[N] += mu / q
        # death (demand)
        if x > 0:
            probs[x - 1] += lam / q
        else:
            # x == 0: death (demand) does not change state in truncated model (it is a lost sale).
            # we already charged lost-sale expected cost in immediate_cost for x==0.
            pass
        probs[x] += 1.0 - probs.sum()  # self-loop probability
        return probs

    # initialize value function v (relative) and average cost g
    v = np.zeros(N + 1)
    g = 0.0  # will be approximated as the shift during RVI
    for it in range(max_iter):
        v_new = np.empty_like(v)
        for x in range(N + 1):
            # compute Q-values for actions 0 and 1
            qvals = []
            for a in (0, 1):
                probs = trans_probs(x, a)
                expected_v = probs.dot(v)
                qvals.append(immediate_cost(x, a) + expected_v)
            v_new[x] = min(qvals)
        # relative value normalization: subtract v_new[0] to keep v_new[0]=0
        shift = v_new[0]
        v_new -= shift
        # approx average cost g estimate is shift (per-step). Convert per-step to per-time by multiplying by q:
        g_new = shift * q
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            g = g_new
            break
        v = v_new
        g = g_new
    # derive greedy policy from final v
    policy = [0] * (N + 1)
    for x in range(N + 1):
        qvals = []
        for a in (0, 1):
            probs = trans_probs(x, a)
            qvals.append(immediate_cost(x, a) + probs.dot(v))
        policy[x] = int(np.argmin(qvals))  # 0 if passive is better, 1 if active is better

    return policy, v.tolist(), g

# -----------------------------
# Bisection to find numeric Whittle index for a given state x
# -----------------------------
def whittle_index_numeric_for_state(lam: float, mu: float, h: float, D: float,
                                    x_query: int, N: int,
                                    W_low: float = -1e6, W_high: float = 1e6,
                                    bisect_tol: float = 1e-2, max_bisect_iter: int = 60) -> float:
    """
    Find smallest W such that state x_query is passive under the optimal policy for that W.
    Uses bisection; solves MDP for each W via solve_single_arm_policy.
    """
    # helper: is state passive under W?
    def is_passive(W):
        policy, _, _ = solve_single_arm_policy(lam, mu, h, D, W, N)
        return policy[x_query] == 0

    # expand bracket if needed (but be careful - limit expansions)
    a, b = W_low, W_high
    # If already passive at lower bound, return lower bound (means index <= W_low)
    if is_passive(a):
        return a
    # If still not passive at upper bound, return upper bound (means index >= W_high)
    if not is_passive(b):
        return b

    for _ in range(max_bisect_iter):
        m = 0.5 * (a + b)
        if is_passive(m):
            b = m
        else:
            a = m
        if abs(b - a) < bisect_tol:
            break
    return 0.5 * (a + b)

# -----------------------------
# Compute numeric indices for states 0..N
# -----------------------------
def compute_whittle_indices_numeric(lam: float, mu: float, h: float, D: float, N: int,
                                    W_low: float = -1e6, W_high: float = 1e6,
                                    bisect_tol: float = 1e-2) -> List[float]:
    indices = []
    for x in range(0, N + 1):
        print(f"Computing W({x}) ...", end="", flush=True)
        W_x = whittle_index_numeric_for_state(lam, mu, h, D, x, N,
                                              W_low=W_low, W_high=W_high,
                                              bisect_tol=bisect_tol)
        print(f" {W_x:.4g}")
        indices.append(W_x)
    return indices

# -----------------------------
# Utilities: infer threshold from numeric indices
# -----------------------------
def inferred_threshold_from_indices(indices: List[float]) -> int:
    """
    Heuristic: the threshold B is smallest x such that W(x) <= 0.
    If none, return None.
    """
    for x, val in enumerate(indices):
        if val <= 0:
            return x
    return None

# -----------------------------
# Example / demo
# -----------------------------
if __name__ == "__main__":
    # Example class parameters (change as needed)
    classes = [
        {'lambda': 0.8, 'mu': 1.2, 'h': 0.5, 'D': 10.0},
        {'lambda': 0.6, 'mu': 1.0, 'h': 0.4, 'D': 8.0},
        {'lambda': 1.0, 'mu': 1.5, 'h': 0.7, 'D': 12.0}
    ]

    # truncation
    N = 30

    all_indices = []
    for i, c in enumerate(classes):
        print(f"\n=== Class {i+1} params: λ={c['lambda']}, µ={c['mu']}, h={c['h']}, D={c['D']} ===")
        idxs = compute_whittle_indices_numeric(c['lambda'], c['mu'], c['h'], c['D'], N,
                                               W_low=-1e6, W_high=1e6, bisect_tol=1e-2)
        th = inferred_threshold_from_indices(idxs)
        print(f"Inferred threshold (first x with W(x) <= 0): {th}")
        all_indices.append(idxs)

    # Print first few indices per class
    for i, idxs in enumerate(all_indices):
        print(f"\nClass {i+1} W(0..9):")
        print([round(v, 4) for v in idxs[:10]])

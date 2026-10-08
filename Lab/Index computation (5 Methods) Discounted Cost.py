"""
Single-machine repairman problem with switching cost, discounted case:
compute the Whittle index FIVE independent ways and compare them side by
side, both individually (per method) and in one summary table.

  1. Closed form   -- closed_form_indices()  (G/H recursion, w_active/w_passive)
  2. VI             -- whittle_index_discounted(..., solver='vi')  (value iteration
                        + bisection on Q0(s)-Q1(s)=0)
  3. PI             -- whittle_index_discounted(..., solver='pi')  (policy iteration
                        + bisection, same wrapper, different inner solver)
  4. Package        -- markovianbandit-pkg's own whittle_indices(discount=beta),
                        a fully independent third-party implementation
  5. Theorem        -- theorem_indices()  (the closed form exactly as stated in
                        Theorem "Indexability under perfect repair": W00/W01/Wk0)

Methods 1 and 5 are algebraically the same formula written two different
ways (verified identical earlier); methods 2 and 3 are two different
numerical algorithms solving the same bisection problem on the real MDP;
method 4 is an independent third-party implementation. All five should
agree WHENEVER C_switch is inside the theorem's proven validity domain
C_switch < c_bar (see c_bar() below) -- outside that domain, expect 1/5
to disagree with the other four, which is itself a diagnostic, not a bug.

This script does the DISCOUNTED case only. Average-cost is a natural
follow-up (using solve_rvi/solve_pi_avg/whittle_index_bisection_avg
instead of the discounted solvers here) but is left for later, as asked.
"""

import numpy as np
import pandas as pd

try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skipping package comparison.")


# =====================================================================
# 1. Extended MDP builder (perfect repair, with switching cost)
# =====================================================================
def build_extended_mdp(N, p, C, K, C_switch):
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)
    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N + 1)]
    for s in range(num_states):
        if s == 0:
            P1[s, 0] = 1.0
            R1[s] = -C
        else:
            P1[s, 0] = 1.0
            R1[s] = -(C + C_switch)
    for s in range(num_states):
        if s == 0:
            P0[s, 2] = p[0]; P0[s, 1] = 1 - p[0]; R0[s] = -(K * (1 - p[0]))
        elif s == 1:
            P0[s, 2] = p[0]; P0[s, 1] = 1 - p[0]; R0[s] = -(K * (1 - p[0]))
        else:
            idx = s - 1
            if idx < N:
                P0[s, s + 1] = p[idx]; P0[s, 1] = 1 - p[idx]; R0[s] = -(K * (1 - p[idx]))
            else:
                P0[s, 1] = 1.0; R0[s] = -K
    return P0, P1, R0, R1, state_labels


# =====================================================================
# 2. Discounted solvers (VI, PI) -- yours, unchanged
# =====================================================================
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
# 3. Closed-form discounted indices -- yours, unchanged
# =====================================================================
def compute_G_H(p, beta):
    N = len(p)
    H = np.zeros(N + 1)
    G = np.zeros(N + 2)
    prod = 1.0
    for k in range(N + 1):
        if k < N:
            H[k] = beta ** (k + 1) * prod
            prod *= p[k]
        else:
            H[k] = beta ** (k + 1) * prod
    G[0] = 0.0
    for k in range(N + 1):
        p_k = p[k] if k < N else 0.0
        G[k + 1] = G[k] + (1 - p_k) * H[k]
    return G, H


def closed_form_indices(p, beta, K, C, C_switch):
    N = len(p)
    G, H = compute_G_H(p, beta)
    p0 = p[0] if N > 0 else 0.0
    w00 = (1 - p0) * K - C - C_switch * (1 - beta)

    if N >= 1:
        G1 = G[1]
        H1 = H[1]
        num = (1 - beta) * (K * G1 + C_switch * H1)
        den = beta * (1 - G1) - H1
        w_active = (num / den) - C if abs(den) > 1e-15 else np.inf
    else:
        w_active = np.inf

    w_passive = np.zeros(N + 1)
    w_passive[0] = w00
    for k in range(1, N + 1):
        pk = p[k] if k < N else 0.0
        Gk = G[k]
        Hk = H[k]
        denom = (1 - pk * beta) * (1 - Gk) - (1 - pk) * Hk
        if abs(denom) > 1e-15:
            w_passive[k] = K - C - C_switch - (K * pk * (1 - beta)) / denom
        else:
            w_passive[k] = np.inf
    return w_active, w_passive


# =====================================================================
# 4. Theorem-based closed form (the other way of writing the same result,
#    from Theorem "Indexability under perfect repair") -- unchanged from
#    earlier in this conversation.
# =====================================================================
def theorem_indices(p, beta, K, C, C_switch):
    """Returns W as a length-(N+2) array in the same state order as
    build_extended_mdp: [W(0,1), W(0,0), W(1,0), ..., W(N,0)]."""
    N = len(p)
    G, H = compute_G_H(p, beta)
    p0 = p[0]
    W00 = (1 - p0) * K - C - (1 - beta) * C_switch
    W01 = (1 - p0) * K - C + beta * p0 * C_switch
    Wk0 = np.zeros(N + 1)
    for k in range(1, N + 1):
        pk = p[k] if k < N else 0.0
        Gk, Gk1 = G[k], G[k + 1]
        num = 1 - Gk1 - pk * (1 - beta * Gk)
        den = 1 - Gk1 - beta * pk * (1 - Gk)
        Wk0[k] = K * num / den - C - C_switch
    W = np.zeros(N + 2)
    W[0] = W01
    W[1] = W00
    for k in range(1, N + 1):
        W[1 + k] = Wk0[k]
    return W


def c_bar(K, p0, p1, beta):
    """Theorem's validity bound: the closed forms above are only proven
    valid for 0 <= C_switch < c_bar. Included here purely as a diagnostic
    -- if the five methods disagree, check this first."""
    return K * (p0 - p1) / (1 + beta * (p0 - p1))


# =====================================================================
# 5. Run all five methods on one instance and compare
# =====================================================================
if __name__ == "__main__":
    # ---- problem parameters (discounted case) ----
    N = 5
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)
    C, K = 5.0, 500.0
    C_switch = 90.0
    beta = 0.95

    cbar = c_bar(K, p[0], p[1], beta)
    print("=" * 78)
    print("PART 2 (DISCOUNTED): FIVE-WAY WHITTLE INDEX COMPARISON")
    print("=" * 78)
    print(f"N = {N}")
    print(f"p = {np.round(p, 4)}")
    print(f"C = {C}, K = {K}, C_switch = {C_switch}, beta = {beta}")
    print(f"c_bar (theorem's validity bound) = {cbar:.4f}   "
          f"[C_switch < c_bar: {C_switch < cbar}]")
    if C_switch >= cbar:
        print("  ** C_switch >= c_bar: the closed-form methods (1 and 5 below) are "
              "OUTSIDE their proven domain here -- expect them to disagree with "
              "VI/PI/Package, not the other way around. **")

    P0, P1, R0, R1, state_labels = build_extended_mdp(N, p_full, C, K, C_switch)
    S = len(R0)

    # ---------------- 1. Closed form ----------------
    print("\n" + "-" * 78)
    print("1. CLOSED FORM  (closed_form_indices)")
    print("-" * 78)
    w_active_cf, w_passive_cf = closed_form_indices(p, beta, K, C, C_switch)
    closed_form_vals = np.concatenate(([w_active_cf], w_passive_cf))
    print(f"  {state_labels[0]:8s}: W = {w_active_cf:.6f}")
    for k in range(N + 1):
        print(f"  {state_labels[1+k]:8s}: W = {w_passive_cf[k]:.6f}")

    # ---------------- 2. VI ----------------
    print("\n" + "-" * 78)
    print("2. VALUE ITERATION  (whittle_index_discounted, solver='vi')")
    print("-" * 78)
    vi_vals = np.zeros(S)
    for s in range(S):
        vi_vals[s] = whittle_index_discounted(state=s, solver='vi', beta=beta,
                                               P0=P0, P1=P1, R0=R0, R1=R1,
                                               w_min=-2000.0, w_max=2000.0,
                                               tol_w=1e-6, tol_mdp=1e-9)
        print(f"  {state_labels[s]:8s}: W = {vi_vals[s]:.6f}")

    # ---------------- 3. PI ----------------
    print("\n" + "-" * 78)
    print("3. POLICY ITERATION  (whittle_index_discounted, solver='pi')")
    print("-" * 78)
    pi_vals = np.zeros(S)
    for s in range(S):
        pi_vals[s] = whittle_index_discounted(state=s, solver='pi', beta=beta,
                                               P0=P0, P1=P1, R0=R0, R1=R1,
                                               w_min=-2000.0, w_max=2000.0,
                                               tol_w=1e-6, tol_mdp=1e-9)
        print(f"  {state_labels[s]:8s}: W = {pi_vals[s]:.6f}")

    # ---------------- 4. Package ----------------
    print("\n" + "-" * 78)
    print("4. PACKAGE  (markovianbandit-pkg, whittle_indices(discount=beta))")
    print("-" * 78)
    if PKG_AVAILABLE:
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        pkg_vals = np.array(model.whittle_indices(discount=beta))
        for s in range(S):
            print(f"  {state_labels[s]:8s}: W = {pkg_vals[s]:.6f}")
    else:
        pkg_vals = np.full(S, np.nan)
        print("  (skipped -- markovianbandit not installed)")

    # ---------------- 5. Theorem ----------------
    print("\n" + "-" * 78)
    print("5. THEOREM  (theorem_indices -- Theorem 'Indexability under perfect repair')")
    print("-" * 78)
    theorem_vals = theorem_indices(p, beta, K, C, C_switch)
    for s in range(S):
        print(f"  {state_labels[s]:8s}: W = {theorem_vals[s]:.6f}")

    # ---------------- summary table ----------------
    print("\n" + "=" * 78)
    print("SUMMARY TABLE -- all five methods, discounted")
    print("=" * 78)
    df = pd.DataFrame({
        "State": state_labels,
        "ClosedForm": closed_form_vals,
        "VI": vi_vals,
        "PI": pi_vals,
        "Package": pkg_vals,
        "Theorem": theorem_vals,
    })
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # quick agreement check across the four non-NaN-guaranteed columns
    ref = df["PI"].values  # PI/VI are the "ground truth" numerical solve of the real MDP
    print("\nMax |method - PI| across states:")
    for col in ["ClosedForm", "VI", "Package", "Theorem"]:
        vals = df[col].values
        if np.any(np.isnan(vals)):
            print(f"  {col:12s}: n/a (package not installed)")
        else:
            print(f"  {col:12s}: {np.max(np.abs(vals - ref)):.6f}")
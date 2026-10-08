"""
Heterogeneous multi-machine (R repairmen) repairman problem: compute the
Whittle index FIVE different ways per machine (Closed Form, VI, PI,
Package, Theorem), build the resulting FIVE index-heuristic joint
policies (one per method), plus the Greedy policy, evaluate all SIX
against the true joint OPT, and compare suboptimality.

Sampling now matches the (D, alpha) parameterization used everywhere else
in this work: each machine's C ~ U(D, D+25), K = alpha * C, rather than
an ad hoc C ~ U(10,35), K = 3*C used in an earlier draft of this script.

Two scenarios are run back to back:

  SCENARIO 1 -- every machine's C_switch stays INSIDE its own c_bar
  domain. Expectation: ClosedForm/Theorem are the same formula written
  two ways, and VI/PI/Package are three independent numerical solves of
  the same real MDP -- so all five should collapse to the SAME policy,
  and hence the same suboptimality numbers. This is duplication BY
  CONSTRUCTION, not a bug -- verified explicitly below via
  np.array_equal on the full policies, not just the summary statistics.

  SCENARIO 2 -- one machine's C_switch is deliberately pushed PAST its
  own c_bar. Expectation, based on the single-machine investigation
  earlier in this work: ClosedForm/Theorem (which assume the theorem's
  validity domain) should now split off from VI/PI/Package (which solve
  the real MDP regardless of any domain restriction, so they stay
  correct and continue to agree with each other). This is the contrast
  case that Scenario 1 can't show by construction.
"""

import numpy as np
import pandas as pd
from itertools import combinations

try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; Package column will be NaN.")


# =====================================================================
# 1. Single-machine local MDP
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
            P1[s, 0] = 1.0; R1[s] = -C
        else:
            P1[s, 0] = 1.0; R1[s] = -(C + C_switch)
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
# 2. Discounted solvers (VI, PI) -> single-machine index by bisection
# =====================================================================
def solve_vi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=10000):
    S = len(R0); V = np.zeros(S)
    for _ in range(max_iter):
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        V_new = np.maximum(Q0, Q1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    Q0 = R0 + beta * (P0 @ V); Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1


def policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta):
    S = len(pi)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])
    return np.linalg.solve(np.eye(S) - beta * P_pi, r_pi)


def solve_pi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=1000):
    S = len(R0); pi = np.zeros(S, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta)
        Q0 = R0 + beta * (P0 @ V); Q1 = (R1 - w) + beta * (P1 @ V)
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + beta * (P0 @ V); Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1


def whittle_index_discounted(state, solver, beta, P0, P1, R0, R1,
                              w_min=-2000.0, w_max=2000.0, tol_w=1e-6, tol_mdp=1e-9):
    solve_f = solve_vi_discounted if solver == 'vi' else solve_pi_discounted

    def f(w):
        _, Q0, Q1 = solve_f(P0, P1, R0, R1, w, beta, tol_mdp)
        return Q0[state] - Q1[state]

    f_min, f_max = f(w_min), f(w_max)
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
            w_min, f_min = w_mid, f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0


# =====================================================================
# 3. Closed-form (two equivalent derivations) + validity bound
# =====================================================================
def compute_G_H(p, beta):
    N = len(p); H = np.zeros(N + 1); G = np.zeros(N + 2); prod = 1.0
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
    N = len(p); G, H = compute_G_H(p, beta)
    p0 = p[0] if N > 0 else 0.0
    w00 = (1 - p0) * K - C - C_switch * (1 - beta)
    if N >= 1:
        G1, H1 = G[1], H[1]
        num = (1 - beta) * (K * G1 + C_switch * H1)
        den = beta * (1 - G1) - H1
        w_active = (num / den) - C if abs(den) > 1e-15 else np.inf
    else:
        w_active = np.inf
    w_passive = np.zeros(N + 1); w_passive[0] = w00
    for k in range(1, N + 1):
        pk = p[k] if k < N else 0.0
        Gk, Hk = G[k], H[k]
        denom = (1 - pk * beta) * (1 - Gk) - (1 - pk) * Hk
        w_passive[k] = (K - C - C_switch - (K * pk * (1 - beta)) / denom
                        if abs(denom) > 1e-15 else np.inf)
    return np.concatenate(([w_active], w_passive))


def theorem_indices(p, beta, K, C, C_switch):
    N = len(p); G, H = compute_G_H(p, beta)
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
    W = np.zeros(N + 2); W[0] = W01; W[1] = W00
    for k in range(1, N + 1):
        W[1 + k] = Wk0[k]
    return W


def c_bar(K, p0, p1, beta):
    return K * (p0 - p1) / (1 + beta * (p0 - p1))


# =====================================================================
# 4. Compute all 5 index arrays for one machine
# =====================================================================
def all_method_indices(mp, beta):
    N, p, C, K, C_switch = mp["N"], mp["p"], mp["C"], mp["K"], mp["C_switch"]
    P0, P1, R0, R1, labels = build_extended_mdp(N, np.append(p, 0.0), C, K, C_switch)
    S = N + 2

    W_cf = closed_form_indices(p, beta, K, C, C_switch)
    W_th = theorem_indices(p, beta, K, C, C_switch)
    W_vi = np.array([whittle_index_discounted(s, 'vi', beta, P0, P1, R0, R1) for s in range(S)])
    W_pi = np.array([whittle_index_discounted(s, 'pi', beta, P0, P1, R0, R1) for s in range(S)])

    if PKG_AVAILABLE:
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        W_pkg = np.array(model.whittle_indices(discount=beta))
    else:
        W_pkg = np.full(S, np.nan)

    return {"ClosedForm": W_cf, "VI": W_vi, "PI": W_pi, "Package": W_pkg, "Theorem": W_th}, labels


# =====================================================================
# 5. Heterogeneous joint MDP + policies
# =====================================================================
def enumerate_actions(M, R):
    R_eff = min(R, M)
    action_subsets = []
    for k in range(0, R_eff + 1):
        for combo in combinations(range(M), k):
            action_subsets.append(combo)
    bitmask_to_action = {}
    for a, combo in enumerate(action_subsets):
        mask = 0
        for m in combo:
            mask |= (1 << m)
        bitmask_to_action[mask] = a
    return action_subsets, bitmask_to_action


def build_joint_mdp_hetero(machine_locals, beta, R=1):
    M = len(machine_locals)
    L_list = [machine_locals[m]["P0"].shape[0] for m in range(M)]
    action_subsets, bitmask_to_action = enumerate_actions(M, R)
    P_joint, R_joint = [], []
    for combo in action_subsets:
        combo_set = set(combo)
        mats = [machine_locals[m]["P1"] if m in combo_set else machine_locals[m]["P0"] for m in range(M)]
        Pa = mats[0]
        for mat in mats[1:]:
            Pa = np.kron(Pa, mat)
        P_joint.append(Pa)
        shape = tuple(L_list)
        Rgrid = np.zeros(shape)
        for m in range(M):
            r_m = machine_locals[m]["R1"] if m in combo_set else machine_locals[m]["R0"]
            bshape = [1] * M; bshape[m] = L_list[m]
            Rgrid = Rgrid + r_m.reshape(bshape)
        R_joint.append(Rgrid.reshape(-1))
    return P_joint, R_joint, L_list, action_subsets, bitmask_to_action


def solve_joint_vi(P_joint, R_joint, beta, tol=1e-8, max_iter=5000):
    S = len(R_joint[0]); A = len(R_joint); V = np.zeros(S)
    for _ in range(max_iter):
        Qs = np.stack([R_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
        V_new = Qs.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new; break
        V = V_new
    Qs = np.stack([R_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
    return V, Qs.argmax(axis=0)


def evaluate_joint_policy(pi, P_joint, R_joint, beta):
    S = len(R_joint[0]); P_pi = np.zeros((S, S)); r_pi = np.zeros(S)
    for a in range(len(R_joint)):
        mask = (pi == a)
        P_pi[mask, :] = P_joint[a][mask, :]
        r_pi[mask] = R_joint[a][mask]
    return np.linalg.solve(np.eye(S) - beta * P_pi, r_pi)


def index_policy_from_W(W_list, L_list, M, R, bitmask_to_action):
    S = int(np.prod(L_list)); shape = tuple(L_list)
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        local_idx = np.unravel_index(s, shape)
        w_vals = [(W_list[m][local_idx[m]], m) for m in range(M)]
        candidates = [(w, m) for w, m in w_vals if w > 0]
        candidates.sort(key=lambda x: -x[0])
        chosen = sorted(m for _, m in candidates[:min(R, len(candidates))])
        mask = 0
        for m in chosen:
            mask |= (1 << m)
        pi[s] = bitmask_to_action[mask]
    return pi


def damage_level(L, N):
    levels = np.zeros(L, dtype=int)
    levels[0] = 0; levels[1] = 0
    for s in range(2, L):
        levels[s] = s - 1
    return levels


def greedy_policy_hetero(L_list, M, N_list, R, bitmask_to_action):
    S = int(np.prod(L_list)); shape = tuple(L_list)
    levels_per_machine = [damage_level(L_list[m], N_list[m]) for m in range(M)]
    R_eff = min(R, M)
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        local_idx = np.unravel_index(s, shape)
        lvl_vals = [(levels_per_machine[m][local_idx[m]], m) for m in range(M)]
        lvl_vals.sort(key=lambda x: (-x[0], x[1]))
        chosen = sorted(m for _, m in lvl_vals[:R_eff])
        mask = 0
        for m in chosen:
            mask |= (1 << m)
        pi[s] = bitmask_to_action[mask]
    return pi


def sample_glazebrook_p(N, rng):
    return np.sort(rng.uniform(0.0, 1.0, N))[::-1]


# =====================================================================
# 6. One full scenario: build machines at given switch_ratios (per-machine
#    multiplier of that machine's own c_bar), compute all 5 index methods,
#    build 6 policies, evaluate against true joint OPT.
# =====================================================================
def run_scenario(label, N_list, switch_ratios, D, alpha, beta=0.95, R=1, seed=7):
    """Uses the SAME (D, alpha) parameterization as the rest of this work:
    C ~ U(D, D+25) per machine, K = alpha * C. switch_ratios[m] =
    C_switch_m / c_bar_m -- a value < 1 keeps machine m INSIDE its
    theorem-valid domain; >= 1 pushes it OUTSIDE (closed-form formulas no
    longer guaranteed correct there)."""
    M = len(N_list)
    rng = np.random.default_rng(seed)
    machine_params = []
    for N, ratio in zip(N_list, switch_ratios):
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + 25)
        K = alpha * C
        cbar_ = c_bar(K, p[0], p[1], beta)
        C_switch = ratio * cbar_
        machine_params.append(dict(N=N, p=p, C=C, K=K, C_switch=C_switch))

    print("=" * 88)
    print(f"{label}   R={R}   N_list={N_list}   D={D}  alpha={alpha}   switch_ratios={switch_ratios}")
    print("=" * 88)

    machine_locals = []
    method_W = {}
    for m, mp in enumerate(machine_params):
        methods, labels = all_method_indices(mp, beta)
        method_W[m] = methods
        P0, P1, R0, R1, _ = build_extended_mdp(mp["N"], np.append(mp["p"], 0.0),
                                                mp["C"], mp["K"], mp["C_switch"])
        machine_locals.append(dict(P0=P0, P1=P1, R0=R0, R1=R1))
        cbar_m = c_bar(mp["K"], mp["p"][0], mp["p"][1], beta)
        inside = mp["C_switch"] < cbar_m
        print(f"\nMachine {m+1}: N={mp['N']}  C={mp['C']:.3f}  K={mp['K']:.3f}  "
              f"C_switch={mp['C_switch']:.3f}  c_bar={cbar_m:.3f}  "
              f"[{'INSIDE' if inside else 'OUTSIDE'} domain]")
        for method, W in methods.items():
            print(f"    {method:10s}: {np.round(W, 4)}")

    P_joint, R_joint, L_list, action_subsets, bitmask_to_action = build_joint_mdp_hetero(
        machine_locals, beta, R=R)
    print(f"\nJoint state space size: {int(np.prod(L_list))} = {'*'.join(map(str, L_list))}")

    V_opt, pi_star = solve_joint_vi(P_joint, R_joint, beta)

    results = {}
    for method in ["ClosedForm", "VI", "PI", "Package", "Theorem"]:
        W_list = [method_W[m][method] for m in range(M)]
        if any(np.any(np.isnan(w)) for w in W_list):
            results[method] = None
            continue
        pi_method = index_policy_from_W(W_list, L_list, M, R, bitmask_to_action)
        V_method = evaluate_joint_policy(pi_method, P_joint, R_joint, beta)
        subopt = 100.0 * (V_method - V_opt) / V_opt
        results[method] = (pi_method, subopt)

    pi_greedy = greedy_policy_hetero(L_list, M, N_list, R, bitmask_to_action)
    V_greedy = evaluate_joint_policy(pi_greedy, P_joint, R_joint, beta)
    subopt_greedy = 100.0 * (V_greedy - V_opt) / V_opt
    results["Greedy"] = (pi_greedy, subopt_greedy)

    print("\n" + "-" * 88)
    print("SUBOPTIMALITY (%) SUMMARY")
    print("-" * 88)
    rows = []
    for method, val in results.items():
        if val is None:
            rows.append(dict(Method=method, q1=np.nan, median=np.nan, q3=np.nan, max=np.nan,
                              pct_matches_OPT=np.nan))
            continue
        pi_method, subopt = val
        rows.append(dict(
            Method=method,
            q1=np.percentile(subopt, 25), median=np.percentile(subopt, 50),
            q3=np.percentile(subopt, 75), max=subopt.max(),
            pct_matches_OPT=100.0 * np.mean(pi_method == pi_star),
        ))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDUPLICATION CHECK:")
    pi_cf, pi_th = results["ClosedForm"][0], results["Theorem"][0]
    pi_vi, pi_pi = results["VI"][0], results["PI"][0]
    print("  ClosedForm policy == Theorem policy (every state)? ", np.array_equal(pi_cf, pi_th))
    print("  VI policy == PI policy (every state)?              ", np.array_equal(pi_vi, pi_pi))
    if results["Package"] is not None:
        pi_pkg = results["Package"][0]
        print("  VI policy == Package policy (every state)?         ", np.array_equal(pi_vi, pi_pkg))
        print("  ClosedForm policy == VI policy (every state)?      ", np.array_equal(pi_cf, pi_vi))
        n_diff = np.sum(pi_cf != pi_vi)
        print(f"  ClosedForm vs VI: differ on {n_diff}/{len(pi_cf)} states "
              f"({100*n_diff/len(pi_cf):.1f}%)")

    print("\nAll(V_opt >= V_method) sanity check (none beats true OPT):")
    for method, val in results.items():
        if val is None:
            continue
        _, subopt = val
        print(f"  {method:10s}: {np.all(subopt >= -1e-6)}")
    return df


if __name__ == "__main__":
    N_list = [3, 4, 5]
    D, alpha = 20.0, 3.0   # same (D, alpha) parameterization used throughout:
                           # C ~ U(D, D+25) per machine, K = alpha * C

    # D_grid = [0,10,20,25,30,40,50]
    # alpha_grid = [1.5, 2.0, 3.0, 4.0, 5.0]

    # D_grid = np.round(np.linspace(0, 50, 7), 2)
    # alpha_grid = np.round(np.linspace(1.5, 5.0, 6), 2)                      

    # SCENARIO 1: every machine safely inside its own c_bar domain.
    df1 = run_scenario("SCENARIO 1 -- all machines INSIDE c_bar domain",
                        N_list, switch_ratios=[0.5, 0.5, 0.5], D=D, alpha=alpha)

    print("\n\n")

    # SCENARIO 2: machine 2 (the last one) deliberately pushed past its own
    # c_bar; machines 0 and 1 stay inside theirs. seed=8, ratio=5.0 chosen
    # (via a small scan over seeds/ratios under this D, alpha sampling --
    # see the conversation this script came from) specifically because it
    # produces a genuine policy split, not just a numerical wobble that
    # happens not to flip any decision.
    df2 = run_scenario("SCENARIO 2 -- machine 2 pushed OUTSIDE its c_bar domain",
                        N_list, switch_ratios=[0.5, 0.5, 5.0], D=D, alpha=alpha, seed=8)
"""
Heterogeneous multi-machine (R repairmen) repairman problem: compute the
Whittle index FIVE different ways per machine (Closed Form, VI, PI,
Package, Theorem), build the resulting FIVE index-heuristic joint
policies (one per method), plus the Greedy policy, evaluate all SIX
against the true joint OPT, and compare suboptimality.

This version generalizes the earlier two-scenario script into a fully
configurable experiment. Everything that used to be hard-coded in
`__main__` is now a field on ExperimentConfig:

    (i)    number of machines               -> cfg.M / cfg.N_list
    (ii)   number of repairmen               -> cfg.R
    (iii)  discount factor                   -> cfg.beta
    (iv)   heterogeneity of states/machine   -> cfg.N_list (fixed) or
                                                 cfg.N_range (random per rep)
    (v)    replications per grid cell        -> cfg.n_reps
    (vi)   D_grid                            -> cfg.D_grid
    (vii)  alpha_grid                        -> cfg.alpha_grid
    (viii) switching-cost fractions of c_bar -> cfg.switch_fracs
               (C_switch = switch_frac * c_bar(machine); < 1 stays inside
               the theorem's proven-valid domain, 0.0 recovers Glazebrook's
               own no-switching-cost model, >= 1 pushes outside the domain)

Sampling matches the (D, alpha) parameterization used throughout this
work: each machine's C ~ U(D, D + C_width), K = alpha * C.

Every replication is one random (p, C, K, C_switch) draw for M machines
at a given (D, alpha, switch_frac). All SIX policies (ClosedForm, VI, PI,
Package, Theorem, Greedy) are built and evaluated against the true joint
OPT for that replication. Results are printed to the terminal AND to a
text file (cfg.output_txt), and every replication's summary row is also
saved to a CSV (cfg.output_csv) for further analysis, with a final
aggregate table (mean over n_reps) printed at the end.
"""

import os
import sys
import datetime
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False


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
# 6. Dual terminal + file output
# =====================================================================
class TeeWriter:
    """Writes every .write() call to all given streams (e.g. stdout + a file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# =====================================================================
# 7. Experiment configuration -- everything tunable lives here
# =====================================================================
@dataclass
class ExperimentConfig:
    # (i) / (iv) machines & per-machine heterogeneity in number of damage levels N.
    # Set N_list explicitly for a fixed, repeatable machine population (length = M).
    # Set N_list=None to instead draw each machine's N ~ Uniform{N_range} fresh
    # every replication (M machines drawn from N_range each rep) for maximum
    # heterogeneity across the experiment.
    N_list: Optional[List[int]] = field(default_factory=lambda: [3, 4, 5])
    M: int = 3                          # only used when N_list is None
    N_range: Tuple[int, int] = (3, 6)   # inclusive range to sample N from, when N_list is None

    # (ii) number of repairmen (activation budget per period)
    R: int = 1

    # (iii) discount factor
    beta: float = 0.95

    # (v) replications per (D, alpha, switch_frac) grid cell
    n_reps: int = 5

    # (vi) / (vii) grids over the (D, alpha) parameterization: C ~ U(D, D+C_width), K = alpha*C
    D_grid: List[float] = field(default_factory=lambda: [20.0, 50.0])
    alpha_grid: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    C_width: float = 25.0

    # (viii) switching-cost fractions of each machine's own c_bar.
    # C_switch = switch_frac * c_bar(machine). switch_frac < 1 stays inside the
    # theorem's proven-valid domain; 0.0 recovers Glazebrook's own
    # no-switching-cost model; switch_frac >= 1 pushes outside the domain.
    switch_fracs: List[float] = field(default_factory=lambda: [0.0, 0.5, 0.9, 1.5, 3.0, 5.0])

    # misc
    seed0: int = 1000
    verbose_indices: bool = True   # print full per-machine index arrays each replication
    output_txt: str = "repairman_results.txt"
    output_csv: str = "repairman_results.csv"


def resolve_N_list(cfg: ExperimentConfig, rng: np.random.Generator) -> List[int]:
    if cfg.N_list is not None:
        return list(cfg.N_list)
    lo, hi = cfg.N_range
    return [int(rng.integers(lo, hi + 1)) for _ in range(cfg.M)]


# =====================================================================
# 8. One replication: sample machines, compute all 5 index methods,
#    build 6 policies, evaluate against true joint OPT.
# =====================================================================
def run_one_replication(cfg: ExperimentConfig, D, alpha, switch_frac, seed, verbose=True):
    rng = np.random.default_rng(seed)
    N_list = resolve_N_list(cfg, rng)
    M = len(N_list)
    beta = cfg.beta

    machine_params = []
    for N in N_list:
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + cfg.C_width)
        K = alpha * C
        cbar_ = c_bar(K, p[0], p[1], beta)
        C_switch = switch_frac * cbar_
        machine_params.append(dict(N=N, p=p, C=C, K=K, C_switch=C_switch, c_bar=cbar_))

    if verbose:
        print("=" * 88)
        print(f"D={D}  alpha={alpha}  switch_frac={switch_frac}  R={cfg.R}  beta={beta}  "
              f"N_list={N_list}  seed={seed}")
        print("=" * 88)

    machine_locals = []
    method_W = {}
    for m, mp in enumerate(machine_params):
        methods, labels = all_method_indices(mp, beta)
        method_W[m] = methods
        P0, P1, R0, R1, _ = build_extended_mdp(mp["N"], np.append(mp["p"], 0.0),
                                                mp["C"], mp["K"], mp["C_switch"])
        machine_locals.append(dict(P0=P0, P1=P1, R0=R0, R1=R1))
        inside = mp["C_switch"] < mp["c_bar"]
        if verbose and cfg.verbose_indices:
            print(f"\nMachine {m + 1}: N={mp['N']}  C={mp['C']:.3f}  K={mp['K']:.3f}  "
                  f"C_switch={mp['C_switch']:.3f}  c_bar={mp['c_bar']:.3f}  "
                  f"[{'INSIDE' if inside else 'OUTSIDE'} domain]")
            for method, W in methods.items():
                print(f"    {method:10s}: {np.round(W, 4)}")

    P_joint, R_joint, L_list, action_subsets, bitmask_to_action = build_joint_mdp_hetero(
        machine_locals, beta, R=cfg.R)

    V_opt, pi_star = solve_joint_vi(P_joint, R_joint, beta)

    method_results = {}
    for method in ["ClosedForm", "VI", "PI", "Package", "Theorem"]:
        W_list = [method_W[m][method] for m in range(M)]
        if any(np.any(np.isnan(w)) for w in W_list):
            method_results[method] = None
            continue
        pi_method = index_policy_from_W(W_list, L_list, M, cfg.R, bitmask_to_action)
        V_method = evaluate_joint_policy(pi_method, P_joint, R_joint, beta)
        subopt = 100.0 * (V_method - V_opt) / V_opt
        method_results[method] = (pi_method, subopt)

    pi_greedy = greedy_policy_hetero(L_list, M, N_list, cfg.R, bitmask_to_action)
    V_greedy = evaluate_joint_policy(pi_greedy, P_joint, R_joint, beta)
    subopt_greedy = 100.0 * (V_greedy - V_opt) / V_opt
    method_results["Greedy"] = (pi_greedy, subopt_greedy)

    rows = []
    for method, val in method_results.items():
        if val is None:
            rows.append(dict(D=D, alpha=alpha, switch_frac=switch_frac, seed=seed, Method=method,
                              q1=np.nan, median=np.nan, q3=np.nan, max=np.nan, pct_matches_OPT=np.nan))
            continue
        pi_method, subopt = val
        rows.append(dict(
            D=D, alpha=alpha, switch_frac=switch_frac, seed=seed, Method=method,
            q1=np.percentile(subopt, 25), median=np.percentile(subopt, 50),
            q3=np.percentile(subopt, 75), max=subopt.max(),
            pct_matches_OPT=100.0 * np.mean(pi_method == pi_star),
        ))
    df = pd.DataFrame(rows)

    if verbose:
        print(f"\nJoint state space size: {int(np.prod(L_list))} = " + "*".join(map(str, L_list)))
        print("\n" + "-" * 88)
        print("SUBOPTIMALITY (%) SUMMARY -- this replication")
        print("-" * 88)
        print(df.drop(columns=["D", "alpha", "switch_frac", "seed"])
                .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\nDUPLICATION CHECK:")
        if method_results["ClosedForm"] is not None and method_results["Theorem"] is not None:
            print("  ClosedForm policy == Theorem policy?", 
                  np.array_equal(method_results["ClosedForm"][0], method_results["Theorem"][0]))
        print("  VI policy == PI policy?             ",
              np.array_equal(method_results["VI"][0], method_results["PI"][0]))
        if method_results["Package"] is not None:
            print("  VI policy == Package policy?        ",
                  np.array_equal(method_results["VI"][0], method_results["Package"][0]))
        if method_results["ClosedForm"] is not None:
            pi_cf, pi_vi = method_results["ClosedForm"][0], method_results["VI"][0]
            n_diff = int(np.sum(pi_cf != pi_vi))
            print(f"  ClosedForm vs VI: differ on {n_diff}/{len(pi_cf)} states "
                  f"({100 * n_diff / len(pi_cf):.1f}%)")

        print("\nSanity check V_opt >= V_method (none beats true OPT):")
        for method, val in method_results.items():
            if val is None:
                continue
            _, subopt = val
            print(f"  {method:10s}: {bool(np.all(subopt >= -1e-6))}")
        print()

    return df


# =====================================================================
# 9. Full grid sweep
# =====================================================================
def main():
    # ---------------------------------------------------------------
    # EDIT THIS BLOCK TO CHOOSE YOUR EXPERIMENT.
    # Runtime scales as len(D_grid) * len(alpha_grid) * len(switch_fracs) * n_reps.
    # The defaults below (2*3*6*5 = 180 replications) take a few minutes; trim
    # any of the grids or n_reps for a quick smoke test.
    # ---------------------------------------------------------------
    cfg = ExperimentConfig(
        N_list=[3, 4, 5],            # fixed heterogeneous machine population
        # N_list=None, M=4, N_range=(3, 6),   # <- alternative: random heterogeneity each rep
        R=1,
        beta=0.95,
        n_reps=5,
        D_grid=[20.0, 50.0],
        alpha_grid=[2.0, 3.0, 5.0],
        C_width=25.0,
        switch_fracs=[0.0, 0.5, 0.9, 1.5, 3.0, 5.0],
        seed0=1000,
        verbose_indices=True,      # True : print full per-machine index arrays every single replication
        output_txt="repairman_results.txt",    # False : just the per-rep suboptimality table + duplication check, skip the raw index dumps
        output_csv="repairman_results.csv",
    )
    # ---------------------------------------------------------------

    old_stdout = sys.stdout
    logfile = open(cfg.output_txt, "w", encoding="utf-8")
    sys.stdout = TeeWriter(old_stdout, logfile)

    try:
        print(f"Run started: {datetime.datetime.now().isoformat()}")
        print(f"markovianbandit package available: {PKG_AVAILABLE}")
        print(f"Config: {cfg}\n")

        all_rows = []
        seed = cfg.seed0
        total_reps = len(cfg.D_grid) * len(cfg.alpha_grid) * len(cfg.switch_fracs) * cfg.n_reps
        run_idx = 0
        for D in cfg.D_grid:
            for alpha in cfg.alpha_grid:
                for switch_frac in cfg.switch_fracs:
                    for rep in range(cfg.n_reps):
                        run_idx += 1
                        seed += 1
                        print(f"\n########## Replication {run_idx}/{total_reps}  "
                              f"(D={D}, alpha={alpha}, switch_frac={switch_frac}, "
                              f"rep={rep + 1}/{cfg.n_reps}) ##########")
                        df = run_one_replication(cfg, D, alpha, switch_frac, seed, verbose=True)
                        all_rows.append(df)

        full_df = pd.concat(all_rows, ignore_index=True)
        full_df.to_csv(cfg.output_csv, index=False)

        print("\n\n" + "=" * 88)
        print("AGGREGATE SUMMARY ACROSS ALL REPLICATIONS (mean over n_reps)")
        print("=" * 88)
        agg = (full_df.groupby(["D", "alpha", "switch_frac", "Method"])
                       .agg(mean_median_subopt=("median", "mean"),
                            mean_max_subopt=("max", "mean"),
                            mean_pct_matches_OPT=("pct_matches_OPT", "mean"),
                            n_reps=("median", "count"))
                       .reset_index())
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print(f"\nFull per-replication results written to: {os.path.abspath(cfg.output_csv)}")
        print(f"This transcript written to: {os.path.abspath(cfg.output_txt)}")
        print(f"Run finished: {datetime.datetime.now().isoformat()}")
    finally:
        sys.stdout = old_stdout
        logfile.close()


if __name__ == "__main__":
    main()
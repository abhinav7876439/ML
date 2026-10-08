"""
Machine repairman problem with switching cost: reproduces Glazebrook's Table 3/4
methodology exactly, extended to include the switching cost from Theorem
"Indexability under perfect repair".

For each (D, alpha) cell of a 7x6 grid (as in Glazebrook), 10 random R=1, M=3
problems are drawn (8 damage states/machine), For 8 damage levels (N=7 non-terminal + 1 terminal), that's:
Reachable per-machine states: {(0,1)} ∪ {(x,0) : x=0..7} = 1 + 8 = 9 (not 16)
Joint state space for M=3: 9³ = 729:

    1. V_OPT(s)   -- the true optimal discounted cost, via value iteration on
                     the full joint MDP (the "expensive but possible" DP of
                     Glazebrook's Table 1 discussion, generalised with the
                     extra per-machine (0,1)/(0,0) flag the switching cost
                     needs, so the joint state space is 9^3=729 rather than
                     8^3=512).
    2. V_INDEX(s) -- the discounted cost of the index-heuristic policy: at
                     every joint state, service the machine with the highest
                     Whittle index (Theorem eqs. W00/W01/Wk0), or idle if all
                     three are negative -- exactly Glazebrook's stated rule --
                     evaluated by solving the linear system for that fixed
                     policy (not simulation).
    3. the % cost suboptimality 100*(V_INDEX - V_OPT)/V_OPT at every one of
       the 729 states (this is Glazebrook's own suboptimality definition,
       just computed in reward-as-negative-cost sign convention -- see the
       note in `run_cell` below).

Exactly as in Glazebrook Table 3/4, each (D, alpha) table entry summarises
10 problems x 729 initial states = 7290 suboptimality values as a
(lower quartile, median, upper quartile) triple, plus the % of those 7290
problem-states in which the index policy chooses to idle.

IMPORTANT -- the switching cost domain restriction: the Theorem's closed-form
indices are only valid for
    0 <= C_switch < c_bar := K*(p0-p1) / (1 + beta*(p0-p1))
c_bar depends on that machine's own p0, p1, K, and beta -- it is NOT simply
proportional to C or K. This script computes c_bar per machine and draws
C_switch = switch_frac * c_bar (switch_frac in [0,1)) so every instance is
guaranteed inside the theorem's validity region. If you want to probe what
happens once C_switch >= c_bar, set switch_frac >= 1 -- but note the closed
form formulas are not proven valid there (see the conversation this script
came out of): a different Whittle index would need to be substituted (e.g.
computed via bisection/PI on the local MDP) to still get meaningful
suboptimality numbers in that regime.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime

# =====================================================================
#  Single-machine local MDP
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
#  Closed-form Whittle indices, exactly per Theorem "Indexability under
#    perfect repair" (W00 / W01 / Wk0 equations), plus the theorem's own
#    validity bound c_bar.
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
    """Theorem's validity bound: closed form is only proven valid for
    0 <= C_switch < c_bar."""
    return K * (p0 - p1) / (1 + beta * (p0 - p1))


# =====================================================================
#  Joint (R=1 repairman, M machines) MDP via Kronecker structure, and
#    the true joint DP solver -- this is "expensive but
#    possible" full value iteration over all machine-damage combinations.
# =====================================================================
def build_joint_mdp(machine_params, beta):
    """machine_params: list of dicts with keys N, p, C, K, C_switch (one per
    machine). All machines must share the same local state count (same N).
    Returns:
      P_joint : list of (L^M, L^M) matrices, P_joint[0]=idle,
                P_joint[a]=service machine a-1 for a=1..M
      R_joint : list of (L^M,) reward vectors, same indexing
      locals_ : per-machine dict with P0,P1,R0,R1,labels,W (Whittle index)
      L       : local state count per machine (N+2)
    """
    M = len(machine_params)
    locals_ = []
    for mp in machine_params:
        P0, P1, R0, R1, labels = build_extended_mdp(
            mp["N"], np.append(mp["p"], 0.0), mp["C"], mp["K"], mp["C_switch"])
        W = theorem_indices(mp["p"], beta, mp["K"], mp["C"], mp["C_switch"])
        locals_.append(dict(P0=P0, P1=P1, R0=R0, R1=R1, labels=labels, W=W))

    L = locals_[0]["P0"].shape[0]
    for l in locals_:
        assert l["P0"].shape[0] == L, "all machines must share the same N for this joint builder"

    P_joint, R_joint = [], []
    for a in range(M + 1):  # a=0 idle, a=1..M service machine a-1
        mats = [locals_[m]["P1"] if a == m + 1 else locals_[m]["P0"] for m in range(M)]
        Pa = mats[0]
        for mat in mats[1:]:
            Pa = np.kron(Pa, mat)
        P_joint.append(Pa)

        shape = (L,) * M
        Rgrid = np.zeros(shape)
        for m in range(M):
            r_m = locals_[m]["R1"] if a == m + 1 else locals_[m]["R0"]
            bshape = [1] * M
            bshape[m] = L
            Rgrid = Rgrid + r_m.reshape(bshape)
        R_joint.append(Rgrid.reshape(-1))
    return P_joint, R_joint, locals_, L

# ----------------------------------------------------------------------
# Discounted solvers (VI)
# ----------------------------------------------------------------------
def solve_joint_vi(P_joint, R_joint, beta, tol=1e-8, max_iter=5000):
    """Full value iteration -> V_OPT and the true optimal joint policy."""
    S = len(R_joint[0])
    A = len(R_joint)
    V = np.zeros(S)
    for _ in range(max_iter):
        Qs = np.stack([R_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
        V_new = Qs.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Qs = np.stack([R_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
    pi_star = Qs.argmax(axis=0)
    return V, pi_star, Qs


def evaluate_joint_policy(pi, P_joint, R_joint, beta):
    """Exact value function of a fixed deterministic joint policy pi (array
    of length S giving an action per joint state), via linear solve."""
    S = len(R_joint[0])
    P_pi = np.zeros((S, S))
    r_pi = np.zeros(S)
    for a in range(len(R_joint)):
        mask = (pi == a)
        P_pi[mask, :] = P_joint[a][mask, :]
        r_pi[mask] = R_joint[a][mask]
    V = np.linalg.solve(np.eye(S) - beta * P_pi, r_pi)
    return V


def index_policy(locals_, L, M):
    """Rule: service the machine with the highest Whittle
    index; idle if all M indices are negative. Returns an action per joint
    state (0=idle, 1..M=service machine a-1), vectorised over all L^M
    states at once."""
    shape = (L,) * M
    idx_grids = []
    for m in range(M):
        bshape = [1] * M
        bshape[m] = L
        idx_grids.append(locals_[m]["W"].reshape(bshape) * np.ones(shape))
    W_stack = np.stack(idx_grids, axis=0).reshape(M, -1)  # (M, L^M)
    best_m = W_stack.argmax(axis=0)
    best_val = W_stack.max(axis=0)
    return np.where(best_val > 0, best_m + 1, 0)


# =====================================================================
# Glazebrook-style (D, alpha) grid experiment
# =====================================================================
def sample_glazebrook_p(N, rng):
    raw = rng.uniform(0.0, 1.0, N)
    return np.sort(raw)[::-1]


def run_cell(N, M, beta, D, alpha, switch_frac, rng):
    """One random R=1, M-machine problem: draw p (Glazebrook-style ordered
    sample), C ~ U(D, D+25), K = alpha*C per machine, and C_switch =
    switch_frac * c_bar (per-machine c_bar, so every instance stays inside
    the theorem's validity domain when switch_frac < 1). Returns the
    suboptimality (%) and idle-flag arrays over all L^M initial states."""
    machine_params = []
    for _ in range(M):
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + 25)
        K = alpha * C
        cbar = c_bar(K, p[0], p[1], beta)
        C_switch = switch_frac * cbar
        machine_params.append(dict(N=N, p=p, C=C, K=K, C_switch=C_switch))

    P_joint, R_joint, locals_, L = build_joint_mdp(machine_params, beta)
    print(f"  D={D:6.2f} alpha={alpha:4.2f}  C_switch={C_switch:.3f}  c_bar={cbar:.3f}")
    #print(f"    Machine 0 labels: {locals_[0]['labels']}")
    #print(f"    Machine 1 labels: {locals_[1]['labels']}")
    #print(f"    Machine 2 labels: {locals_[2]['labels']}")
    print(f"    Joint state space size: {L**M} = {L}^{M}")
    print(f"    Joint action space size: {M+1} = {M}+1 (idle + service each machine)")
    print(f"    Joint transition matrices: {[P.shape for P in P_joint]}")
    print(f"    Joint reward vectors: {[R.shape for R in R_joint]}")
    #print(f" P_joint =\n{P_joint}")
    #print(f" R_joint =\n{R_joint}")
    # print(f" Locals =\n{locals_}")
    # print(f"    Machine 0 Whittle indices: {locals_[0]['W']}")
    # print(f"    Machine 1 Whittle indices: {locals_[1]['W']}")
    # print(f"    Machine 2 Whittle indices: {locals_[2]['W']}") 
    V_opt, pi_star, _ = solve_joint_vi(P_joint, R_joint, beta)
    pi_idx = index_policy(locals_, L, M)
    print(f"    Optimal policy (pi_star) = {pi_star}")
    print(f"    Index policy (pi_idx) = {pi_idx}")
    V_idx = evaluate_joint_policy(pi_idx, P_joint, R_joint, beta)
    print(f"    V_opt = {V_opt}")
    print(f"    V_idx = {V_idx}")
    print(f"sum of V_opt = {V_opt.sum():.3f}, sum of V_idx = {V_idx.sum():.3f}")

    # Sign note: R0/R1 are negative costs, so V = -discounted cost, and
    # V_OPT >= V_idx (OPT has the higher/less-negative reward = lower cost).
    # Glazebrook's suboptimality = 100*(Cost_idx - Cost_opt)/Cost_opt
    #   = 100*((-V_idx) - (-V_opt)) / (-V_opt) = 100*(V_idx - V_opt)/V_opt
    # which is >= 0 given the sign conventions above -- verified in testing.
    subopt_pct = 100.0 * (V_idx - V_opt) / V_opt
    idle_flags = (pi_idx == 0).astype(float)
    return subopt_pct, idle_flags


def _emit(msg, f=None):
    """Print to console AND, if a file handle was given, write the same
    line to the results text file."""
    print(msg)
    if f is not None:
        f.write(msg + "\n")


def run_glazebrook_table(N=7, M=3, beta=0.95, switch_frac=0.5, n_reps=1,
                          D_grid=None, alpha_grid=None, seed=42, verbose=True,
                          f=None):
    """Reproduces one Glazebrook Table-3-style 7x6 grid of four-vector
    entries (here: quartile triple + % idle), for the switching-cost
    extension. Returns a tidy DataFrame, one row per (D, alpha) cell.

    If a file handle `f` is given (e.g. from `open(path, "w")`), every
    progress line is written to that file as well as printed to the
    console, so the run's full text output is saved to disk."""
    if D_grid is None:
        D_grid = np.round(np.linspace(0, 50, 7), 2)
    if alpha_grid is None:
        alpha_grid = np.round(np.linspace(1.5, 5.0, 6), 2)

    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for D in D_grid:
        for alpha in alpha_grid:
            all_subopt, all_idle = [], []
            for _ in range(n_reps):
                subopt, idle = run_cell(N, M, beta, D, alpha, switch_frac, rng)
                all_subopt.append(subopt)
                all_idle.append(idle)
            all_subopt = np.concatenate(all_subopt)
            all_idle = np.concatenate(all_idle)
            q1, med, q3 = np.percentile(all_subopt, [25, 50, 75])
            rows.append(dict(D=D, alpha=alpha, subopt_q1=q1, subopt_median=med,
                              subopt_q3=q3, pct_idle=100.0 * all_idle.mean()))
            if verbose:
                _emit(f"  D={D:6.2f} alpha={alpha:4.2f}  "
                      f"subopt%[{q1:.3f},{med:.3f},{q3:.3f}]  idle%={100*all_idle.mean():.2f}"
                      f"   (elapsed {time.time()-t0:.0f}s)", f)
    return pd.DataFrame(rows)


def pretty_print(df, D_grid, alpha_grid, title="", f=None):
    """Prints the (D, alpha) pivot tables to the console and, if a file
    handle `f` is given, writes the identical text to that file too."""
    if title:
        _emit("=" * 78, f)
        _emit(title, f)
        _emit("=" * 78, f)
    for col in ["subopt_q1", "subopt_median", "subopt_q3", "pct_idle"]:
        _emit(f"\n-- {col} --", f)
        pivot = df.pivot(index="D", columns="alpha", values=col)
        pivot = pivot.reindex(index=D_grid, columns=alpha_grid)
        _emit(pivot.round(3).to_string(), f)


if __name__ == "__main__":
    np.random.seed(42)
    N = 7          # 8 damage states per machine (0..6 non-terminal, 7 terminal)
    M = 3          # R=1 repairman, M=3 machines
    beta = 0.95    # discount rate
    n_reps = 1    # 10 random problems per (D, alpha) cell, as in Glazebrook

    D_grid = [10]
    alpha_grid = [1.5]

    #D_grid = [0,10,20,25,30,40,50]
    #alpha_grid = [1.5, 2.0, 3.0, 4.0, 5.0]

    #D_grid = np.round(np.linspace(0, 50, 7), 2)
    #alpha_grid = np.round(np.linspace(1.5, 5.0, 6), 2)

    # C_switch = switch_frac * c_bar(machine). switch_frac must be < 1 to stay
    # inside the theorem's proven-valid domain; 0.0 recovers Glazebrook's own
    # no-switching-cost Table 3 as a sanity check.

    switch_fracs = [0.0]
    #switch_fracs = [0.0, 0.5, 0.9, 1.5]

    ####
    W_theorem = theorem_indices(sample_glazebrook_p(N, np.random.default_rng(42)), beta, 500.0, 5.0, 90.0)
    print("\nTheorem-based Whittle indices (discounted, beta = {}):".format(beta))
    print(f"  Active state (0,1): W = {W_theorem[0]:.6f}")
    for k in range(N + 1):
        print(f"  Passive state ({k},0): W = {W_theorem[1 + k]:.6f}")
    ####
    

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_filename = f"Glazebrook_JointOPT_Results_{now_str}.txt"

    with open(out_filename, "w") as f:
        _emit("=" * 78, f)
        _emit("INDEX POLICY vs TRUE JOINT OPT", f)
        _emit("(R=1 repairman, M=3 machines, 8 damage states/machine, switching cost)", f)
        _emit(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        _emit("=" * 78, f)
        _emit(f"N (non-terminal damage states) = {N}   M (machines) = {M}   beta = {beta}", f)
        _emit(f"n_reps per (D, alpha) cell = {n_reps}", f)
        _emit(f"D_grid = {list(D_grid)}", f)
        _emit(f"alpha_grid = {list(alpha_grid)}", f)
        _emit(f"switch_fracs (C_switch = switch_frac * c_bar per machine) = {switch_fracs}", f)
        _emit("\n" + "#" * 78 + "\n", f)

        for switch_frac in switch_fracs:
            df = run_glazebrook_table(N=N, M=M, beta=beta, switch_frac=switch_frac,
                                       n_reps=n_reps, D_grid=D_grid, alpha_grid=alpha_grid,
                                       seed=42, verbose=True, f=f)
            pretty_print(df, D_grid, alpha_grid,
                         title=f"switch_frac = {switch_frac}  (C_switch = {switch_frac} * c_bar per machine)",
                         f=f)
            _emit("", f)

    print(f"\nAll experiments completed. Results saved to '{out_filename}'.")
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
                     three are negative.
    3. the % cost suboptimality 100*(V_INDEX - V_OPT)/V_OPT at every one of
       the 729 states (this is Glazebrook's own suboptimality definition,
       just computed in reward-as-negative-cost sign convention).

Exactly as in Glazebrook Table 3/4, each (D, alpha) table entry summarises
10 problems x 729 initial states = 7290 suboptimality values as a
(lower quartile, median, upper quartile) triple, plus the % of those 7290
problem states in which the index policy chooses to idle.

IMPORTANT -- the switching cost domain restriction: the Theorem's closed form
indices are only valid for
    0 <= C_switch < c_bar := K*(p0-p1) / (1 + beta*(p0-p1))
c_bar depends on that machine's own p0, p1, K, and beta -- it is NOT simply
proportional to C or K. This script computes c_bar per machine and draws
C_switch = switch_frac * c_bar (switch_frac in [0,1)) so every instance is
guaranteed inside the theorem's validity region. If you want to probe what
happens once C_switch >= c_bar, set switch_frac >= 1 -- but note the closed
form formulas are not proven valid there: a different Whittle index would need 
to be substituted (e.g. computed via bisection/PI on the local MDP) to still 
get meaningful suboptimality numbers in that regime.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import combinations

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
# 2. Closed-form Whittle indices, exactly per Theorem "Indexability under
#    perfect repair" (W00 / W01 / Wk0 equations)
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
# 3. Joint (R=1 repairman, M machines) MDP via Kronecker structure, and
#    the true joint DP solver -- this is "expensive but
#    possible" full value iteration over all machine-damage combinations.
# =====================================================================
def enumerate_actions(M, R):
    """The feasible joint actions for R repairmen and M machines: every
    subset of machine indices of size 0..min(R,M) to service simultaneously
    (each serviced machine occupies exactly one repairman; repairmen are
    interchangeable). At R=1 this reduces EXACTLY to Glazebrook's own
    {idle, service machine 1, ..., service machine M}. At R>1 this is a larger set of combinations, 
    but still far fewer than the 2^M or 2*M naive combinations of independent 
    per-machine active/passive choices.

    Returns:
      action_subsets   : list of tuples, action_subsets[a] = machine
                          indices serviced under action a (empty tuple = idle)
      bitmask_to_action : dict, bitmask of a subset -> its action index a
    """
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


def build_joint_mdp(machine_params, beta, R=1):
    """machine_params: list of dicts with keys N, p, C, K, C_switch (one per
    machine). All machines must share the same local state count (same N).
    R: number of repairmen (R>1 allows up to R machines to be serviced 
    simultaneously each period).

    Returns:
      P_joint : list of (L^M, L^M) matrices, one per feasible action
      R_joint : list of (L^M,) reward vectors, same indexing
      locals_ : per-machine dict with P0,P1,R0,R1,labels,W (Whittle index)
      L       : local state count per machine (N+2)
      action_subsets, bitmask_to_action : action-space bookkeeping, needed
                by index_policy/greedy_policy to translate a chosen set of
                machines into the matching action index.

    NOTE ON WHY THIS ISN'T 2^M OR 2*M ACTIONS: with R repairmen you cannot
    simply combine per-machine active/passive choices independently (that
    would allow more than R machines active at once, which needs more than
    R repairmen). The physically realizable joint actions are exactly "pick
    a subset of size <=R of the M machines to service"; passive is never
    chosen for its own sake, it's just what happens to whichever machines
    aren't in that subset. Hence sum_{k=0}^{min(R,M)} C(M,k) actions -- 4
    for R=1,M=3 (Glazebrook Table 3), 11 for R=2,M=4 (Glazebrook Table 4).
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

    action_subsets, bitmask_to_action = enumerate_actions(M, R)
    P_joint, R_joint = [], []
    for combo in action_subsets:
        combo_set = set(combo)
        mats = [locals_[m]["P1"] if m in combo_set else locals_[m]["P0"] for m in range(M)]
        Pa = mats[0]
        for mat in mats[1:]:
            Pa = np.kron(Pa, mat)
        P_joint.append(Pa)

        shape = (L,) * M
        Rgrid = np.zeros(shape)
        for m in range(M):
            r_m = locals_[m]["R1"] if m in combo_set else locals_[m]["R0"]
            bshape = [1] * M
            bshape[m] = L
            Rgrid = Rgrid + r_m.reshape(bshape)
        R_joint.append(Rgrid.reshape(-1))
    return P_joint, R_joint, locals_, L, action_subsets, bitmask_to_action


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


def _decode_local_indices(s, L, M, strides):
    rem = s
    local_idx = []
    for m in range(M):
        local_idx.append(rem // strides[m])
        rem %= strides[m]
    return local_idx


def index_policy(locals_, L, M, R, bitmask_to_action):
    """Rule to R repairmen: at every joint state,
    service the top min(R, #machines with a positive Whittle index)
    machines by index value, each by one repairman; idle any repairman
    left over once no positive-index machines remain. At R=1 this is
    exactly Glazebrook's own "highest index, unless all negative -> idle."

    Not fully vectorized (loops over the L^M joint states with a small
    Python sort each), but this only runs once per problem instance, not
    inside the value-iteration loop, so it's cheap at the M<=4-ish scale
    this script targets."""
    S = L ** M
    W_per_machine = np.stack([locals_[m]["W"] for m in range(M)], axis=0)  # (M, L)
    strides = [L ** (M - 1 - m) for m in range(M)]
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        local_idx = _decode_local_indices(s, L, M, strides)
        w_vals = [(W_per_machine[m, local_idx[m]], m) for m in range(M)]
        candidates = [(w, m) for w, m in w_vals if w > 0]
        candidates.sort(key=lambda x: -x[0])
        chosen = sorted(m for _, m in candidates[:min(R, len(candidates))])
        mask = 0
        for m in chosen:
            mask |= (1 << m)
        pi[s] = bitmask_to_action[mask]
    return pi


def damage_level(L, N):
    """Map each local state index (0..L-1) to its damage level, independent
    of the (x,1)/(x,0) flag. Local state layout (see build_extended_mdp) is
    [ (0,1), (0,0), (1,0), (2,0), ..., (N,0) ], so:
        level(0) = 0   -- (0,1)
        level(1) = 0   -- (0,0)
        level(s) = s-1 -- (s-1, 0) for s >= 2
    """
    levels = np.zeros(L, dtype=int)
    levels[0] = 0
    levels[1] = 0
    for s in range(2, L):
        levels[s] = s - 1
    return levels


def greedy_policy(L, M, N, R, bitmask_to_action):
    """Generalizes greedy to R repairmen: always service the top min(R, M)
    machines by CURRENT damage level, ignoring cost, indices, and the
    (x,1)/(x,0) flag entirely -- "attend the machine(s) which have higher
    damage state, do not worry about anything else". It never idles: there
    is no "nothing to gain" branch here, it always attends to whoever is
    currently worst off (up to R of them), even when several machines are
    tied at level 0.

    Worked example at R=1 (matches the one used to specify this policy):
    state {(3,0), (5,0), (0,1)} -> damage levels {3, 5, 0} -> machine index
    1 (i.e. "machine 2") has the strictly highest level, so it's serviced.
    Verified against this exact case, and against the old R=1-only
    implementation, in testing.

    TIE-BREAKING. When two or more machines share a damage level at the
    R-th cutoff (e.g. {(0,1),(0,1),(0,1)}, a 3-way tie at level 0 with
    R=1), this implementation picks the LOWEST-INDEXED tied machine(s)
    first. Two things worth noting about this:

      1. A fully-tied state like {(0,1),(0,1),(0,1)} can never actually be
         *reached* by the joint dynamics from any real trajectory: the
         active-action transition always sends only the SERVICED machines
         to local state 0 = "(0,1)"; since at most R machines are serviced
         per period, at most R machines can carry that flag at the start
         of the next period. So a state where MORE than R machines carry
         it (like this example under R=1) only ever shows up as one of the
         L^M *hypothetical* initial states Glazebrook's own methodology
         evaluates (every combination is used as an initial state for the
         DP, not just reachable ones) -- it isn't something you'd see
         mid-trajectory.
      2. Given that, the exact tie-break rule has very little influence on
         the aggregate suboptimality statistics (a tied state is just one
         of L^M initial states, weighted the same as any other), but it
         does matter for reproducibility of exactly which tied machines get
         serviced. Lowest-index-first is the simplest deterministic choice
         used here. Two other reasonable alternatives, if you'd rather use
         them: (a) prefer whichever tied machines are already flagged
         (x,1), since continuing avoids a fresh switching cost; (b) break
         ties uniformly at random and average over several draws so the
         arbitrary choice doesn't bias a single run's numbers.
    """
    levels = damage_level(L, N)
    S = L ** M
    strides = [L ** (M - 1 - m) for m in range(M)]
    R_eff = min(R, M)
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        local_idx = _decode_local_indices(s, L, M, strides)
        lvl_vals = [(levels[local_idx[m]], m) for m in range(M)]
        lvl_vals.sort(key=lambda x: (-x[0], x[1]))   # ties -> lowest machine index
        chosen = sorted(m for _, m in lvl_vals[:R_eff])
        mask = 0
        for m in chosen:
            mask |= (1 << m)
        pi[s] = bitmask_to_action[mask]
    return pi


def fraction_tied_states(L, M, N, R):
    """Diagnostic: what fraction of the L^M joint states have a damage-level
    tie AT the R-th cutoff (i.e. the greedy rule's choice of which machines
    to service is ambiguous without a tie-break rule)?"""
    levels = damage_level(L, N)
    shape = (L,) * M
    lvl_grids = []
    for m in range(M):
        bshape = [1] * M
        bshape[m] = L
        lvl_grids.append(levels.reshape(bshape) * np.ones(shape, dtype=int))
    lvl_stack = np.stack(lvl_grids, axis=0).reshape(M, -1)  # (M, L^M)
    R_eff = min(R, M)
    sorted_desc = -np.sort(-lvl_stack, axis=0)  # each column sorted descending
    if R_eff < M:
        at_cutoff = sorted_desc[R_eff - 1, :]
        just_past = sorted_desc[R_eff, :]
        is_tie = at_cutoff == just_past
    else:
        is_tie = np.zeros(lvl_stack.shape[1], dtype=bool)  # R>=M: no cutoff, everyone serviced
    return is_tie.mean()

def sample_glazebrook_p(N, rng):
    raw = rng.uniform(0.0, 1.0, N)
    return np.sort(raw)[::-1]


def run_cell(N, M, R, beta, D, alpha, switch_frac, rng):
    """One random R-repairman, M-machine problem: draw p (ordered sample), 
    C ~ U(D, D+25), K = alpha*C per machine, and C_switch =
    switch_frac * c_bar (per-machine c_bar, so every instance stays inside
    the theorem's validity domain when switch_frac < 1). Returns the index
    policy's suboptimality (%), the greedy policy's suboptimality (%), and
    the index policy's idle-flag array, each over all L^M initial states.
    R=1 reproduces Glazebrook's own Table 3 setting; R=2 (with M machines)
    is Glazebrook's Table 4 setting, generalized here to include the
    switching cost."""
    machine_params = []
    for _ in range(M):
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + 25)
        K = alpha * C
        cbar = c_bar(K, p[0], p[1], beta)
        C_switch = switch_frac * cbar
        machine_params.append(dict(N=N, p=p, C=C, K=K, C_switch=C_switch))

    P_joint, R_joint, locals_, L, action_subsets, bitmask_to_action = build_joint_mdp(
        machine_params, beta, R=R)
    print(f"  D={D:6.2f} alpha={alpha:4.2f}  C_switch={C_switch:.3f}  c_bar={cbar:.3f}")
    print(f"    Joint state space size: {L**M} = {L}^{M}")
    print(f"    Joint action space size: {M+1} = {M}+1 (idle + service each machine)")
    print(f"    Joint transition matrices: {[P.shape for P in P_joint]}")
    print(f"    Joint reward vectors: {[R.shape for R in R_joint]}")
    V_opt, pi_star, _ = solve_joint_vi(P_joint, R_joint, beta)

    pi_idx = index_policy(locals_, L, M, R, bitmask_to_action)
    V_idx = evaluate_joint_policy(pi_idx, P_joint, R_joint, beta)

    pi_greedy = greedy_policy(L, M, N, R, bitmask_to_action)
    V_greedy = evaluate_joint_policy(pi_greedy, P_joint, R_joint, beta)
    print(f"    Optimal policy (pi_star) = {pi_star}")
    print(f"    Index policy (pi_idx) = {pi_idx}")
    print(f"    Greedy policy (pi_greedy) = {pi_greedy}")
    print(f"    V_opt = {V_opt}")
    print(f"    V_idx = {V_idx}")
    print(f"    V_greedy = {V_greedy}")
    print(f"sum of V_opt = {V_opt.sum():.3f}, sum of V_idx = {V_idx.sum():.3f}, sum of V_greedy = {V_greedy.sum():.3f}")

    # Sign note: R0/R1 are negative costs, so V = -discounted cost, and
    # V_OPT >= V_idx/V_greedy (OPT has the higher/less-negative reward =
    # lower cost). Glazebrook's suboptimality = 100*(Cost_x - Cost_opt)/Cost_opt
    #   = 100*((-V_x) - (-V_opt)) / (-V_opt) = 100*(V_x - V_opt)/V_opt
    # which is >= 0 given the sign conventions above -- verified in testing.
    subopt_idx_pct = 100.0 * (V_idx - V_opt) / V_opt
    subopt_greedy_pct = 100.0 * (V_greedy - V_opt) / V_opt
    idle_flags = (pi_idx == 0).astype(float)   # action 0 is always the empty/idle subset
    return subopt_idx_pct, subopt_greedy_pct, idle_flags


def _emit(msg, f=None):
    """Print to console AND, if a file handle was given, write the same
    line to the results text file."""
    print(msg)
    if f is not None:
        f.write(msg + "\n")


def run_glazebrook_table(N=7, M=3, R=1, beta=0.95, switch_frac=0.5, n_reps=10,
                          D_grid=None, alpha_grid=None, seed=42, verbose=True,
                          f=None):
    """Reproduces one Glazebrook Table-3/4-style 7x6 grid of four-vector
    entries (here: quartile triple + % idle), for the switching-cost
    extension, for R repairmen and M machines (R=1 -> Table 3 setting,
    R=2 -> Table 4 setting). Returns a tidy DataFrame, one row per
    (D, alpha) cell.

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
            all_subopt_idx, all_subopt_greedy, all_idle = [], [], []
            for _ in range(n_reps):
                subopt_idx, subopt_greedy, idle = run_cell(N, M, R, beta, D, alpha, switch_frac, rng)
                all_subopt_idx.append(subopt_idx)
                all_subopt_greedy.append(subopt_greedy)
                all_idle.append(idle)
            all_subopt_idx = np.concatenate(all_subopt_idx)
            all_subopt_greedy = np.concatenate(all_subopt_greedy)
            all_idle = np.concatenate(all_idle)
            q1, med, q3 = np.percentile(all_subopt_idx, [25, 50, 75])
            gq1, gmed, gq3 = np.percentile(all_subopt_greedy, [25, 50, 75])
            rows.append(dict(D=D, alpha=alpha,
                              subopt_q1=q1, subopt_median=med, subopt_q3=q3,
                              greedy_subopt_q1=gq1, greedy_subopt_median=gmed, greedy_subopt_q3=gq3,
                              pct_idle=100.0 * all_idle.mean()))
            if verbose:
                _emit(f"  D={D:6.2f} alpha={alpha:4.2f}  "
                      f"index%[{q1:.3f},{med:.3f},{q3:.3f}]  "
                      f"greedy%[{gq1:.3f},{gmed:.3f},{gq3:.3f}]  idle%={100*all_idle.mean():.2f}"
                      f"   (elapsed {time.time()-t0:.0f}s)", f)
    return pd.DataFrame(rows)


def pretty_print(df, D_grid, alpha_grid, title="", f=None):
    """Prints the (D, alpha) pivot tables to the console and, if a file
    handle `f` is given, writes the identical text to that file too."""
    if title:
        _emit("=" * 78, f)
        _emit(title, f)
        _emit("=" * 78, f)
    cols = ["subopt_q1", "subopt_median", "subopt_q3",
            "greedy_subopt_q1", "greedy_subopt_median", "greedy_subopt_q3",
            "pct_idle"]
    for col in cols:
        _emit(f"\n-- {col} --", f)
        pivot = df.pivot(index="D", columns="alpha", values=col)
        pivot = pivot.reindex(index=D_grid, columns=alpha_grid)
        _emit(pivot.round(3).to_string(), f)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # BLOCK 1: R=1 repairman, M=3 machines, extended with the switching cost
    # ------------------------------------------------------------------
    N = 5          # 8 damage states per machine (0..6 non-terminal, 7 terminal)
    M = 3          # M=3 machines
    R = 1          # R=1 repairman
    beta = 0.95    # discount rate 
    n_reps = 1    # 10 random problems per (D, alpha) cell


    D_grid = [10]
    alpha_grid = [1.5]

    # D_grid = [0,10,20,25,30,40,50]
    # alpha_grid = [1.5, 2.0, 3.0, 4.0, 5.0]

    # D_grid = np.round(np.linspace(0, 50, 7), 2)
    # alpha_grid = np.round(np.linspace(1.5, 5.0, 6), 2)

    # C_switch = switch_frac * c_bar(machine). switch_frac must be < 1 to stay
    # inside the theorem's proven-valid domain; 0.0 recovers Glazebrook's own
    # no-switching-cost Table 3 as a sanity check.
    switch_fracs = [1] # 1.5
    #switch_fracs = [0.0, 0.5, 0.9]

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_filename = f"Glazebrook_Joint_OPT_Results(Full)_{now_str}.txt"

    with open(out_filename, "w") as f:
        _emit("=" * 78, f)
        _emit("BLOCK 1: R=1 repairman, M=3 machines (Glazebrook Table 3 setting)", f)
        _emit("INDEX POLICY vs GREEDY POLICY vs TRUE JOINT OPT", f)
        _emit(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        _emit("=" * 78, f)
        _emit(f"N (non-terminal damage states) = {N}   M (machines) = {M}   R (repairmen) = {R}   beta = {beta}", f)
        _emit(f"n_reps per (D, alpha) cell = {n_reps}", f)
        _emit(f"D_grid = {list(D_grid)}", f)
        _emit(f"alpha_grid = {list(alpha_grid)}", f)
        _emit(f"switch_fracs (C_switch = switch_frac * c_bar per machine) = {switch_fracs}", f)
        _emit("\n" + "#" * 78 + "\n", f)

        for switch_frac in switch_fracs:
            df = run_glazebrook_table(N=N, M=M, R=R, beta=beta, switch_frac=switch_frac,
                                       n_reps=n_reps, D_grid=D_grid, alpha_grid=alpha_grid,
                                       seed=42, verbose=True, f=f)
            pretty_print(df, D_grid, alpha_grid,
                         title=f"switch_frac = {switch_frac}  (C_switch = {switch_frac} * c_bar per machine)",
                         f=f)
            _emit("", f)

        # --------------------------------------------------------------
        # BLOCK 2: R=2 repairmen, M=4 machines Joint state space is L^M 
        # and grows fast (M=4, R=2 means 11 action (4C2 + 4C1 + 4C0) matrices 
        # of size L^4 x L^4 each get built/solved per instance), 
        # so this block deliberately uses a SMALLER N and a
        # smaller (D, alpha) grid than Block 1, purely to keep runtime and
        # memory sane as a demonstration. Scale N/D_grid/alpha_grid/n_reps
        # up if your machine can afford it.
        # --------------------------------------------------------------
        N2, M2, R2 = 7, 3, 2      # L = N2+2 = 9 -> 9^3 = 729 joint states
        n_reps2 = 1

        D_grid2 = [10]
        alpha_grid2 = [1.5]

        # D_grid = [0,10,20,25,30,40,50]
        # alpha_grid = [1.5, 2.0, 3.0, 4.0, 5.0]

        # D_grid = np.round(np.linspace(0, 50, 7), 2)
        # alpha_grid = np.round(np.linspace(1.5, 5.0, 6), 2)
        # D_grid2 = np.round(np.linspace(0, 50, 3), 2)
        # alpha_grid2 = np.round(np.linspace(1.5, 5.0, 3), 2)
        switch_fracs2 = [0.0, 0.5]

        _emit("\n\n" + "=" * 78, f)
        _emit("BLOCK 2: R=2 repairmen, M=4 machines (Glazebrook Table 4 setting)", f)
        _emit("(smaller-scale demo of the general-R machinery -- see comments in __main__)", f)
        _emit("=" * 78, f)
        _emit(f"N (non-terminal damage states) = {N2}   M (machines) = {M2}   R (repairmen) = {R2}   beta = {beta}", f)
        _emit(f"n_reps per (D, alpha) cell = {n_reps2}", f)
        _emit(f"D_grid = {list(D_grid2)}", f)
        _emit(f"alpha_grid = {list(alpha_grid2)}", f)
        _emit(f"switch_fracs = {switch_fracs2}", f)
        _emit("\n" + "#" * 78 + "\n", f)

        for switch_frac in switch_fracs2:
            df2 = run_glazebrook_table(N=N2, M=M2, R=R2, beta=beta, switch_frac=switch_frac,
                                        n_reps=n_reps2, D_grid=D_grid2, alpha_grid=alpha_grid2,
                                        seed=42, verbose=True, f=f)
            pretty_print(df2, D_grid2, alpha_grid2,
                         title=f"[R=2,M=4] switch_frac = {switch_frac}",
                         f=f)
            _emit("", f)

    print(f"\nAll experiments completed. Results saved to '{out_filename}'.")

import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import combinations
import sys
sys.path.insert(0, '.')


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
#  Joint (R=1 repairman, M machines) MDP via Kronecker structure, and
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

def build_joint_mdp_hetero(machine_params, beta, R=1):
    """Like build_joint_mdp, but machines may have DIFFERENT local state
    counts (different N_m -> different L_m = N_m+2). No assumption that
    all machines share L. The Kronecker-product construction never actually
    needed same-size factors -- np.kron handles differently-sized matrices
    natively -- so the only real fix needed is: (a) drop the equal-L
    assertion, (b) use each machine's own L_m when broadcasting the reward
    grid, (c) use each machine's own L_m when decoding a flat joint state
    index back into per-machine local indices (np.unravel_index handles
    this generally already)."""
    M = len(machine_params)
    locals_ = []
    for mp in machine_params:
        P0, P1, R0, R1, labels = build_extended_mdp(
            mp["N"], np.append(mp["p"], 0.0), mp["C"], mp["K"], mp["C_switch"])
        W = theorem_indices(mp["p"], beta, mp["K"], mp["C"], mp["C_switch"])
        locals_.append(dict(P0=P0, P1=P1, R0=R0, R1=R1, labels=labels, W=W, N=mp["N"]))

    L_list = [locals_[m]["P0"].shape[0] for m in range(M)]   # <-- per-machine

    action_subsets, bitmask_to_action = enumerate_actions(M, R)
    P_joint, R_joint = [], []
    for combo in action_subsets:
        combo_set = set(combo)
        mats = [locals_[m]["P1"] if m in combo_set else locals_[m]["P0"] for m in range(M)]
        Pa = mats[0]
        for mat in mats[1:]:
            Pa = np.kron(Pa, mat)   # kron handles differently-sized factors fine
        P_joint.append(Pa)

        shape = tuple(L_list)       # <-- was (L,)*M, now per-axis L_m
        Rgrid = np.zeros(shape)
        for m in range(M):
            r_m = locals_[m]["R1"] if m in combo_set else locals_[m]["R0"]
            bshape = [1] * M
            bshape[m] = L_list[m]
            Rgrid = Rgrid + r_m.reshape(bshape)
        R_joint.append(Rgrid.reshape(-1))
    return P_joint, R_joint, locals_, L_list, action_subsets, bitmask_to_action




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



def index_policy_hetero(locals_, L_list, M, R, bitmask_to_action):
    S = int(np.prod(L_list))
    shape = tuple(L_list)
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        local_idx = np.unravel_index(s, shape)   # <-- handles per-axis L_m generally
        w_vals = [(locals_[m]["W"][local_idx[m]], m) for m in range(M)]
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
    levels[0] = 0; 
    levels[1] = 0
    for s in range(2, L):
        levels[s] = s - 1
    return levels

def greedy_policy_hetero(L_list, M, N_list, R, bitmask_to_action):
    S = int(np.prod(L_list))
    shape = tuple(L_list)
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

def sample_glazebrook_p(N, rng):
    raw = rng.uniform(0.0, 1.0, N)
    return np.sort(raw)[::-1]

# =============== TEST 1: regression -- homogeneous N must match old results ===============
np.random.seed(31)
N, M, beta, R = 5, 3, 0.95, 1
D, alpha, switch_frac = 20.0, 3.0, 0.5
rng = np.random.default_rng(13)
machine_params = []
for _ in range(M):
    p = sample_glazebrook_p(N, rng)
    C = rng.uniform(D, D+25); K = alpha*C
    cbar_ = c_bar(K, p[0], p[1], beta)
    machine_params.append(dict(N=N, p=p, C=C, K=K, C_switch=switch_frac*cbar_))


P_old, R_old, loc_old, L_old, as_old, b2a_old = build_joint_mdp(machine_params, beta, R=R)
V_opt_old, _, _ = solve_joint_vi(P_old, R_old, beta)
pi_idx_old = index_policy(loc_old, L_old, M, R, b2a_old)
pi_greedy_old = greedy_policy(L_old, M, N, R, b2a_old)

P_new, R_new, loc_new, L_list_new, as_new, b2a_new = build_joint_mdp_hetero(machine_params, beta, R=R)
V_opt_new, _, _ = solve_joint_vi(P_new, R_new, beta)
pi_idx_new = index_policy_hetero(loc_new, L_list_new, M, R, b2a_new)
pi_greedy_new = greedy_policy_hetero(L_list_new, M, [N]*M, R, b2a_new)

print("REGRESSION (homogeneous N via hetero code path):")
print("  L_list:", L_list_new, " (expect all equal, matches old scalar L =", L_old, ")")
print("  V_opt matches old:", np.allclose(V_opt_old, V_opt_new))
print("  index policy matches old:", np.array_equal(pi_idx_old, pi_idx_new))
print("  greedy policy matches old:", np.array_equal(pi_greedy_old, pi_greedy_new))

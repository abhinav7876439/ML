"""
Heterogeneous multi-machine (R repairmen) repairman problem.

Computes the Whittle index FOUR different ways per machine (Closed Form,
VI, PI, Package) (plus a Theorem based closed form), builds the resulting
FIVE index heuristic joint policies, plus a Greedy policy, evaluates all
SIX against the true joint OPT (via full value iteration on the joint
MDP), and reports suboptimality -- reproducing the structure of
Glazebrook's Table 3 (R= **, M= ** , ** damage states/machine, D in
{0,10,20,25,30,40,50}, alpha in {1.5,2.0,3.0,4.0,5.0,"Various"}, 10
randomly drawn problem instances per (D,alpha) cell, each instance
evaluated over all L^M joint states, everything POOLED across the 10
instances before quantiles are taken).
"""


import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False


# =====================================================================
# Single machine local MDP
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
            P0[s, 2] = p[0]; 
            P0[s, 1] = 1 - p[0]; 
            R0[s] = -(K * (1 - p[0]))
        elif s == 1:
            P0[s, 2] = p[0]; 
            P0[s, 1] = 1 - p[0]; 
            R0[s] = -(K * (1 - p[0]))
        else:
            idx = s - 1
            if idx < N:
                P0[s, s + 1] = p[idx]; 
                P0[s, 1] = 1 - p[idx]; 
                R0[s] = -(K * (1 - p[idx]))
            else:
                P0[s, 1] = 1.0; 
                R0[s] = -K
    return P0, P1, R0, R1, state_labels


# =====================================================================
# Discounted solvers (VI, PI) -> single machine index by bisection
# =====================================================================
def solve_vi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=10000):
    S = len(R0); 
    V = np.zeros(S)
    for _ in range(max_iter):
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        V_new = np.maximum(Q0, Q1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    Q0 = R0 + beta * (P0 @ V); 
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1


def policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta):
    S = len(pi)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])
    return np.linalg.solve(np.eye(S) - beta * P_pi, r_pi)


def solve_pi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=1000):
    S = len(R0); 
    pi = np.zeros(S, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta)
        Q0 = R0 + beta * (P0 @ V); 
        Q1 = (R1 - w) + beta * (P1 @ V)
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + beta * (P0 @ V); 
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1


def whittle_index_discounted(state, solver, beta, P0, P1, R0, R1,
                              w_min=-2000.0, w_max=2000.0, tol_w=1e-6, tol_mdp=1e-9, Verbose=False):
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
# Closed-form (two equivalent derivations) + validity bound
# =====================================================================
def compute_G_H(p, beta):
    N = len(p); 
    H = np.zeros(N + 1); 
    G = np.zeros(N + 2); 
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
    """Returns W as a length-(N+2) array in the same state order as
    build_extended_mdp: [W(0,1), W(0,0), W(1,0), ..., W(N,0)]."""
    N = len(p); 
    G, H = compute_G_H(p, beta)
    p0 = p[0] if N > 0 else 0.0
    w00 = (1 - p0) * K - C - C_switch * (1 - beta)
    if N >= 1:
        G1, H1 = G[1], H[1]
        num = (1 - beta) * (K * G1 + C_switch * H1)
        den = beta * (1 - G1) - H1
        w_active = (num / den) - C if abs(den) > 1e-15 else np.inf
    else:
        w_active = np.inf
    w_passive = np.zeros(N + 1); 
    w_passive[0] = w00
    for k in range(1, N + 1):
        pk = p[k] if k < N else 0.0
        Gk, Hk = G[k], H[k]
        denom = (1 - pk * beta) * (1 - Gk) - (1 - pk) * Hk
        w_passive[k] = (K - C - C_switch - (K * pk * (1 - beta)) / denom
                        if abs(denom) > 1e-15 else np.inf)
    return np.concatenate(([w_active], w_passive))


def theorem_indices(p, beta, K, C, C_switch):
    """Returns W as a length-(N+2) array in the same state order as
    build_extended_mdp: [W(0,1), W(0,0), W(1,0), ..., W(N,0)]."""
    N = len(p); 
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
    W = np.zeros(N + 2); 
    W[0] = W01; 
    W[1] = W00
    for k in range(1, N + 1):
        W[1 + k] = Wk0[k]
    return W


def c_bar(K, p0, p1, beta):
    """Theorem's validity bound: closed form is only proven valid for
    0 <= C_switch < c_bar."""
    return K * (p0 - p1) / (1 + beta * (p0 - p1))


# =====================================================================
# Compute all index arrays (different methods) for one machine
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
#  Heterogeneous joint MDP (via Kronecker structure) + policies
# =====================================================================
def enumerate_actions(M, R):
    """The feasible joint actions for R repairmen and M machines: every
    subset of machine indices of size 0..min(R,M) to service simultaneously
    (each serviced machine occupies exactly one repairman; repairmen are
    interchangeable). At R=1 this reduces EXACTLY to Glazebrook's own
    {idle, service machine 1, ..., service machine M}. At R>1 this is a larger 
    set of combinations, but still far fewer than the 2^M or 2*M naive combinations 
    of independent per machine active/passive choices.

    
    Returns:
    action_subsets[a] = machine indices serviced by joint action a;
    bitmask_to_action maps a bitmask of serviced machines -> action index.
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


def build_joint_mdp_hetero(machine_locals, beta, R):
    """Kronecker structured joint MDP for M heterogeneous machines, each with its 
    own local MDP (P0,P1,R0,R1) and possibly different number of damage states N. 
    The joint state space is the Cartesian product of the local state spaces, and 
    the joint action space is all feasible combinations of servicing up to R 
    machines simultaneously. The joint MDP has a Kronecker structure, allowing 
    efficient construction of the joint transition matrices and reward vectors. The
    joint transition matrix for a given joint action is the Kronecker product of 
    the local transition matrices for each machine, where the local transition 
    matrix is P1 if the machine is serviced and P0 if it is not. The joint reward 
    vector for a given joint action is the sum of the local reward vectors for each 
    machine, where the local reward vector is R1 if the machine is serviced and R0
    if it is not. The function returns the joint transition matrices, joint reward 
    vectors, local state space sizes, feasible joint actions, and a mapping from 
    serviced machine bitmasks to joint action indices.  

    Returns:
    P_joint : list of transition matrices, one per joint action
    R_joint : list of reward vectors, one per joint action
    L_list : list of local state space sizes
    action_subsets : list of feasible joint actions
    bitmask_to_action : dict mapping serviced machine bitmasks to action indices
    
    
    NOTE ON WHY THIS ISN'T 2^M OR 2*M ACTIONS: with R repairmen we cannot
    simply combine per machine active/passive choices independently (that
    would allow more than R machines active at once, which needs more than
    R repairmen). The physically realizable joint actions are exactly "pick
    a subset of size <=R of the M machines to service"; passive is never
    chosen for its own sake, it's just what happens to whichever machines
    aren't in that subset. Hence sum_{k=0}^{min(R,M)} C(M,k) actions i.e. 4
    for R=1,M=3, 11 for R=2,M=4 and likewise.
    """

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
    """Full value iteration -> V_OPT and the true optimal joint policy."""
    S = len(R_joint[0]); 
    A = len(R_joint); 
    V = np.zeros(S)
    for _ in range(max_iter):
        Qs = np.stack([R_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
        V_new = Qs.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new; break
        V = V_new
    Qs = np.stack([R_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
    pi_star = Qs.argmax(axis=0)
    return V, pi_star, Qs


def evaluate_joint_policy(pi, P_joint, R_joint, beta):
    """Exact value function of a fixed deterministic joint policy pi, via
    linear solve."""
    S = len(R_joint[0]); 
    P_pi = np.zeros((S, S)); 
    r_pi = np.zeros(S)
    for a in range(len(R_joint)):
        mask = (pi == a)
        P_pi[mask, :] = P_joint[a][mask, :]
        r_pi[mask] = R_joint[a][mask]
    return np.linalg.solve(np.eye(S) - beta * P_pi, r_pi)


def index_policy_from_Whittle_indices(W_list, L_list, M, R, bitmask_to_action):
    S = int(np.prod(L_list)); 
    shape = tuple(L_list)
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
    """Map each local state index (0..L-1) to its damage level, independent
    of the (x,1)/(x,0) flag. Local state layout (see build_extended_mdp) is
    [ (0,1), (0,0), (1,0), (2,0), ..., (N,0) ], so:
        level(0) = 0   -- (0,1)
        level(1) = 0   -- (0,0) # (the two (0,*) states)
        level(s) = s-1 -- (s-1, 0) for s >= 2
    """
    levels = np.zeros(L, dtype=int)
    levels[0] = 0; levels[1] = 0
    for s in range(2, L):
        levels[s] = s - 1
    return levels


def greedy_policy_hetero(L_list, M, N_list, R, bitmask_to_action):
    """ Generalizes greedy to R repairmen: always service the top min(R, M)
    machines by CURRENT damage level, ignoring cost, indices and the
    (x,1)/(x,0) flag entirely -- "attend the machine(s) which have higher
    damage state, do not worry about anything else". It never idles: there
    is no "nothing to gain" branch here, it always attends to whoever is
    currently worst off (up to R of them), even when several machines are
    tied at level 0.

    Worked example at R=1 (matches the one used to specify this policy):
    state {(3,0), (5,0), (0,1)} -> damage levels {3, 5, 0} -> machine index
    1 (i.e. "machine 2") has the strictly highest level, so it's serviced.
    Verified against the exact case, and greedy_policy_hetero() implementation, 
    which was correct for R=1.

    TIE-BREAKING. When two or more machines share a damage level at the
    R-th cutoff (e.g. {(0,1),(0,1),(0,1)}, a 3 way tie at level 0 with
    R=1), this implementation picks the LOWEST INDEXED tied machine(s)
    first. Two things worth noting about this:

      1. A fully tied state like {(0,1),(0,1),(0,1)} can never actually be
         *reached* by the joint dynamics from any real trajectory: the
         active action transition always sends only the SERVICED machines
         to local state 0 = "(0,1)"; since at most R machines are serviced
         per period, at most R machines can carry that flag at the start
         of the next period. So a state where MORE than R machines carry
         it (like this example under R=1) only ever shows up as one of the
         L^M *hypothetical* initial states for the aggregate suboptimality 
         statistics, not as a real state visited by the joint dynamics. 
         Hence the tie break rule is only relevant for reproducibility of 
         which machines get serviced in a tied state, not for the actual 
         dynamics of the system.
      2. Given that, the exact tie break rule has very little influence on
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


def fraction_tied_states(L_list, M, N_list, R):
    """Diagnostic: what fraction of the prod(L_list) joint states have a
    damage level tie AT the Rth cutoff (i.e. the greedy rule's choice of
    which machines to service is ambiguous without a tie break rule)?"""

    shape = tuple(L_list)
    levels_per_machine = [damage_level(L_list[m], N_list[m]) for m in range(M)]
    lvl_grids = []
    for m in range(M):
        bshape = [1] * M
        bshape[m] = L_list[m]
        lvl_grids.append(levels_per_machine[m].reshape(bshape) * np.ones(shape, dtype=int))
    lvl_stack = np.stack(lvl_grids, axis=0).reshape(M, -1)
    R_eff = min(R, M)
    sorted_desc = -np.sort(-lvl_stack, axis=0)
    if R_eff < M:
        at_cutoff = sorted_desc[R_eff - 1, :]
        just_past = sorted_desc[R_eff, :]
        is_tie = at_cutoff == just_past
    else:
        is_tie = np.zeros(lvl_stack.shape[1], dtype=bool)
    return is_tie.mean()


def sample_glazebrook_p(N, rng):
    return np.sort(rng.uniform(0.0, 1.0, N))[::-1]


# =====================================================================
# Dual terminal + file output
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
#  Experiment configuration
# =====================================================================
# Paper's "Various" column (Table 3): K1=1.5*C1, K2=3*C2, K3=4.5*C3.
VARIOUS_ALPHA_MULTIPLIERS = [1.5, 3.0, 4.5]

AlphaSpec = Union[float, str, List[float], Tuple[float, ...]]


def resolve_alpha_per_machine(alpha: AlphaSpec, M: int) -> List[float]:
    """alpha can be:
      - a plain number  -> same alpha for every machine (K_m = alpha * C_m)
      - the string "Various" -> paper's fixed per machine multipliers,
        only defined for M=3
      - a list/tuple of length M -> explicit per machine alphas
    """
    if isinstance(alpha, str):
        if alpha != "Various":
            raise ValueError(f"Unknown alpha spec: {alpha!r}")
        if M != 3:
            raise ValueError('"Various" alpha is only defined for M=3 (paper Table 3)')
        return list(VARIOUS_ALPHA_MULTIPLIERS)
    if isinstance(alpha, (list, tuple)):
        if len(alpha) != M:
            raise ValueError(f"alpha list length {len(alpha)} != M={M}")
        return [float(a) for a in alpha]
    return [float(alpha)] * M


SwitchFracSpec = Union[float, List[float], Tuple[float, ...]]


def resolve_switch_frac_per_machine(switch_frac: SwitchFracSpec, M: int) -> List[float]:
    """switch_frac can be:
      - a plain number  -> same switch_frac for every machine (homogeneous;
      - a list/tuple of length M -> explicit per machine switch_frac, e.g.
        [0.3, 1.2, 0.0] lets some machines sit inside their own c_bar domain
        and others sit outside it, within the SAME problem instance (heterogeneous switch_frac). 
        This is not a case the paper explores, but it is allowed here for completeness.
    """
    if isinstance(switch_frac, (list, tuple)):
        if len(switch_frac) != M:
            raise ValueError(f"switch_frac list length {len(switch_frac)} != M={M}")
        return [float(s) for s in switch_frac]
    return [float(switch_frac)] * M


def switch_frac_cell_label(switch_frac):
    """Hashable, sortable, human-readable label for one switch_fracs entry,
    for use as a DataFrame grouping key / table column. Scalars pass
    through unchanged (so homogeneous cells look exactly as before);
    list/tuple entries become a short 'mixed_...' tag."""
    if isinstance(switch_frac, (list, tuple)):
        return "mixed_" + "-".join(f"{float(s):g}" for s in switch_frac)
    return switch_frac

@dataclass
class ExperimentConfig:
    # machines & per machine heterogeneity in number of damage levels N.
    # Set N_list=None to draw each machine's N ~ Uniform N_range. 
    N_list: Optional[List[int]] = field(default_factory=lambda: [6, 6, 6])
    M: int = 3                          # only used when N_list is None
    N_range: Tuple[int, int] = (3, 6)

    R: int = 1                          # number of repairmen
    beta: float = 0.95                  # discount factor

    n_reps: int = 10                    # free input: independent problem
                                         # instances per (D,alpha,switch_frac)
                                         # cell, pooled before quantiles matches 
                                         # the paper results

    D_grid: List[float] = field(default_factory=lambda: [0, 10, 20, 25, 30, 40, 50])
    alpha_grid: List[AlphaSpec] = field(
        default_factory=lambda: [1.5, 2.0, 3.0, 4.0, 5.0, "Various"])
    C_width: float = 25.0

    switch_fracs: List[SwitchFracSpec] = field(default_factory=lambda: [0.0])

    seed0: int = 1000

    verbose_indices: bool = True        # log per machine params + all W arrays
    verbose_policies: bool = True       # log pi_star / each method's pi / V's
    verbose_joint_shapes: bool = True   # log joint action/state/matrix shapes
    dump_full_matrices: bool = False    # DANGEROUS for large S -- see below
    policy_dump_max_states: int = 300 #1000   # skip full pi/V dump above this S

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_txt: str = f"repairman_results_{now_str}.txt"


def resolve_N_list(cfg: ExperimentConfig, rng: np.random.Generator) -> List[int]:
    if cfg.N_list is not None:
        return list(cfg.N_list)
    lo, hi = cfg.N_range
    return [int(rng.integers(lo, hi + 1)) for _ in range(cfg.M)]


METHODS = ["ClosedForm", "VI", "PI", "Package", "Theorem", "Greedy"]
def run_cell(cfg: ExperimentConfig, D, alpha, switch_frac, rng, instance_id, verbose=True):
    """Draws ONE random set of machine parameters (one problem instance),
    builds the joint MDP, computes the Whittle index four ways (+Theorem),
    builds all five resulting index policies plus Greedy, evaluates all six
    against the true joint OPT.

    `switch_frac` may be a single scalar (homogeneous across machines,
    original behavior) or a length-M list/tuple (heterogeneous: some
    machines can be inside their own c_bar domain, others outside it,
    within the same instance).

    `instance_id` is a display only running counter (cfg.seed0 + however
    many draws have happened so far) -- it is NOT used to seed a fresh
    np.random.Generator. Randomness comes entirely from the single `rng`
    object passed in, which the caller (run_glazebrook_table) creates ONCE
    from cfg.seed0 and advances across every draw in the whole sweep.
    `instance_id` exists purely so the transcript can label each instance
    for reference, matching the "seed=NNNN" header line.
    """
    N_list = resolve_N_list(cfg, rng)
    M = len(N_list)
    beta = cfg.beta
    alpha_per_machine = resolve_alpha_per_machine(alpha, M)
    switch_frac_per_machine = resolve_switch_frac_per_machine(switch_frac, M)

    machine_params = []
    for N, alpha_m, sf_m in zip(N_list, alpha_per_machine, switch_frac_per_machine):
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + cfg.C_width)
        K = alpha_m * C
        cbar_ = c_bar(K, p[0], p[1], beta)   # per-machine c_bar (depends on K, p, beta)
        # C_switch = switch_frac * cbar_    # same scalar multiplier for every machine (current homogeneous-switch_frac design is artificial case)

                # the ratio C_switch / c_bar is forced to be identical across all M machines in a 
                # given instance, because it's the one switch_frac scalar for that cell. 
                # So within any single instance, every machine is simultaneously inside the 
                # domain (switch_frac < 1) or simultaneously outside it (switch_frac ≥ 1)

        
        C_switch = sf_m * cbar_              # per-machine multiplier, machines can
                                              # independently be inside (sf_m < 1) or
                                              # outside (sf_m >= 1) their own domain
        machine_params.append(dict(N=N, p=p, C=C, K=K, C_switch=C_switch,
                                    c_bar=cbar_, alpha_m=alpha_m, switch_frac_m=sf_m))

    machine_locals = []
    method_W = {}
    for m, mp in enumerate(machine_params):
        methods, _ = all_method_indices(mp, beta)
        method_W[m] = methods
        P0, P1, R0, R1, _ = build_extended_mdp(mp["N"], np.append(mp["p"], 0.0),
                                                mp["C"], mp["K"], mp["C_switch"])
        machine_locals.append(dict(P0=P0, P1=P1, R0=R0, R1=R1))

    P_joint, R_joint, L_list, action_subsets, bitmask_to_action = build_joint_mdp_hetero(
        machine_locals, beta, R=cfg.R)

    V_opt, pi_star, _ = solve_joint_vi(P_joint, R_joint, beta)  # solving the full joint MDP
                                                                 # for the optimal policy and value function

    results = {}
    for method in ["ClosedForm", "VI", "PI", "Package", "Theorem"]:
        W_list = [method_W[m][method] for m in range(M)]
        if any(np.any(np.isnan(w)) for w in W_list):
            results[method] = None
            continue
        pi_method = index_policy_from_Whittle_indices(W_list, L_list, M, cfg.R, bitmask_to_action)
        V_method = evaluate_joint_policy(pi_method, P_joint, R_joint, beta)
        subopt = 100.0 * (V_method - V_opt) / V_opt
        idle = (pi_method == 0).astype(float)
        results[method] = dict(subopt=subopt, idle=idle, pi=pi_method, V=V_method,
                                pct_matches_opt=100.0 * np.mean(pi_method == pi_star))

    pi_greedy = greedy_policy_hetero(L_list, M, N_list, cfg.R, bitmask_to_action)
    V_greedy = evaluate_joint_policy(pi_greedy, P_joint, R_joint, beta)
    subopt_greedy = 100.0 * (V_greedy - V_opt) / V_opt
    idle_greedy = (pi_greedy == 0).astype(float)
    results["Greedy"] = dict(subopt=subopt_greedy, idle=idle_greedy, pi=pi_greedy, V=V_greedy,
                              pct_matches_opt=100.0 * np.mean(pi_greedy == pi_star))

    log_info = dict(N_list=N_list, M=M, machine_params=machine_params, method_W=method_W,
                     L_list=L_list, action_subsets=action_subsets,
                     S=int(np.prod(L_list)), pi_star=pi_star, V_opt=V_opt,
                     P_joint=P_joint, R_joint=R_joint, instance_id=instance_id,
                     alpha_label=alpha, switch_frac_label=switch_frac)

    if verbose:
        _log_instance(D, alpha, switch_frac, cfg, log_info, results)

    return results, log_info

    
def _log_instance(D, alpha, switch_frac, cfg: ExperimentConfig, log_info, results):
    """All readable logging for one instance. Every print() here goes
    to both the terminal and the .txt transcript, because main() redirects
    sys.stdout through a TeeWriter before calling any of this."""
    M = log_info["M"]
    print("=" * 88)
    print(f"D={D}  alpha={alpha}  switch_frac={switch_frac}  R={cfg.R}  M={M}  beta={cfg.beta}  "
          f"N_list={log_info['N_list']}  seed={log_info['instance_id']}")
    print("=" * 88)

    if cfg.verbose_indices:
        for m, mp in enumerate(log_info["machine_params"]):
            inside = mp["C_switch"] < mp["c_bar"]
            print(f"\nMachine {m + 1}: N={mp['N']}  C={mp['C']:.3f}  K={mp['K']:.3f}  "
                  f"(alpha_m={mp['alpha_m']})  C_switch={mp['C_switch']:.3f}  "
                  f"c_bar={mp['c_bar']:.3f}  [{'INSIDE' if inside else 'OUTSIDE'} domain]")
            for method, W in log_info["method_W"][m].items():
                print(f"    {method:10s}: {np.round(W, 4)}")

    S = log_info["S"]
    L_list = log_info["L_list"]
    A = len(log_info["action_subsets"])
    print(f"\nJoint state space size: {S} = " + "*".join(map(str, L_list)))

    if cfg.verbose_joint_shapes:
        print(f"Joint action space size: {A}  (R={cfg.R} repairmen, M={M} machines)")
        print(f"Joint transition matrices: {[P.shape for P in log_info['P_joint']]}")
        print(f"Joint reward vectors: {[R.shape for R in log_info['R_joint']]}")
        if cfg.dump_full_matrices:
            # WARNING: prints dense S x S matrices, one per action -- only
            # sane for small S (a handful of machines with small N). Left
            # off by default; flip on only for a genuinely tiny sanity case.
            print(f"P_joint =\n{log_info['P_joint']}")
            print(f"R_joint =\n{log_info['R_joint']}")

    rows = []
    for method, val in results.items():
        if val is None:
            rows.append(dict(Method=method, q1=np.nan, median=np.nan, q3=np.nan,
                              max=np.nan, pct_idle=np.nan, pct_matches_OPT=np.nan))
            continue
        subopt = val["subopt"]
        rows.append(dict(
            Method=method,
            q1=np.percentile(subopt, 25), median=np.percentile(subopt, 50),
            q3=np.percentile(subopt, 75), max=subopt.max(),
            pct_idle=100.0 * val["idle"].mean(),
            pct_matches_OPT=val["pct_matches_opt"],
        ))
    inst_df = pd.DataFrame(rows)
    print("\n" + "-" * 88)
    print("THIS INSTANCE -- suboptimality (%) summary")
    print("-" * 88)
    print(inst_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if cfg.verbose_policies:
        print("\nAction legend (action index -> machines serviced, 0-based):")
        for a, combo in enumerate(log_info["action_subsets"]):
            print(f"    action {a}: service machines {combo if combo else '(idle)'}")

        print(f"\nsum(V_opt) = {log_info['V_opt'].sum():.3f}")
        for method, val in results.items():
            if val is None:
                continue
            print(f"sum(V_{method}) = {val['V'].sum():.3f}")

        if S <= cfg.policy_dump_max_states:
            print(f"\nOptimal (DP) policy pi_star, one action per joint state (S={S}):")
            print(f"    {log_info['pi_star'].tolist()}")
            print(f"V_opt ({S} states): {np.round(log_info['V_opt'], 3).tolist()}")
            for method, val in results.items():
                if val is None:
                    continue
                print(f"{method:10s} policy pi ({S} states): {val['pi'].tolist()}")
                print(f"{method:10s} V        ({S} states): {np.round(val['V'], 3).tolist()}")
        else:
            print(f"\n(S={S} > policy_dump_max_states={cfg.policy_dump_max_states} -- "
                  f"skipping full pi/V dump; see pct_matches_OPT above)")

    print("\nDUPLICATION CHECK:")
    if results.get("ClosedForm") is not None and results.get("Theorem") is not None:
        print("  ClosedForm policy == Theorem policy?",
              np.array_equal(results["ClosedForm"]["pi"], results["Theorem"]["pi"]))
    if results.get("VI") is not None and results.get("PI") is not None:
        print("  VI policy == PI policy?             ",
              np.array_equal(results["VI"]["pi"], results["PI"]["pi"]))
    if results.get("VI") is not None and results.get("Package") is not None:
        print("  VI policy == Package policy?        ",
              np.array_equal(results["VI"]["pi"], results["Package"]["pi"]))
    if results.get("ClosedForm") is not None and results.get("VI") is not None:
        pi_cf, pi_vi = results["ClosedForm"]["pi"], results["VI"]["pi"]
        n_diff = int(np.sum(pi_cf != pi_vi))
        print(f"  ClosedForm vs VI: differ on {n_diff}/{len(pi_cf)} states "
              f"({100 * n_diff / len(pi_cf):.1f}%)")

    print("\nSanity check V_opt >= V_method (none beats true OPT):")
    for method, val in results.items():
        if val is None:
            continue
        print(f"  {method:10s}: {bool(np.all(val['subopt'] >= -1e-6))}")
    print()


# =====================================================================
#  n_reps problem instances per cell, POOLED, then quantiled
# =====================================================================
def run_glazebrook_table(cfg: ExperimentConfig, verbose=True):
    """For every (D, alpha, switch_frac) cell: draw cfg.n_reps independent 
    problem instances, POOL every instance's per state subopt/idle arrays
    together (n_reps * S values), and take quartiles of the POOLED data --
    matching Table 3's own description of a 10*512=5120-pooled summary, not
    an average of n_reps separate quartiles.

    One rng is created ONCE from cfg.seed0 and advanced through every draw
    across the whole sweep (switch_frac outer, then D, then alpha, then
    rep) -- re-running with the same seed0 reproduces the whole table.
    """
    rng = np.random.default_rng(cfg.seed0)
    rows = []
    instance_counter = cfg.seed0
    for switch_frac in cfg.switch_fracs:
        sf_label = switch_frac_cell_label(switch_frac)          
        for D in cfg.D_grid:
            for alpha in cfg.alpha_grid:
                pooled_subopt = {m: [] for m in METHODS}
                pooled_idle = {m: [] for m in METHODS}
                pooled_match = {m: [] for m in METHODS}
                for rep in range(cfg.n_reps):
                    instance_counter += 1
                    print(f"\n########## switch_frac={sf_label}  D={D}  alpha={alpha}  "   # <-- sf_label, not switch_frac
                          f"rep={rep + 1}/{cfg.n_reps} ##########")
                    results, _ = run_cell(cfg, D, alpha, switch_frac, rng,                  # <-- raw switch_frac unchanged
                                           instance_id=instance_counter, verbose=verbose)
                    for method, val in results.items():
                        if val is None:
                            continue
                        pooled_subopt[method].append(val["subopt"])
                        pooled_idle[method].append(val["idle"])
                        pooled_match[method].append(val["pct_matches_opt"])

                for method in METHODS:
                    if not pooled_subopt[method]:
                        rows.append(dict(D=D, alpha=alpha, switch_frac=sf_label, Method=method,   # <-- sf_label
                                          q1=np.nan, median=np.nan, q3=np.nan, max=np.nan,
                                          pct_idle=np.nan, mean_pct_matches_OPT=np.nan, n_pooled=0))
                        continue
                    subopt_all = np.concatenate(pooled_subopt[method])
                    idle_all = np.concatenate(pooled_idle[method])
                    q1, med, q3 = np.percentile(subopt_all, [25, 50, 75])
                    rows.append(dict(
                        D=D, alpha=alpha, switch_frac=sf_label, Method=method,     # <-- sf_label
                        q1=q1, median=med, q3=q3, max=subopt_all.max(),
                        pct_idle=100.0 * idle_all.mean(),
                        mean_pct_matches_OPT=float(np.mean(pooled_match[method])),
                        n_pooled=len(subopt_all),
                    ))
                    if verbose:
                        print(f"  CELL SUMMARY [{method}] switch_frac={sf_label} D={D} alpha={alpha}  "   # <-- sf_label
                              f"subopt%[{q1:.3f},{med:.3f},{q3:.3f}] max={subopt_all.max():.3f}  "
                              f"idle%={100 * idle_all.mean():.2f}  (pooled over {len(subopt_all)} states)")
    return pd.DataFrame(rows)

# =====================================================================
#  Aggregate printout: rows=D, columns=alpha, each cell a
#  four-vector (q1, median, q3) on one line + pct_idle on the next.
# =====================================================================
def print_table_style(df: pd.DataFrame, cfg: ExperimentConfig, method: str, switch_frac: float):
    sub = df[(df["Method"] == method) & (df["switch_frac"] == switch_frac)]
    if sub.empty:
        return
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {switch_frac}   "
          f"(cost suboptimality %, quartile triple / pct idle -- Table)")
    print("=" * 100)

    col_width = 22
    header = "D".rjust(6) + "".join(str(a).rjust(col_width) for a in cfg.alpha_grid)
    print(header)
    for D in cfg.D_grid:
        line1 = str(D).rjust(6)
        line2 = " " * 6
        for alpha in cfg.alpha_grid:
            row = sub[(sub["D"] == D) & (sub["alpha"] == alpha)]
            if row.empty or pd.isna(row["median"].values[0]):
                line1 += "n/a".rjust(col_width)
                line2 += "".rjust(col_width)
                continue
            q1, med, q3, idle = (row["q1"].values[0], row["median"].values[0],
                                  row["q3"].values[0], row["pct_idle"].values[0])
            triple = f"{q1:.3f} {med:.3f} {q3:.3f}"
            line1 += triple.rjust(col_width)
            line2 += f"{idle:.2f}".rjust(col_width)
        print(line1)
        print(line2)
    print()


def main():
    # -----------------------------------------------------------------
    # QUICK LOW COST PASS (uncomment to eyeball the transcript shape in a
    # couple of minutes before running the real sweep below):
    #
    # cfg = ExperimentConfig(N_list=[7, 7, 7], n_reps=2, D_grid=[20.0],
    #                         alpha_grid=[2.0, "Various"], switch_fracs=[0.0],
    #                         seed0=1000, output_txt="repairman_quicktest.txt")
    # -----------------------------------------------------------------
##########

    # cfg = ExperimentConfig(
    # N_list=[7, 7, 7],
    # R=1,
    # beta=0.95,
    # n_reps=10,
    # D_grid=[0, 10, 20, 25, 30, 40, 50],
    # alpha_grid=[1.5, 2.0, 3.0, 4.0, 5.0, "Various"],
    # C_width=25.0,
    # switch_fracs=[[0.3, 1.2, 0.0]],  # heterogeneous switch_frac per machine
    # seed0=1000,)

##########





    # REAL CONFIG -- reproduces Table 3's structure (R=1, M=3, 8 damage
    # states/machine, 7x6=42 cells, n_reps=10 pooled instances/cell).
    # Runtime scales as len(D_grid) * len(alpha_grid) * len(switch_fracs) *
    # n_reps -- the full 7x6x1x10 = 420-replication sweep is the ~23-minute
    # run; trim any grid or n_reps for a faster pass.
    cfg = ExperimentConfig(
        N_list=[7,7,7],  # 8 damage states per machine
        R=1,
        beta=0.95,
        n_reps=10,
        D_grid=[0, 10, 20, 25, 30, 40, 50],
        alpha_grid=[1.5, 2.0, 3.0, 4.0, 5.0, "Various"],
        C_width=25.0,
        switch_fracs=[[0.3, 1.2, 0.0]],  # heterogeneous switch_frac per machine
        seed0=1000,
        verbose_indices=True,
        verbose_policies=True,
        verbose_joint_shapes=True,
        dump_full_matrices=False,   # keep off -- S x S dumps get huge fast
        output_txt=f"repairman_results_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
    )


    old_stdout = sys.stdout
    logfile = open(cfg.output_txt, "w", encoding="utf-8")
    sys.stdout = TeeWriter(old_stdout, logfile)
    try:
        #print(f"Run started: {datetime.datetime.now().isoformat()}")
        print(f"Run started: {datetime.now().isoformat()}")
        print(f"markovianbandit package available: {PKG_AVAILABLE}")
        print(f"Config: {cfg}\n")


        df = run_glazebrook_table(cfg, verbose=True)

        print("\n\n" + "=" * 100)
        print("AGGREGATE TABLES (pooled quartiles per (D, alpha, switch_frac, Method) cell)")
        print("=" * 100)
        for method in METHODS:
            for switch_frac in cfg.switch_fracs:
                print_table_style(df, cfg, method, switch_frac_cell_label(switch_frac))
                                                                                           



        print(f"\nRun finished: {datetime.now().isoformat()}")
        print(f"Transcript written to: {cfg.output_txt}")
        
    finally:
        sys.stdout = old_stdout
        logfile.close()


if __name__ == "__main__":
    main()
"""
Heterogeneous multi-machine (R repairmen) maintenance problem  --  COST MINIMISATION.

Computes the Whittle index FOUR different ways per machine (Closed Form,
VI, PI, Package) (plus a Theorem based closed form), builds the resulting
FIVE index heuristic joint policies, plus a Greedy policy, evaluates all
SIX against the true joint OPT (via full value iteration on the joint
MDP). We build BOTH a switching-AWARE joint policy (index computed with
each machine's real, positive C_switch) and a switching-UNAWARE joint
policy (the SAME formula, but with C_switch equal to 0), evaluate both,
together with the true OPT and a non-index Greedy policy on the TRUE joint
system and report:

  (a) the per-STATE suboptimality  delta_pi(s) = 100 (V_pi(s)-V*(s))/V*(s)
      (pooled across n_reps instances, quartiles taken)   [Glazebrook et al. 2005]
  (b) the scalar J_mu(pi) = avg_s V_pi(s) = (1/S) sum_s V_pi(s)
      ("expected discounted COST if the initial joint state is uniformly random")
      and the ratio-of-means measure
           Delta_mu(pi) = 100 (J_mu(pi) - J_mu(pi*)) / J_mu(pi*)
      The two measures differ in general: delta_pi(s) is per state (pooled over
      random instances and summarised by quartiles); Delta_mu is one number
      per individual instance.

"""

# cd "Project Lab System/Machine Maintenance"

import os
import sys
import platform
import secrets
from datetime import datetime
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False




# =====================================================================
# Single machine local MDP  (COSTS, all >= 0)
# =====================================================================
def build_extended_mdp(N, p, C, K, C_switch):
    """Returns P0, P1 (transition matrices), c0, c1 (one period COSTS, >=0),
    and the local state labels.
    Local state order: [(0,1), (0,0), (1,0), ..., (N,0)]
    Active (a=1) cost:  C  from state (0,1) (already attended last period),
                        C + C_switch from every other state.
    Passive (a=0) cost: expected breakdown cost  K * (1 - p_x)."""
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    c0 = np.zeros(num_states)
    c1 = np.zeros(num_states)
    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N + 1)]
    for s in range(num_states):
        if s == 0:
            P1[s, 0] = 1.0
            c1[s] = C
        else:
            P1[s, 0] = 1.0
            c1[s] = C + C_switch
    for s in range(num_states):
        if s == 0:
            P0[s, 2] = p[0]
            P0[s, 1] = 1 - p[0]
            c0[s] = K * (1 - p[0])
        elif s == 1:
            P0[s, 2] = p[0]
            P0[s, 1] = 1 - p[0]
            c0[s] = K * (1 - p[0])
        else:
            idx = s - 1
            if idx < N:
                P0[s, s + 1] = p[idx]
                P0[s, 1] = 1 - p[idx]
                c0[s] = K * (1 - p[idx])
            else:
                P0[s, 1] = 1.0
                c0[s] = K
    return P0, P1, c0, c1, state_labels


# =====================================================================
# Discounted solvers (VI, PI) -> single machine index by bisection
# (cost minimisation; w = tax on being active)
# =====================================================================
def solve_vi_discounted(P0, P1, c0, c1, w, beta, tol=1e-8, max_iter=10000):
    S = len(c0)
    V = np.zeros(S)
    for _ in range(max_iter):
        Q0 = c0 + beta * (P0 @ V)
        Q1 = (c1 + w) + beta * (P1 @ V)
        V_new = np.minimum(Q0, Q1)  # Cost Settings
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    Q0 = c0 + beta * (P0 @ V)
    Q1 = (c1 + w) + beta * (P1 @ V)
    return V, Q0, Q1


def policy_evaluation_discounted(pi, P0, P1, c0, c1, w, beta):
    S = len(pi)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    c_pi = np.array([c0[s] if pi[s] == 0 else c1[s] + w for s in range(S)])
    return np.linalg.solve(np.eye(S) - beta * P_pi, c_pi)


def solve_pi_discounted(P0, P1, c0, c1, w, beta, tol=1e-8, max_iter=1000):
    S = len(c0)
    pi = np.zeros(S, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation_discounted(pi, P0, P1, c0, c1, w, beta)
        Q0 = c0 + beta * (P0 @ V)
        Q1 = (c1 + w) + beta * (P1 @ V)
        pi_new = (Q1 < Q0).astype(int)        # active only if strictly cheaper
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = c0 + beta * (P0 @ V)
    Q1 = (c1 + w) + beta * (P1 @ V)
    return V, Q0, Q1


def whittle_index_discounted(state, solver, beta, P0, P1, c0, c1,
                              w_min=-2000.0, w_max=2000.0, tol_w=1e-6, tol_mdp=1e-9, Verbose=False):
    """Index = activation tax w at which Q1(state) = Q0(state) (cost to go)."""
    solve_f = solve_vi_discounted if solver == 'vi' else solve_pi_discounted

    def f(w):
        _, Q0, Q1 = solve_f(P0, P1, c0, c1, w, beta, tol_mdp)
        return Q1[state] - Q0[state]          # increasing in w

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
    """Returns W as a length-(N+2) array in the same state order as
    build_extended_mdp: [W(0,1), W(0,0), W(1,0), ..., W(N,0)]."""
    N = len(p)
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
    w_passive = np.zeros(N + 1)
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
    """Indexability bound: BOTH ClosedForm and Theorem (they are the same
    index, two algebraic derivations of it) are only proven to be a valid
    (monotone) Whittle index for 0 <= C_switch < c_bar."""
    return K * (p0 - p1) / (1 + beta * (p0 - p1))


# =====================================================================
# Compute all index arrays (different methods) for one machine
# =====================================================================
def all_method_indices(mp, beta):
    N, p, C, K, C_switch = mp["N"], mp["p"], mp["C"], mp["K"], mp["C_switch"]
    P0, P1, c0, c1, labels = build_extended_mdp(N, np.append(p, 0.0), C, K, C_switch)
    S = N + 2

    W_cf = closed_form_indices(p, beta, K, C, C_switch)
    W_th = theorem_indices(p, beta, K, C, C_switch)
    W_vi = np.array([whittle_index_discounted(s, 'vi', beta, P0, P1, c0, c1) for s in range(S)])
    W_pi = np.array([whittle_index_discounted(s, 'pi', beta, P0, P1, c0, c1) for s in range(S)])

    if PKG_AVAILABLE:
        # the package is REWARD based: reward = -cost.  Same index as ours.
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, -c0, -c1)
        W_pkg = np.array(model.whittle_indices(discount=beta))
    else:
        W_pkg = np.full(S, np.nan)

    return {"ClosedForm": W_cf, "VI": W_vi, "PI": W_pi, "Package": W_pkg, "Theorem": W_th}, labels


# =====================================================================
#  Heterogeneous joint MDP (via Kronecker structure) + policies
#  Needed to compute the true optimal value/policy on small instances, precisely so one can
#  measure the optimality gap of the Whittle index policy against ground truth
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
    own local MDP (P0,P1,c0,c1) and possibly different number of damage states N.
    The joint state space is the Cartesian product of the local state spaces and
    the joint action space is all feasible combinations of servicing up to R
    machines simultaneously. The joint transition matrix for a given joint action
    is the Kronecker product of the local transition matrices (P1 if the machine
    is serviced, P0 if not). The joint one period COST vector is the sum of the
    local cost vectors (c1 if serviced, c0 if not).

    Returns:
    P_joint : list of transition matrices, one per joint action
    C_joint : list of one-period cost vectors, one per joint action
    L_list : list of local state space sizes
    action_subsets : list of feasible joint actions
    bitmask_to_action : dict mapping serviced machine bitmasks to action indices


    np.kron(A, B) builds the tensor product matrix. If the two machines evolve 
    independently conditional on the joint action,
    the joint transition probability factorizes as
               P(s0', s1' | s0, s1) = P_m0(s0' | s0) * P_m1(s1' | s1)
    The Kronecker product is precisely the matrix that computes this
    factorized probability for every pair of joint states in one shot, with
    the joint state flattened to a single index. 

    NOTE ON WHY THIS ISN'T 2^M OR 2*M ACTIONS: with R repairmen we cannot
    simply combine per machine active/passive choices independently (that
    would allow more than R machines active at once). The physically realizable
    joint actions are exactly "pick a subset of size <=R of the M machines to
    service"; hence sum_{k=0}^{min(R,M)} C(M,k) actions i.e. 4
    for R=1,M=3, 11 for R=2,M=4 and likewise.

    """

    M = len(machine_locals)
    L_list = [machine_locals[m]["P0"].shape[0] for m in range(M)]
    action_subsets, bitmask_to_action = enumerate_actions(M, R)
    P_joint, C_joint = [], []
    for combo in action_subsets:
        combo_set = set(combo)
        mats = [machine_locals[m]["P1"] if m in combo_set else machine_locals[m]["P0"] for m in range(M)]
        Pa = mats[0]
        for mat in mats[1:]:
            Pa = np.kron(Pa, mat)
        P_joint.append(Pa)
        shape = tuple(L_list)
        Cgrid = np.zeros(shape)
        for m in range(M):
            c_m = machine_locals[m]["c1"] if m in combo_set else machine_locals[m]["c0"]
            bshape = [1] * M # bshape is just a way of telling numpy "this vector
                # varies along axis m and is constant (size 1) along every other axis
            bshape[m] = L_list[m]   # c_m varies along axis m, constant along the others
            Cgrid = Cgrid + c_m.reshape(bshape)
        C_joint.append(Cgrid.reshape(-1))
    return P_joint, C_joint, L_list, action_subsets, bitmask_to_action


def solve_joint_vi(P_joint, C_joint, beta, tol=1e-8, max_iter=5000):
    """Full value iteration (MINIMISING cost) -> V_OPT (>=0) and the true
    optimal joint policy."""
    S = len(C_joint[0])
    A = len(C_joint)
    V = np.zeros(S)
    for _ in range(max_iter):
        Qs = np.stack([C_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
        V_new = Qs.min(axis=0)  # Cost settings
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Qs = np.stack([C_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
    pi_star = Qs.argmin(axis=0)
    return V, pi_star, Qs


def evaluate_joint_policy(pi, P_joint, C_joint, beta):
    """Exact expected discounted COST V_pi of a fixed deterministic joint
    policy pi, via linear solve."""
    S = len(C_joint[0])
    P_pi = np.zeros((S, S))
    c_pi = np.zeros(S)
    for a in range(len(C_joint)):
        mask = (pi == a)
        P_pi[mask, :] = P_joint[a][mask, :]
        c_pi[mask] = C_joint[a][mask]
    return np.linalg.solve(np.eye(S) - beta * P_pi, c_pi)


def index_policy_from_Whittle_indices(W_list, L_list, M, R, bitmask_to_action):
    S = int(np.prod(L_list))
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
        level(1) = 0   -- (0,0)
        level(s) = s-1 -- (s-1, 0) for s >= 2
    """
    levels = np.zeros(L, dtype=int)
    levels[0] = 0
    levels[1] = 0
    for s in range(2, L):
        levels[s] = s - 1
    return levels


def greedy_policy_hetero(L_list, M, N_list, R, bitmask_to_action):
    """Always service the top min(R, M) machines by CURRENT damage level,
    ignoring cost, indices and the (x,1)/(x,0) flag. Never idles.
    Ties are broken by lowest machine index (see earlier discussion: fully tied
    states are only hypothetical initial states, so this only matters for
    reproducibility).

    TIE-BREAKING. When two or more machines share a damage level at the
    R-th cutoff (e.g. {(0,1),(0,1),(0,1)}, a 3 way tie at level 0 with
    R=1), this implementation picks the LOWEST INDEXED tied machine(s)
    first. Two things worth noting about this:

          1. A fully tied state like {(0,1),(0,1),(0,1)} can never actually be
         *reached* by the joint dynamics from any real trajectory; the
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


def fraction_tied_states(L_list, M, N_list, R):
    """Diagnostic: what fraction of the prod(L_list) joint states have a
    damage level tie AT the Rth cutoff?"""

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
# This is a fixed, paper-specific column that is ONLY defined for M=3.
VARIOUS_ALPHA_MULTIPLIERS = [1.5, 3.0, 4.5]

AlphaSpec = Union[float, str, List[float], Tuple[float, ...]]
VariousMultipliersSpec = Union[List[float], Tuple[float, ...], Callable[[int], List[float]]]


def resolve_alpha_per_machine(alpha: AlphaSpec, M: int,
                               various_multipliers: Optional[List[float]] = None) -> List[float]:
    """alpha can be:
      - a plain number  -> same alpha for every machine (K_m = alpha * C_m)
      - the string "Various" -> per machine multipliers taken from
        various_multipliers (already resolved to a length-M list by
        ExperimentConfig.__post_init__)
      - a list/tuple of length M -> explicit per machine alphas
    """
    if isinstance(alpha, str):
        if alpha != "Various":
            raise ValueError(f"Unknown alpha spec: {alpha!r}")
        if various_multipliers is None or len(various_multipliers) != M:
            # Should already have been caught by ExperimentConfig.__post_init__;
            # this is just a safety net in case resolve_alpha_per_machine is
            # ever called directly with a hand-built (not __post_init__'d) cfg.
            raise ValueError(
                f'"Various" alpha needs a length-M={M} list of per-machine '
                f"multipliers, got {various_multipliers!r}. Set "
                f"cfg.various_alpha_multipliers explicitly (list or callable "
                f"M -> List[float]) -- required whenever M != 3, since the "
                f"paper's own {VARIOUS_ALPHA_MULTIPLIERS} column is only ever "
                f"defined for M=3.")
        return [float(a) for a in various_multipliers]
    if isinstance(alpha, (list, tuple)):
        if len(alpha) != M:
            raise ValueError(f"alpha list length {len(alpha)} != M={M}")
        return [float(a) for a in alpha]
    return [float(alpha)] * M


SwitchFracSpec = Union[float, List[float], Tuple[float, ...]]


def resolve_switch_frac_per_machine(switch_frac: SwitchFracSpec, M: int) -> List[float]:
    """switch_frac can be:
      - a plain number  -> same switch_frac for every machine (homogeneous)
      - a list/tuple of length M (heterogeneous, per machine) -> explicit per
        machine switch_frac, e.g. [0.3, 1.2, 0.0] lets some machines sit inside
        their own c_bar domain and others outside it within the SAME instance (heterogeneous switch_frac). 
        It is allowed here for completeness (Not explored in paper).
    """
    if isinstance(switch_frac, (list, tuple)):
        if len(switch_frac) != M:
            raise ValueError(f"switch_frac list length {len(switch_frac)} != M={M}")
        return [float(s) for s in switch_frac]
    return [float(switch_frac)] * M


def switch_frac_cell_label(switch_frac):
    """Hashable, sortable, human readable label for one switch_fracs entry,
    for use as a DataFrame grouping key / table column. Scalars pass
    through unchanged; list/tuple entries become a short 'mixed_...' tag."""
    if isinstance(switch_frac, (list, tuple)):
        return "mixed_" + "-".join(f"{float(s):g}" for s in switch_frac)
    return switch_frac


@dataclass
class ExperimentConfig:
    # machines & per machine heterogeneity in number of damage levels N.
    # Set N_list=None to draw each machine's N ~ Uniform N_range, drawing
    # M itself from M_range too (see __post_init__ / make_random_config).
    N_list: Optional[List[int]] = field(default_factory=lambda: [6, 6, 6])
    M: int = 3                          # DERIVED from N_list in __post_init__
    N_range: Tuple[int, int] = (3, 6)   # only used if N_list is None
    M_range: Tuple[int, int] = (2, 5)   # only used if N_list is None

    R: int = 1                         # number of repairmen (joint action constraint)
    beta: float = 0.95                 # discount factor

    n_reps: int = 10                   # independent problem instances per cell

    D_grid: List[float] = field(default_factory=lambda: [0, 10, 20, 25, 30, 40, 50])
    alpha_grid: List[AlphaSpec] = field(
        default_factory=lambda: [1.5, 2.0, 3.0, 4.0, 5.0, "Various"])
    C_width: float = 25.0

    various_alpha_multipliers: Optional[VariousMultipliersSpec] = None

    # each entry may be a scalar (homogeneous) or a length-M list/tuple (heterogeneous)
    switch_fracs: List[SwitchFracSpec] = field(default_factory=lambda: [0.0])

    # MASTER SEED.  Instance k of the sweep uses instance_seed = seed0 + k
    # (k = 1,2,3,...).  Keep seed0 values at least (#instances) apart if you run
    # several sweeps and want disjoint instances.
    seed0: int = 1000

    verbose_indices: bool = True        # log per machine params + all W arrays (aware & unaware)
    verbose_policies: bool = True       # log pi_star / each method's pi / V's
    verbose_joint_shapes: bool = True   # log joint action/state/matrix shapes
    dump_full_matrices: bool = False    # DANGEROUS for large S
    policy_dump_max_states: int = 10000

    # state legend + original-form state->action printouts
    print_state_legend: bool = True
    print_state_action_tables: bool = True
    state_print_chunk: int = 100        # first / middle / last this many states

    output_txt: Optional[str] = None    # filled in __post_init__ if left None

    def __post_init__(self):
        if self.N_list is not None:
            self.M = len(self.N_list)

        if "Various" in self.alpha_grid:
            if callable(self.various_alpha_multipliers):
                mult = list(self.various_alpha_multipliers(self.M))
                if len(mult) != self.M:
                    raise ValueError(
                        f"various_alpha_multipliers callable returned "
                        f"{len(mult)} values for M={self.M}; it must return "
                        f"exactly M values.")
                self.various_alpha_multipliers = mult
            elif self.various_alpha_multipliers is not None:
                if len(self.various_alpha_multipliers) != self.M:
                    raise ValueError(
                        f"cfg.various_alpha_multipliers has length "
                        f"{len(self.various_alpha_multipliers)} but M={self.M}; "
                        f"they must match.")
                self.various_alpha_multipliers = list(self.various_alpha_multipliers)
            elif self.M == 3:
                self.various_alpha_multipliers = list(VARIOUS_ALPHA_MULTIPLIERS)
            else:
                raise ValueError(
                    f'alpha_grid contains "Various" but M={self.M} (!=3). The '
                    f"paper only defines \"Various\" for M=3. You must explicitly "
                    f"set cfg.various_alpha_multipliers to a length-{self.M} list "
                    f"or a callable M -> List[float].")

        if self.output_txt is None:
            now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_txt = f"Discounted_cost_repairman_results_{now_str}.txt"


def resolve_N_list(cfg: ExperimentConfig, rng: np.random.Generator) -> List[int]:
    if cfg.N_list is not None:
        return list(cfg.N_list)
    lo, hi = cfg.N_range
    return [int(rng.integers(lo, hi + 1)) for _ in range(cfg.M)]


def make_random_config(
    *,
    M_range: Tuple[int, int] = (2, 5),
    N_range: Tuple[int, int] = (3, 8),
    R_max: Optional[int] = None,
    various_alpha_multipliers: Optional[VariousMultipliersSpec] = None,
    seed0: Optional[int] = None,
    meta_seed: Optional[int] = None,
    **overrides,
) -> ExperimentConfig:
    """Builds a FRESH ExperimentConfig with a randomly drawn number of
    machines M, per-machine damage-state count N_list, repairman count R and
    (if not given) seed0.  EVERYTHING is determined by meta_seed, which is
    drawn from the OS CSPRNG if not supplied and is ALWAYS printed: pass that
    integer back as meta_seed=... to regenerate exactly the same experiment
    (keep the printed value and the library versions that the run header prints).

    NOTE on "Various": if alpha_grid includes "Various" and M is drawn
    randomly, various_alpha_multipliers must be a CALLABLE (M -> List[float]).
    """
    if meta_seed is None:
        meta_seed = secrets.randbits(31)
    meta_rng = np.random.default_rng(meta_seed)
    M = int(meta_rng.integers(M_range[0], M_range[1] + 1))
    N_list = [int(meta_rng.integers(N_range[0], N_range[1] + 1)) for _ in range(M)]
    R = int(meta_rng.integers(1, (R_max or M) + 1))
    resolved_seed0 = seed0 if seed0 is not None else int(meta_rng.integers(0, 2 ** 31 - 1))

    cfg = ExperimentConfig(
        N_list=N_list,
        R=R,
        various_alpha_multipliers=various_alpha_multipliers,
        seed0=resolved_seed0,
        **overrides,
    )
    print(f"[random experiment] meta_seed={meta_seed} -> M={M} machines, "
          f"N_list={N_list}, R={R}, seed0={resolved_seed0}\n"
          f"  (re-run make_random_config(meta_seed={meta_seed}, ...) with the same "
          f"other arguments to replay exactly this experiment)")
    return cfg


# =====================================================================
# {Aware, Unaware}
# =====================================================================
INDEX_METHODS = ["ClosedForm", "VI", "PI", "Package", "Theorem"]

# 12 total: OPT, {method}_Aware x5, {method}_Unaware x5, Greedy
UNIFORM_METHODS = (
    ["OPT"]
    + [f"{m}_Aware" for m in INDEX_METHODS]
    + [f"{m}_Unaware" for m in INDEX_METHODS]
    + ["Greedy"]
)

_SHORT = {"ClosedForm": "CF", "VI": "VI", "PI": "PI", "Package": "Pkg", "Theorem": "Thm"}
SHORT_NAME = {"OPT": "OPT", "Greedy": "Greedy"}
for _m in INDEX_METHODS:
    SHORT_NAME[f"{_m}_Aware"] = f"{_SHORT[_m]}_A"
    SHORT_NAME[f"{_m}_Unaware"] = f"{_SHORT[_m]}_U"


def _format_action_combo(combo):
    """1 based, human readable label, e.g. () -> '(idle)',
    (0,) -> 'machine (1,)', (0,2) -> 'machines (1, 3)'."""
    if not combo:
        return "(idle)"
    one_based = tuple(m + 1 for m in combo)
    noun = "machine" if len(one_based) == 1 else "machines"
    return f"{noun} {one_based}"


# ---------------------------------------------------------------------
# sampled state printing, state strings
# ---------------------------------------------------------------------
def sampled_state_blocks(S, chunk):
    """Index blocks to print: everything if S <= 3*chunk, otherwise the first,
    the middle and the last `chunk` states."""
    if chunk <= 0 or S <= 3 * chunk:
        return [list(range(S))]
    mid_start = S // 2 - chunk // 2
    return [list(range(0, chunk)),
            list(range(mid_start, mid_start + chunk)),
            list(range(S - chunk, S))]


def print_sampled_rows(S, chunk, row_fn, indent="    "):
    blocks = sampled_state_blocks(S, chunk)
    prev_end = -1
    for block in blocks:
        if prev_end >= 0 and block[0] > prev_end + 1:
            print(f"{indent}   ...   (states {prev_end + 1}..{block[0] - 1} not shown)")
        for s in block:
            print(indent + row_fn(s))
        prev_end = block[-1]


def make_state_str_fn(L_list, labels_by_machine):
    shape = tuple(L_list)

    def state_str(s):
        idx = np.unravel_index(s, shape)
        return "[" + " | ".join(labels_by_machine[m][idx[m]] for m in range(len(L_list))) + "]"
    return state_str


def idle_pct_from_counts(idle_count, R):
    """% of states with exactly k idle repairmen, k = 0..R."""
    return 100.0 * np.bincount(idle_count, minlength=R + 1)[:R + 1] / len(idle_count)


def _make_result(pi, V, V_opt, pi_star, n_active_per_action, R):
    S = len(V)
    idle_count = R - n_active_per_action[pi]
    return dict(
        subopt=100.0 * (V - V_opt) / V_opt,                  # per-state delta_pi(s) (%)
        idle_count=idle_count,                               # #idle repairmen in each state
        pi=pi, V=V, avg=V.mean(), sum=V.sum(),
        delta_mu=100.0 * (V.mean() - V_opt.mean()) / V_opt.mean(),   # ratio of means (%)
        pct_matches_opt=100.0 * np.mean(pi == pi_star),
    )


def run_cell_uniform(cfg: ExperimentConfig, D, alpha, switch_frac, instance_seed, verbose=True):
    """Draws ONE random problem instance, fully determined by instance_seed.
    For EVERY index method builds BOTH a switching AWARE policy (index computed
    with each machine's real C_switch) and a switching UNAWARE policy (C_switch=0),
    plus OPT (true joint value iteration) and Greedy. Every policy is evaluated
    via linear solve on the TRUE joint system.

    Returns:
      results  : dict {method_name -> dict(subopt, idle_count, pi, V, avg, sum,
                 delta_mu, pct_matches_opt)} for every entry in UNIFORM_METHODS
      log_info : dict with everything needed for logging
    """
    rng = np.random.default_rng(instance_seed)
    N_list = resolve_N_list(cfg, rng)
    M = len(N_list)
    beta = cfg.beta
    alpha_per_machine = resolve_alpha_per_machine(alpha, M, cfg.various_alpha_multipliers)
    switch_frac_per_machine = resolve_switch_frac_per_machine(switch_frac, M)

    machine_params = []
    for N, alpha_m, sf_m in zip(N_list, alpha_per_machine, switch_frac_per_machine):
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + cfg.C_width)
        K = alpha_m * C
        cbar_ = c_bar(K, p[0], p[1], beta)    # per machine c_bar (depends on K, p, beta)
        C_switch = sf_m * cbar_               # the REAL per machine switching cost (multiplier), 
                                              # machines can independently be inside (sf_m < 1) or
                                              # outside (sf_m >= 1) their own domain
        machine_params.append(dict(N=N, p=p, C=C, K=K, C_switch=C_switch,
                                    c_bar=cbar_, alpha_m=alpha_m, switch_frac_m=sf_m))

    machine_locals = []
    labels_by_machine = []
    methods_aware_by_machine = []
    methods_unaware_by_machine = []
    W_aware = {method: [] for method in INDEX_METHODS}
    W_unaware = {method: [] for method in INDEX_METHODS}
    indexability_warnings = []
    equivalence_warnings = []

    for m_idx, mp in enumerate(machine_params):
        P0, P1, c0, c1, labels = build_extended_mdp(
            mp["N"], np.append(mp["p"], 0.0), mp["C"], mp["K"], mp["C_switch"])
        machine_locals.append(dict(P0=P0, P1=P1, c0=c0, c1=c1))
        labels_by_machine.append(labels)

        if mp["C_switch"] >= mp["c_bar"]:
            indexability_warnings.append(
                f"machine {m_idx+1}: C_switch={mp['C_switch']:.3f} >= c_bar={mp['c_bar']:.3f} "
                f"(switch_frac_m={mp['switch_frac_m']:.3f}) -- "
                f"ClosedForm/Theorem indexability not guaranteed here.")

        mp_aware = mp
        mp_unaware = dict(mp)
        mp_unaware["C_switch"] = 0.0

        methods_aware, _ = all_method_indices(mp_aware, beta)
        methods_unaware, _ = all_method_indices(mp_unaware, beta)
        methods_aware_by_machine.append(methods_aware)
        methods_unaware_by_machine.append(methods_unaware)

        for method in INDEX_METHODS:
            W_aware[method].append(methods_aware[method])
            W_unaware[method].append(methods_unaware[method])


        # EQUIVALENCE CHECK: ClosedForm and Theorem must agree numerically
        # (same index, two derivations) -- a mismatch beyond tolerance
        # signals a bug, not a modeling choice.        
        if not np.allclose(methods_aware["ClosedForm"], methods_aware["Theorem"],
                            rtol=1e-4, atol=1e-4, equal_nan=True):
            diff = np.nanmax(np.abs(methods_aware["ClosedForm"] - methods_aware["Theorem"]))
            equivalence_warnings.append(
                f"machine {m_idx+1}: ClosedForm vs Theorem differ by {diff:.4g} "
                f"at the real C_switch -- unexpected, check for a bug.")

    P_joint, C_joint, L_list, action_subsets, bitmask_to_action = \
        build_joint_mdp_hetero(machine_locals, beta, R=cfg.R)
    n_active_per_action = np.array([len(c) for c in action_subsets])

    V_opt, pi_star, _ = solve_joint_vi(P_joint, C_joint, beta)
    S = int(np.prod(L_list))

    results = {"OPT": _make_result(pi_star, V_opt, V_opt, pi_star, n_active_per_action, cfg.R)}

    for method in INDEX_METHODS:
        for tag, W_list in (("Aware", W_aware[method]), ("Unaware", W_unaware[method])):
            name = f"{method}_{tag}"
            if any(np.any(np.isnan(w)) for w in W_list):
                results[name] = None  # e.g. Package unavailable
                continue
            pi = index_policy_from_Whittle_indices(W_list, L_list, M, cfg.R, bitmask_to_action)
            V = evaluate_joint_policy(pi, P_joint, C_joint, beta)
            results[name] = _make_result(pi, V, V_opt, pi_star, n_active_per_action, cfg.R)

    pi_greedy = greedy_policy_hetero(L_list, M, N_list, cfg.R, bitmask_to_action)
    V_greedy = evaluate_joint_policy(pi_greedy, P_joint, C_joint, beta)
    results["Greedy"] = _make_result(pi_greedy, V_greedy, V_opt, pi_star, n_active_per_action, cfg.R)

    log_info = dict(
        N_list=N_list, M=M, machine_params=machine_params,
        labels_by_machine=labels_by_machine,
        methods_aware_by_machine=methods_aware_by_machine,
        methods_unaware_by_machine=methods_unaware_by_machine,
        indexability_warnings=indexability_warnings,
        equivalence_warnings=equivalence_warnings,
        L_list=L_list, action_subsets=action_subsets,
        S=S, pi_star=pi_star, V_opt=V_opt,
        P_joint=P_joint, C_joint=C_joint, instance_seed=instance_seed,
        alpha_label=alpha, switch_frac_label=switch_frac,
    )

    if verbose:
        _log_instance_uniform(D, alpha, switch_frac, cfg, log_info, results)

    return results, log_info


def replay_instance(cfg: ExperimentConfig, D, alpha, switch_frac, instance_seed, verbose=True):
    """Regenerates ONE instance exactly, e.g. from the header line
        'D=0  alpha=1.5  switch_frac=0.5 ... instance_seed=1001'
    of an old transcript.  Needs the same cfg (R, beta, N_list, C_width, ...),
    the same (D, alpha, switch_frac) cell, and the printed instance_seed."""
    return run_cell_uniform(cfg, D, alpha, switch_frac, instance_seed, verbose=verbose)


def _log_instance_uniform(D, alpha, switch_frac, cfg: ExperimentConfig, log_info, results):
    """Rich, human readable logging for one instance (goes to terminal and .txt)."""
    M = log_info["M"]
    R = cfg.R
    sf_label = switch_frac_cell_label(switch_frac)
    print("=" * 92)
    print(f"D={D}  alpha={alpha}  switch_frac={sf_label}  R={R}  M={M}  beta={cfg.beta}  "
          f"N_list={log_info['N_list']}  instance_seed={log_info['instance_seed']}")
    print("=" * 92)

    for w in log_info["indexability_warnings"]:
        print(f"  [WARNING] {w}")
    for w in log_info["equivalence_warnings"]:
        print(f"  [WARNING] {w}")

    if cfg.verbose_indices:
        for m, mp in enumerate(log_info["machine_params"]):
            inside = mp["C_switch"] < mp["c_bar"]
            print(f"\nMachine {m + 1}: N={mp['N']}  C={mp['C']:.3f}  K={mp['K']:.3f}  "
                  f"(alpha_m={mp['alpha_m']})  switch_frac_m={mp['switch_frac_m']:.3f}  "
                  f"C_switch(real)={mp['C_switch']:.3f}  c_bar={mp['c_bar']:.3f}  "
                  f"[{'INSIDE' if inside else 'OUTSIDE'} domain]")

            print("  -- AWARE indices (computed with the REAL C_switch above) --")
            for method in INDEX_METHODS:
                W = log_info["methods_aware_by_machine"][m][method]
                print(f"    {method:10s}: {np.round(W, 4)}")

            print("  -- UNAWARE indices (computed pretending C_switch=0) --")
            for method in INDEX_METHODS:
                W = log_info["methods_unaware_by_machine"][m][method]
                print(f"    {method:10s}: {np.round(W, 4)}")

    S = log_info["S"]
    L_list = log_info["L_list"]
    A = len(log_info["action_subsets"])
    print(f"\nJoint state space size: {S} = " + "*".join(map(str, L_list)))

    if cfg.verbose_joint_shapes:
        print(f"Joint action space size: {A}  (R={R} repairmen, M={M} machines)")
        print(f"Joint transition matrices: {[P.shape for P in log_info['P_joint']]}")
        print(f"Joint one-period cost vectors: {[c.shape for c in log_info['C_joint']]}")
        if cfg.dump_full_matrices:
            print(f"P_joint =\n{log_info['P_joint']}")
            print(f"C_joint =\n{log_info['C_joint']}")

    # ---- per-state suboptimality% + idle-repairman summary, ALL methods incl. OPT
    rows = []
    idle_cols = [f"pct_{k}_idle" for k in range(R + 1)]
    for method in UNIFORM_METHODS:
        val = results.get(method)
        if val is None:
            row = dict(Method=method, q1=np.nan, median=np.nan, q3=np.nan, max=np.nan,
                       delta_mu=np.nan, pct_matches_OPT=np.nan)
            row.update({c: np.nan for c in idle_cols})
            rows.append(row)
            continue
        subopt = val["subopt"]
        idle_pct = idle_pct_from_counts(val["idle_count"], R)
        row = dict(Method=method,
                   q1=np.percentile(subopt, 25), median=np.percentile(subopt, 50),
                   q3=np.percentile(subopt, 75), max=subopt.max(),
                   delta_mu=val["delta_mu"],
                   pct_matches_OPT=val["pct_matches_opt"])
        row.update({c: v for c, v in zip(idle_cols, idle_pct)})
        rows.append(row)
    inst_df = pd.DataFrame(rows)
    print("\n" + "-" * 92)
    print("THIS INSTANCE -- cost suboptimality (%) and repairman idleness")
    print("-" * 92)
    print("  q1/median/q3/max : per-state delta_pi(s) = 100 (V_pi(s)-V*(s))/V*(s)")
    print("  delta_mu         : ratio of means Delta_mu(pi) = 100 (J(pi)-J(pi*))/J(pi*), "
          "J = avg_s V (uniform start)")
    print(f"  pct_k_idle       : % of the {S} joint states in which exactly k of the R={R} repairmen "
          f"are idle\n                     (repairmen are interchangeable -> only the COUNT of idle "
          f"repairmen matters; k={R} = nobody works)")
    print(inst_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if cfg.verbose_policies:
        print("\nAction legend (action index -> machines serviced, 1-based):")
        for a, combo in enumerate(log_info["action_subsets"]):
            print(f"    action {a}: service {_format_action_combo(combo)}"
                  f"   [{len(combo)} repairmen working, {R - len(combo)} idle]")

        # ---- STATE LEGEND ------------------------------------------------
        state_str = make_state_str_fn(L_list, log_info["labels_by_machine"])
        chunk = cfg.state_print_chunk
        if cfg.print_state_legend:
            print(f"\nState legend (joint state index s -> local state of machine 1 | ... | machine {M}):")
            print("  local state (x,f): x = damage level, f = 1 if the machine was attended in the "
                  "previous period (only for x=0), else 0.")
            print(f"  Machine 1 is the slowest varying coordinate.  Local state spaces: "
                  + ", ".join(f"m{m+1}: {L_list[m]} states" for m in range(M)))
            if S > 3 * chunk:
                print(f"  (showing first {chunk}, middle {chunk}, last {chunk} of {S} states)")
            print_sampled_rows(S, chunk, lambda s: f"s={s:>7d} : {state_str(s)}")

        # ---- ORIGINAL FORM STATE -> ACTION ---------------------------------
        if cfg.print_state_action_tables:
            pi_star = log_info["pi_star"]
            action_subsets = log_info["action_subsets"]
            print(f"\nOPT policy in original form (state -> machines serviced):")
            print_sampled_rows(
                S, chunk,
                lambda s: f"s={s:>7d} : {state_str(s)}  ->  a={pi_star[s]:>3d}  "
                          f"service {_format_action_combo(action_subsets[pi_star[s]])}")

            avail = [m for m in UNIFORM_METHODS if results.get(m) is not None]
            print("\nSide-by-side action INDEX of every method (abbreviations: "
                  "CF=ClosedForm, Thm=Theorem, Pkg=Package, _A=switching-aware, _U=unaware):")
            header = f"{'s':>9s} : {'state':<{max(10, 13 * M)}s} | " + " ".join(
                f"{SHORT_NAME[m]:>7s}" for m in avail)
            print("    " + header)
            print_sampled_rows(
                S, chunk,
                lambda s: f"s={s:>7d} : {state_str(s):<{max(10, 13 * M)}s} | " + " ".join(
                    f"{results[m]['pi'][s]:>7d}" for m in avail))

        # ---- sum(V) paired with avg_s V(s) ------------------
        avg_opt = results["OPT"]["avg"]
        print(f"\nsum(V_opt) = {results['OPT']['sum']:.3f}   "
              f"avg_s V_opt(s) = sum/S = {avg_opt:.4f}")
        print("  (V is an expected discounted COST, so it is positive and smaller is better. "
              "avg_s V(s) = (1/S) * sum_s V(s) = J_mu(pi) is the expected discounted cost if the "
              "initial joint state is drawn UNIFORMLY AT RANDOM over all S joint states, i.e. "
              "min_pi (1/|S|) sum_s V^pi(s). Every avg_s V_method(s) line below is that same "
              "quantity for that method's policy; its gap vs avg_s V_opt(s) is how much MORE "
              "expected cost that policy incurs from a uniformly random start, and Delta_mu is "
              "that gap in percent of the optimal cost.)")
        for method in UNIFORM_METHODS:
            if method == "OPT":
                continue
            val = results.get(method)
            if val is None:
                print(f"sum(V_{method}) = n/a (method unavailable)")
                continue
            print(f"sum(V_{method}) = {val['sum']:.3f}   "
                  f"avg_s V_{method}(s) = {val['avg']:.4f}   "
                  f"(gap vs avg_s V_opt(s): {val['avg'] - avg_opt:+.4f};  "
                  f"Delta_mu = {val['delta_mu']:.4f}%)")

        if S <= cfg.policy_dump_max_states:
            print(f"\nOptimal (DP) policy pi_star, one action per joint state (S={S}):")
            print(f"    {log_info['pi_star'].tolist()}")
            print(f"V_opt ({S} states): {np.round(log_info['V_opt'], 3).tolist()}")
            for method in UNIFORM_METHODS:
                if method == "OPT":
                    continue
                val = results.get(method)
                if val is None:
                    continue
                print(f"{method:18s} policy pi ({S} states): {val['pi'].tolist()}")
                print(f"{method:18s} V        ({S} states): {np.round(val['V'], 3).tolist()}")
        else:
            print(f"\n(S={S} > policy_dump_max_states={cfg.policy_dump_max_states} -- "
                  f"skipping full pi/V dump; see the summary tables above)")

    # ---- duplication check, for BOTH the aware and the unaware method set
    print("\nDUPLICATION CHECK (does the joint POLICY built from two different "
          "index formulas actually match?):")
    for tag, suffix in (("AWARE", "_Aware"), ("UNAWARE", "_Unaware")):
        cf = results.get(f"ClosedForm{suffix}")
        th = results.get(f"Theorem{suffix}")
        if cf is not None and th is not None:
            print(f"  [{tag}] ClosedForm policy == Theorem policy? "
                  f"{np.array_equal(cf['pi'], th['pi'])}")
        vi = results.get(f"VI{suffix}")
        pi_ = results.get(f"PI{suffix}")
        if vi is not None and pi_ is not None:
            print(f"  [{tag}] VI policy == PI policy?              "
                  f"{np.array_equal(vi['pi'], pi_['pi'])}")
        pkg = results.get(f"Package{suffix}")
        if vi is not None and pkg is not None:
            print(f"  [{tag}] VI policy == Package policy?         "
                  f"{np.array_equal(vi['pi'], pkg['pi'])}")
        if cf is not None and vi is not None:
            n_diff = int(np.sum(cf["pi"] != vi["pi"]))
            print(f"  [{tag}] ClosedForm vs VI: differ on {n_diff}/{S} states "
                  f"({100 * n_diff / S:.1f}%)")

    # ---- sanity check: nothing beats true OPT (V_method >= V_opt for COSTS)
    print("\nSanity check V_method >= V_opt (no policy has lower cost than true OPT):")
    for method in UNIFORM_METHODS:
        if method == "OPT":
            continue
        val = results.get(method)
        if val is None:
            continue
        print(f"  {method:18s}: {bool(np.all(val['subopt'] >= -1e-6))}")
    print()


# =====================================================================
#  n_reps problem instances per cell, POOLED, then quantiled
# =====================================================================
def run_uniform_glazebrook_table(cfg: ExperimentConfig, verbose=True):
    """For every (D, alpha, switch_frac) cell: draw cfg.n_reps independent
    problem instances (instance k uses seed cfg.seed0 + k) and produce TWO DataFrames:

      subopt_df : per-STATE suboptimality% pooled across n_reps instances
                  (n_reps * S values), quantiled; plus % of states with k idle
                  repairmen (k=0..R).  All 12 methods (OPT rows are trivial in
                  subopt, but give OPT's idleness).
      avg_df    : the scalar avg_s V(s) per instance for all 12 methods and
                  Delta_mu, quantiled/averaged over the n_reps instances per cell.
    """
    R = cfg.R
    subopt_rows = []
    avg_rows = []
    instance_counter = 0
    for switch_frac in cfg.switch_fracs:
        sf_label = switch_frac_cell_label(switch_frac)
        for D in cfg.D_grid:
            for alpha in cfg.alpha_grid:
                pooled_subopt = {m: [] for m in UNIFORM_METHODS}
                pooled_idle = {m: [] for m in UNIFORM_METHODS}
                pooled_match = {m: [] for m in UNIFORM_METHODS}
                pooled_avg = {m: [] for m in UNIFORM_METHODS}
                pooled_dmu = {m: [] for m in UNIFORM_METHODS}

                for rep in range(cfg.n_reps):
                    instance_counter += 1
                    instance_seed = cfg.seed0 + instance_counter
                    print(f"\n########## switch_frac={sf_label}  D={D}  alpha={alpha}  "
                          f"rep={rep + 1}/{cfg.n_reps}  instance_seed={instance_seed} ##########")
                    results, _ = run_cell_uniform(cfg, D, alpha, switch_frac, instance_seed,
                                                   verbose=verbose)
                    for method, val in results.items():
                        if val is None:
                            continue
                        pooled_avg[method].append(val["avg"])
                        pooled_dmu[method].append(val["delta_mu"])
                        pooled_subopt[method].append(val["subopt"])
                        pooled_idle[method].append(val["idle_count"])
                        pooled_match[method].append(val["pct_matches_opt"])

                for method in UNIFORM_METHODS:
                    idle_cols = [f"pct_{k}_idle" for k in range(R + 1)]
                    if not pooled_subopt[method]:
                        row = dict(D=D, alpha=alpha, switch_frac=sf_label, Method=method,
                                   q1=np.nan, median=np.nan, q3=np.nan, max=np.nan,
                                   pct_idle=np.nan, mean_pct_matches_OPT=np.nan, n_pooled=0)
                        row.update({c: np.nan for c in idle_cols})
                        subopt_rows.append(row)
                    else:
                        subopt_all = np.concatenate(pooled_subopt[method])
                        idle_all = np.concatenate(pooled_idle[method])
                        idle_pct = idle_pct_from_counts(idle_all, R)
                        q1, med, q3 = np.percentile(subopt_all, [25, 50, 75])
                        row = dict(D=D, alpha=alpha, switch_frac=sf_label, Method=method,
                                   q1=q1, median=med, q3=q3, max=subopt_all.max(),
                                   pct_idle=idle_pct[R],       # all R repairmen idle
                                   mean_pct_matches_OPT=float(np.mean(pooled_match[method])),
                                   n_pooled=len(subopt_all))
                        row.update({c: v for c, v in zip(idle_cols, idle_pct)})
                        subopt_rows.append(row)
                        if verbose and method != "OPT":
                            idle_str = "/".join(f"{v:.1f}" for v in idle_pct)
                            print(f"  CELL SUMMARY [{method}] switch_frac={sf_label} D={D} "
                                  f"alpha={alpha}  subopt%[{q1:.3f},{med:.3f},{q3:.3f}] "
                                  f"max={subopt_all.max():.3f}  "
                                  f"idle-repairmen% (k=0..{R}) = {idle_str}  "
                                  f"(pooled over {len(subopt_all)} states)")

                    vals = np.array(pooled_avg[method], dtype=float)
                    vals = vals[~np.isnan(vals)]
                    dmus = np.array(pooled_dmu[method], dtype=float)
                    dmus = dmus[~np.isnan(dmus)]
                    if len(vals) == 0:
                        avg_rows.append(dict(D=D, alpha=alpha, switch_frac=sf_label, Method=method,
                                              mean=np.nan, q1=np.nan, median=np.nan, q3=np.nan,
                                              max=np.nan, mean_delta_mu=np.nan, n_reps=0))
                    else:
                        aq1, amed, aq3 = np.percentile(vals, [25, 50, 75])
                        avg_rows.append(dict(
                            D=D, alpha=alpha, switch_frac=sf_label, Method=method,
                            mean=vals.mean(), q1=aq1, median=amed, q3=aq3, max=vals.max(),
                            mean_delta_mu=dmus.mean(), n_reps=len(vals),
                        ))

    return pd.DataFrame(subopt_rows), pd.DataFrame(avg_rows)


# =====================================================================
#  Aggregate printouts
# =====================================================================
def print_subopt_table_style(df: pd.DataFrame, cfg: ExperimentConfig, method: str, switch_frac_label):
    """rows=D, cols=alpha, cell=(q1,median,q3) subopt% triple / pct all-idle."""
    sub = df[(df["Method"] == method) & (df["switch_frac"] == switch_frac_label)]
    if sub.empty:
        return
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {switch_frac_label}   "
          f"(cost suboptimality %, per-state, quartile triple / pct states with ALL {cfg.R} repairmen idle)")
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


def print_idle_table_style(df: pd.DataFrame, cfg: ExperimentConfig, method: str, switch_frac_label):
    """rows=D, cols=alpha, cell = '% of states with k idle repairmen' for k=0..R."""
    sub = df[(df["Method"] == method) & (df["switch_frac"] == switch_frac_label)]
    if sub.empty:
        return
    R = cfg.R
    idle_cols = [f"pct_{k}_idle" for k in range(R + 1)]
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {switch_frac_label}   "
          f"(% of joint states with k idle repairmen, k = 0 / 1 / ... / {R}; R={R} repairmen, "
          f"interchangeable)")
    print("=" * 100)
    col_width = 6 * (R + 1) + 2
    header = "D".rjust(6) + "".join(str(a).rjust(col_width) for a in cfg.alpha_grid)
    print(header)
    for D in cfg.D_grid:
        line = str(D).rjust(6)
        for alpha in cfg.alpha_grid:
            row = sub[(sub["D"] == D) & (sub["alpha"] == alpha)]
            if row.empty or pd.isna(row[idle_cols[0]].values[0]):
                line += "n/a".rjust(col_width)
            else:
                cell = "/".join(f"{row[c].values[0]:.1f}" for c in idle_cols)
                line += cell.rjust(col_width)
        print(line)
    print()


def print_avg_table_style(df: pd.DataFrame, cfg: ExperimentConfig, method: str, switch_frac_label,
                           col="mean", what="mean of avg_s V^pi(s) (expected discounted COST, "
                                             "uniform initial state)"):
    """rows=D, cols=alpha, cell=mean over n_reps instances of `col`."""
    sub = df[(df["Method"] == method) & (df["switch_frac"] == switch_frac_label)]
    if sub.empty:
        return
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {switch_frac_label}   ({what} over n_reps instances)")
    print("=" * 100)
    col_width = 16
    header = "D".rjust(6) + "".join(str(a).rjust(col_width) for a in cfg.alpha_grid)
    print(header)
    for D in cfg.D_grid:
        line = str(D).rjust(6)
        for alpha in cfg.alpha_grid:
            row = sub[(sub["D"] == D) & (sub["alpha"] == alpha)]
            if row.empty or pd.isna(row[col].values[0]):
                line += "n/a".rjust(col_width)
            else:
                line += f"{row[col].values[0]:.3f}".rjust(col_width)
        print(line)
    print()


def main():
    # -----------------------------------------------------------------
    #   True  -> a fresh, randomly drawn M/N_list/R every run. The meta_seed is
    #            printed: pass it back as meta_seed=... to replay exactly.
    #   False -> the fully pinned, manually written PINNED_CFG below.
    # -----------------------------------------------------------------
    USE_RANDOM_EXPERIMENT = False

    if USE_RANDOM_EXPERIMENT:
        cfg = make_random_config(
            M_range=(2, 5),
            N_range=(4, 8),
            R_max=None,
            meta_seed=None,          # <- put the printed meta_seed here to replay a past run
            beta=0.95,
            n_reps=10,
            D_grid=[0, 10, 20, 25, 30, 40, 50],
            alpha_grid=[1.5, 2.0, 3.0, 4.0, 5.0],
            C_width=25.0,
            switch_fracs=[0.0, 0.5, 1.2, 3.0],
            verbose_indices=True,
            verbose_policies=True,
            verbose_joint_shapes=True,
            dump_full_matrices=False,
            policy_dump_max_states=10000,
        )
    else:
        cfg = ExperimentConfig(
            N_list=[7, 7, 7, 7],  # 4 machines, 7 damage levels each (9 local states each)
            R=2,                   # 2 repairmen
            beta=0.95,
            n_reps=3,
            D_grid=[0, 10, 20, 50],
            alpha_grid=[1.5, 4.0, "Various"],
            various_alpha_multipliers=[1.5, 3.0, 4.5, 6.0],
            C_width=25.0,
            switch_fracs=[0.5, [0.3, 0.6, 0.8, 0.2], [0.3, 0.6, 0.8, 1.2], 1.2],
            seed0=1000,
            verbose_indices=True,
            verbose_policies=True,
            verbose_joint_shapes=True,
            dump_full_matrices=False,
            policy_dump_max_states=10000,
            print_state_legend=False,  # False
            print_state_action_tables=True,
            state_print_chunk=100,
        )

    old_stdout = sys.stdout
    logfile = open(cfg.output_txt, "w", encoding="utf-8")
    sys.stdout = TeeWriter(old_stdout, logfile)
    try:
        print(f"Run started: {datetime.now().isoformat()}")
        print(f"python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__}")
        print(f"markovianbandit package available: {PKG_AVAILABLE}")
        print(f"Config: {cfg}")
        print(f"REPRODUCIBILITY: instance k uses seed = seed0 + k = {cfg.seed0} + k. "
              f"Use replay_instance(cfg, D, alpha, switch_frac, instance_seed) to regenerate one instance.\n")

        subopt_df, avg_df = run_uniform_glazebrook_table(cfg, verbose=True)

        print("\n\n" + "=" * 100)
        print("AGGREGATE TABLES -- per-state cost suboptimality % (pooled quartiles)")
        print("=" * 100)
        for method in UNIFORM_METHODS:
            if method == "OPT":
                continue
            for switch_frac in cfg.switch_fracs:
                print_subopt_table_style(subopt_df, cfg, method, switch_frac_cell_label(switch_frac))

        print("\n\n" + "=" * 100)
        print(f"AGGREGATE TABLES -- idle repairmen: % of joint states with k = 0..{cfg.R} idle repairmen")
        print("=" * 100)
        for method in UNIFORM_METHODS:
            for switch_frac in cfg.switch_fracs:
                print_idle_table_style(subopt_df, cfg, method, switch_frac_cell_label(switch_frac))

        print("\n\n" + "=" * 100)
        print("AGGREGATE TABLES -- avg_s V(s) (expected discounted cost) under a uniform initial state")
        print("=" * 100)
        for method in UNIFORM_METHODS:
            for switch_frac in cfg.switch_fracs:
                print_avg_table_style(avg_df, cfg, method, switch_frac_cell_label(switch_frac))

        print("\n\n" + "=" * 100)
        print("AGGREGATE TABLES -- ratio of means Delta_mu(pi) = 100 (J(pi)-J(pi*))/J(pi*)  [%]")
        print("=" * 100)
        for method in UNIFORM_METHODS:
            if method == "OPT":
                continue
            for switch_frac in cfg.switch_fracs:
                print_avg_table_style(avg_df, cfg, method, switch_frac_cell_label(switch_frac),
                                       col="mean_delta_mu",
                                       what="mean Delta_mu(pi) in %")

        print(f"\nRun finished: {datetime.now().isoformat()}")
        print(f"Transcript written to: {cfg.output_txt}")

    finally:
        sys.stdout = old_stdout
        logfile.close()


if __name__ == "__main__":
    main()
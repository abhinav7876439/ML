"""
Heterogeneous multi-machine (R repairmen) maintenance problem  --  COST MINIMISATION.

Whittle index computed several ways per machine:
    ClosedForm, Theorem  : only valid for CONSTANT K and C (derived for that case)
    VI, PI               : bisection on the activation tax w (any cost profile)
    Package              : markovianbandit (if installed; any cost profile)
Each index method gives a switching-AWARE policy (real C_switch) and a
switching-UNAWARE policy (C_switch = 0).  All are evaluated on the TRUE joint
system against OPT (full joint value iteration) and a non-index Greedy policy.

-------------------
 * STATE-DEPENDENT breakdown cost K(x) and repair cost C(x):
       K(x) = K0 * m_K(x),   C(x) = C0 * m_C(x)
   with profiles const / linear / convex / concave / step / decreasing and a
   growth parameter g (g=1 -> cost doubles at x=N).
   ClosedForm / Theorem are set to NaN (-> "n/a" in tables) when state dependent.
 * Numerical INDEXABILITY check (replaces the c_bar bound when state dependent).
 * Several base-cost generators (cost_scheme): paper, random_ratio,
   independent, lognormal, fixed.
 * Optional profile normalisation (same mean multiplier -> isolates SHAPE from SCALE).
 * A sweep driver over many cost models with paired seeds (same seed0 -> same p, C0).

Outputs per instance / cell (unchanged):
  (a) per-STATE suboptimality delta_pi(s) = 100 (V_pi(s)-V*(s))/V*(s), pooled, quartiles
  (b) J_mu(pi) = avg_s V_pi(s) and Delta_mu(pi) = 100 (J(pi)-J(pi*))/J(pi*)
  (c) states where actions DIFFER (aware vs unaware, OPT vs method, all methods)

Run:  python repairman_whittle_experiments.py            (sweep over COST_MODELS)
Edit RUN_MODE in main() for "single" or "random" experiments.
"""

import sys
import platform
import secrets
from datetime import datetime
from dataclasses import dataclass, field, replace
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
    """Returns P0, P1, c0, c1 (one period COSTS) and local state labels.
    C, K: scalar OR array of length N+2 indexed by LOCAL state
    (order: (0,1),(0,0),(1,0),...,(N,0)).
    p: length N+1 (p[N] = 0 appended by the caller).

    Active (a=1) cost:  C(x)            from state (0,1) (attended last period),
                        C(x) + C_switch from every other state.
    Passive (a=0) cost: expected breakdown cost  K(x) * (1 - p_x)."""
    num_states = N + 2
    C_vec = np.broadcast_to(np.asarray(C, dtype=float), (num_states,))
    K_vec = np.broadcast_to(np.asarray(K, dtype=float), (num_states,))
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    c0 = np.zeros(num_states)
    c1 = np.zeros(num_states)
    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N + 1)]
    for s in range(num_states):
        P1[s, 0] = 1.0
        c1[s] = C_vec[s] + (0.0 if s == 0 else C_switch)
    for s in range(num_states):
        if s in (0, 1):
            P0[s, 2] = p[0]
            P0[s, 1] = 1 - p[0]
            c0[s] = K_vec[s] * (1 - p[0])
        else:
            idx = s - 1
            if idx < N:
                P0[s, s + 1] = p[idx]
                P0[s, 1] = 1 - p[idx]
                c0[s] = K_vec[s] * (1 - p[idx])
            else:
                P0[s, 1] = 1.0
                c0[s] = K_vec[s]
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
        V_new = np.minimum(Q0, Q1)
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
                             w_min=-2000.0, w_max=2000.0, tol_w=1e-6, tol_mdp=1e-9):
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


def check_indexability(P0, P1, c0, c1, W, beta, gap=1e-4):
    """Numerical indexability check for an index vector W (cost-min, tax w on active).
    Indexable  <=>  for every w between consecutive distinct indices the optimal
    active set equals {s : W_s > w}.  Returns (ok, first_violating_w)."""
    Wu = np.unique(np.round(W, 6))
    cuts = ([Wu[0] - 1.0]
            + [(a + b) / 2 for a, b in zip(Wu[:-1], Wu[1:]) if b - a > gap]
            + [Wu[-1] + 1.0])
    for w in cuts:
        _, Q0, Q1 = solve_pi_discounted(P0, P1, c0, c1, w, beta)
        if not np.array_equal(Q1 < Q0 - 1e-9, W > w):
            return False, float(w)
    return True, None


# =====================================================================
# Closed form (two equivalent derivations) (CONSTANT K, C)
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
    """W as length-(N+2) array in build_extended_mdp state order. K, C scalars."""
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
    W = np.zeros(N + 2)
    W[0] = W01
    W[1] = W00
    for k in range(1, N + 1):
        pk = p[k] if k < N else 0.0
        Gk, Gk1 = G[k], G[k + 1]
        num = 1 - Gk1 - pk * (1 - beta * Gk)
        den = 1 - Gk1 - beta * pk * (1 - Gk)
        W[1 + k] = K * num / den - C - C_switch
    return W


def c_bar(K, p0, p1, beta):
    """Indexability bound for the CONSTANT cost closed form: valid for
    0 <= C_switch < c_bar.  With state dependent costs it is only a reference scale."""
    return K * (p0 - p1) / (1 + beta * (p0 - p1))


# =====================================================================
# All index arrays for one machine
# =====================================================================
def all_method_indices(mp, beta):
    N, p, C, K, C_switch = mp["N"], mp["p"], mp["C"], mp["K"], mp["C_switch"]
    P0, P1, c0, c1, labels = build_extended_mdp(N, np.append(p, 0.0), C, K, C_switch)
    S = N + 2

    if mp.get("state_dep", False):      # closed forms only derived for constant K, C
        W_cf = np.full(S, np.nan)
        W_th = np.full(S, np.nan)
    else:
        W_cf = closed_form_indices(p, beta, mp["K0"], mp["C0"], C_switch)
        W_th = theorem_indices(p, beta, mp["K0"], mp["C0"], C_switch)
    W_vi = np.array([whittle_index_discounted(s, 'vi', beta, P0, P1, c0, c1) for s in range(S)])
    W_pi = np.array([whittle_index_discounted(s, 'pi', beta, P0, P1, c0, c1) for s in range(S)])

    W_pkg = np.full(S, np.nan)
    if PKG_AVAILABLE:
        try:   # package is REWARD based: reward = -cost
            model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, -c0, -c1)
            W_pkg = np.array(model.whittle_indices(discount=beta))
        except Exception as e:
            print(f"  [WARNING] Package failed: {e}")
    return {"ClosedForm": W_cf, "VI": W_vi, "PI": W_pi, "Package": W_pkg, "Theorem": W_th}, labels


# =====================================================================
#  Heterogeneous joint MDP (Kronecker structure) + policies
# =====================================================================
def enumerate_actions(M, R):
    """Feasible joint actions: every subset of machines of size 0..min(R,M)."""
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
    """Joint transition = Kronecker product of local P (P1 if serviced else P0);
    joint cost = sum of local costs.  Machine 1 is the slowest varying coordinate.
    Returns P_joint, C_joint, L_list, action_subsets, bitmask_to_action."""
    M = len(machine_locals)
    L_list = [machine_locals[m]["P0"].shape[0] for m in range(M)]
    action_subsets, bitmask_to_action = enumerate_actions(M, R)
    P_joint, C_joint = [], []
    for combo in action_subsets:
        combo_set = set(combo)
        mats = [machine_locals[m]["P1"] if m in combo_set else machine_locals[m]["P0"]
                for m in range(M)]
        Pa = mats[0]
        for mat in mats[1:]:
            Pa = np.kron(Pa, mat)
        P_joint.append(Pa)
        Cgrid = np.zeros(tuple(L_list))
        for m in range(M):
            c_m = machine_locals[m]["c1"] if m in combo_set else machine_locals[m]["c0"]
            bshape = [1] * M
            bshape[m] = L_list[m]
            Cgrid = Cgrid + c_m.reshape(bshape)
        C_joint.append(Cgrid.reshape(-1))
    return P_joint, C_joint, L_list, action_subsets, bitmask_to_action


def solve_joint_vi(P_joint, C_joint, beta, tol=1e-8, max_iter=5000):
    """Full value iteration (MINIMISING cost) -> V_OPT and the optimal joint policy."""
    S = len(C_joint[0])
    A = len(C_joint)
    V = np.zeros(S)
    for _ in range(max_iter):
        Qs = np.stack([C_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
        V_new = Qs.min(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Qs = np.stack([C_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
    return V, Qs.argmin(axis=0), Qs


def evaluate_joint_policy(pi, P_joint, C_joint, beta):
    """Exact expected discounted COST of a fixed deterministic joint policy."""
    S = len(C_joint[0])
    P_pi = np.zeros((S, S))
    c_pi = np.zeros(S)
    for a in range(len(C_joint)):
        mask = (pi == a)
        P_pi[mask, :] = P_joint[a][mask, :]
        c_pi[mask] = C_joint[a][mask]
    return np.linalg.solve(np.eye(S) - beta * P_pi, c_pi)


def index_policy_from_Whittle_indices(W_list, L_list, M, R, bitmask_to_action):
    """Service the (up to R) machines with the largest POSITIVE index."""
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
    """Local state index (0..L-1) -> damage level.  Layout:
    [(0,1),(0,0),(1,0),...,(N,0)] -> levels [0,0,1,...,N]."""
    levels = np.zeros(L, dtype=int)
    for s in range(2, L):
        levels[s] = s - 1
    return levels


def greedy_policy_hetero(L_list, M, N_list, R, bitmask_to_action):
    """Always service the top min(R,M) machines by CURRENT damage level (never idles).
    Ties broken by lowest machine index (such fully tied states are mostly
    hypothetical initial states, see earlier notes)."""
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
    """Diagnostic: fraction of joint states with a damage-level tie at the R-th cutoff."""
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
        is_tie = sorted_desc[R_eff - 1, :] == sorted_desc[R_eff, :]
    else:
        is_tie = np.zeros(lvl_stack.shape[1], dtype=bool)
    return is_tie.mean()


def sample_glazebrook_p(N, rng):
    return np.sort(rng.uniform(0.0, 1.0, N))[::-1]


# =====================================================================
# State dependent cost profiles and base cost generators
# =====================================================================
PROFILES = ["const", "linear", "convex", "concave", "step", "decreasing"]


def cost_profile(kind, N, g, normalise=False):
    """Multiplier m(x) per LOCAL state (length N+2); m=1 at damage 0 (unless normalised).
    g = relative change at x=N (g=1 -> doubles).  If normalise, m is divided by its
    mean over the local states so the average cost level matches the baseline
    (separates SHAPE from SCALE)."""
    L = N + 2
    x = damage_level(L, N) / N
    if kind == "const":
        m = np.ones(L)
    elif kind == "linear":
        m = 1 + g * x
    elif kind == "convex":
        m = 1 + g * x ** 2
    elif kind == "concave":
        m = 1 + g * np.sqrt(x)
    elif kind == "step":
        m = 1 + g * (x >= 0.5)
    elif kind == "decreasing":
        m = 1 - g * x
        if np.any(m <= 0):
            raise ValueError("decreasing profile needs g < 1")
    else:
        raise ValueError(f"unknown profile {kind!r}")
    if normalise:
        m = m / m.mean()
    return m


COST_SCHEMES = ["paper", "random_ratio", "independent", "lognormal", "fixed"]


def draw_base_costs(scheme, D, alpha_m, width, rng):
    """Base (damage-level-0) repair cost C0 and breakdown cost K0."""
    if scheme == "paper":              # C~U(D,D+w), K=alpha*C   (original RNG order)
        C = rng.uniform(D, D + width)
        K = alpha_m * C
    elif scheme == "random_ratio":     # K/C ~ U(1, 2*alpha-1), mean alpha
        C = rng.uniform(D, D + width)
        K = C * rng.uniform(1.0, 2 * alpha_m - 1.0)
    elif scheme == "independent":      # K decoupled from C
        C = rng.uniform(D, D + width)
        K = alpha_m * rng.uniform(D, D + width)
    elif scheme == "lognormal":        # heavy-tailed costs
        C = (D + width / 2) * rng.lognormal(0.0, 0.5)
        K = alpha_m * C
    elif scheme == "fixed":            # no cost randomness, only p random
        C = D + width / 2
        K = alpha_m * C
    else:
        raise ValueError(f"unknown cost_scheme {scheme!r}")
    return float(C), float(K)


# =====================================================================
# Dual terminal + file output
# =====================================================================
class TeeWriter:
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
VARIOUS_ALPHA_MULTIPLIERS = [1.5, 3.0, 4.5]   # paper's "Various" column (M=3 only)

AlphaSpec = Union[float, str, List[float], Tuple[float, ...]]
VariousMultipliersSpec = Union[List[float], Tuple[float, ...], Callable[[int], List[float]]]


def resolve_alpha_per_machine(alpha: AlphaSpec, M: int,
                              various_multipliers: Optional[List[float]] = None) -> List[float]:
    if isinstance(alpha, str):
        if alpha != "Various":
            raise ValueError(f"Unknown alpha spec: {alpha!r}")
        if various_multipliers is None or len(various_multipliers) != M:
            raise ValueError(
                f'"Various" alpha needs a length-M={M} list of multipliers, got '
                f"{various_multipliers!r}. Set cfg.various_alpha_multipliers.")
        return [float(a) for a in various_multipliers]
    if isinstance(alpha, (list, tuple)):
        if len(alpha) != M:
            raise ValueError(f"alpha list length {len(alpha)} != M={M}")
        return [float(a) for a in alpha]
    return [float(alpha)] * M


SwitchFracSpec = Union[float, List[float], Tuple[float, ...]]


def resolve_switch_frac_per_machine(switch_frac: SwitchFracSpec, M: int) -> List[float]:
    if isinstance(switch_frac, (list, tuple)):
        if len(switch_frac) != M:
            raise ValueError(f"switch_frac list length {len(switch_frac)} != M={M}")
        return [float(s) for s in switch_frac]
    return [float(switch_frac)] * M


def switch_frac_cell_label(switch_frac):
    if isinstance(switch_frac, (list, tuple)):
        return "mixed_" + "-".join(f"{float(s):g}" for s in switch_frac)
    return switch_frac


@dataclass
class ExperimentConfig:
    N_list: Optional[List[int]] = field(default_factory=lambda: [6, 6, 6])
    M: int = 3                          # DERIVED from N_list in __post_init__
    N_range: Tuple[int, int] = (3, 6)   # used only if N_list is None
    M_range: Tuple[int, int] = (2, 5)   # used only if N_list is None

    R: int = 1
    beta: float = 0.95
    n_reps: int = 10

    D_grid: List[float] = field(default_factory=lambda: [0, 10, 20, 25, 30, 40, 50])
    alpha_grid: List[AlphaSpec] = field(
        default_factory=lambda: [1.5, 2.0, 3.0, 4.0, 5.0, "Various"])
    C_width: float = 25.0
    various_alpha_multipliers: Optional[VariousMultipliersSpec] = None

    switch_fracs: List[SwitchFracSpec] = field(default_factory=lambda: [0.0])
    seed0: int = 1000

    # ---- cost structure (NEW) ----
    cost_scheme: str = "paper"          # one of COST_SCHEMES
    K_profile: str = "const"            # one of PROFILES  (breakdown cost K(x) = K0*m(x))
    K_growth: float = 0.0
    C_profile: str = "const"            # one of PROFILES  (repair cost    C(x) = C0*m(x))
    C_growth: float = 0.0
    normalise_profiles: bool = False    # divide each profile by its mean (shape vs scale)

    verbose_indices: bool = True
    verbose_policies: bool = True
    verbose_joint_shapes: bool = True
    dump_full_matrices: bool = False
    policy_dump_max_states: int = 10000

    print_state_legend: bool = True
    print_state_action_tables: bool = True
    state_print_chunk: int = 100
    print_action_diffs: bool = True

    output_txt: Optional[str] = None

    @property
    def state_dependent(self) -> bool:
        return ((self.K_profile != "const" and self.K_growth != 0) or
                (self.C_profile != "const" and self.C_growth != 0))

    def __post_init__(self):
        if self.N_list is not None:
            self.M = len(self.N_list)

        if self.cost_scheme not in COST_SCHEMES:
            raise ValueError(f"cost_scheme must be one of {COST_SCHEMES}")
        for prof in (self.K_profile, self.C_profile):
            if prof not in PROFILES:
                raise ValueError(f"profile must be one of {PROFILES}")

        if "Various" in self.alpha_grid:
            if callable(self.various_alpha_multipliers):
                mult = list(self.various_alpha_multipliers(self.M))
                if len(mult) != self.M:
                    raise ValueError("various_alpha_multipliers callable must return M values.")
                self.various_alpha_multipliers = mult
            elif self.various_alpha_multipliers is not None:
                if len(self.various_alpha_multipliers) != self.M:
                    raise ValueError(
                        f"cfg.various_alpha_multipliers has length "
                        f"{len(self.various_alpha_multipliers)} but M={self.M}.")
                self.various_alpha_multipliers = list(self.various_alpha_multipliers)
            elif self.M == 3:
                self.various_alpha_multipliers = list(VARIOUS_ALPHA_MULTIPLIERS)
            else:
                raise ValueError(
                    f'alpha_grid contains "Various" but M={self.M} (!=3): set '
                    f"cfg.various_alpha_multipliers explicitly.")

        if self.output_txt is None:
            now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_txt = f"Discounted_cost_repairman_results_{now_str}.txt"


def resolve_N_list(cfg: ExperimentConfig, rng: np.random.Generator) -> List[int]:
    if cfg.N_list is not None:
        return list(cfg.N_list)
    lo, hi = cfg.N_range
    return [int(rng.integers(lo, hi + 1)) for _ in range(cfg.M)]


def make_random_config(*, M_range=(2, 5), N_range=(3, 8), R_max=None,
                       various_alpha_multipliers=None, seed0=None, meta_seed=None,
                       **overrides) -> ExperimentConfig:
    """Fresh ExperimentConfig with random M, N_list, R (and seed0), all determined
    by meta_seed (printed; pass it back to replay)."""
    if meta_seed is None:
        meta_seed = secrets.randbits(31)
    meta_rng = np.random.default_rng(meta_seed)
    M = int(meta_rng.integers(M_range[0], M_range[1] + 1))
    N_list = [int(meta_rng.integers(N_range[0], N_range[1] + 1)) for _ in range(M)]
    R = int(meta_rng.integers(1, (R_max or M) + 1))
    resolved_seed0 = seed0 if seed0 is not None else int(meta_rng.integers(0, 2 ** 31 - 1))
    cfg = ExperimentConfig(N_list=N_list, R=R,
                           various_alpha_multipliers=various_alpha_multipliers,
                           seed0=resolved_seed0, **overrides)
    print(f"[random experiment] meta_seed={meta_seed} -> M={M}, N_list={N_list}, "
          f"R={R}, seed0={resolved_seed0}")
    return cfg


# =====================================================================
# {Aware, Unaware} method bookkeeping
# =====================================================================
INDEX_METHODS = ["ClosedForm", "VI", "PI", "Package", "Theorem"]

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
    if not combo:
        return "(idle)"
    one_based = tuple(m + 1 for m in combo)
    noun = "machine" if len(one_based) == 1 else "machines"
    return f"{noun} {one_based}"


def _act_str(a, action_subsets):
    return f"a={int(a):>3d}  service {_format_action_combo(action_subsets[int(a)])}"


# ---------------------------------------------------------------------
# sampled state printing
# ---------------------------------------------------------------------
def sampled_state_blocks(S, chunk):
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


def print_sampled_list(idx_list, chunk, row_fn, indent="    "):
    """Sample first/middle/last `chunk` entries of an explicit list of state indices."""
    n = len(idx_list)
    if n == 0:
        print(f"{indent}(none)")
        return
    if chunk <= 0 or n <= 3 * chunk:
        blocks = [(0, n)]
    else:
        mid = n // 2 - chunk // 2
        blocks = [(0, chunk), (mid, mid + chunk), (n - chunk, n)]
    prev_end = None
    for lo, hi in blocks:
        if prev_end is not None and lo > prev_end:
            print(f"{indent}   ...   ({lo - prev_end} differing states not shown)")
        for s in idx_list[lo:hi]:
            print(indent + row_fn(int(s)))
        prev_end = hi


def make_state_str_fn(L_list, labels_by_machine):
    shape = tuple(L_list)

    def state_str(s):
        idx = np.unravel_index(s, shape)
        return "[" + " | ".join(labels_by_machine[m][idx[m]] for m in range(len(L_list))) + "]"
    return state_str


def idle_pct_from_counts(idle_count, R):
    return 100.0 * np.bincount(idle_count, minlength=R + 1)[:R + 1] / len(idle_count)


# ---------------------------------------------------------------------
# action-difference printouts
# ---------------------------------------------------------------------
def print_aware_vs_unaware_diffs(results, log_info, cfg, state_str):
    S = log_info["S"]
    action_subsets = log_info["action_subsets"]
    chunk = cfg.state_print_chunk
    print("\n" + "-" * 92)
    print("STATES WHERE AWARE and UNAWARE ACTIONS DIFFER (per method)")
    print("-" * 92)
    for method in INDEX_METHODS:
        A_ = results.get(f"{method}_Aware")
        U_ = results.get(f"{method}_Unaware")
        if A_ is None or U_ is None:
            print(f"\n[{method}] n/a (method unavailable)")
            continue
        diff = np.where(A_["pi"] != U_["pi"])[0]
        print(f"\n[{method}] Aware vs Unaware differ on {len(diff)}/{S} states "
              f"({100.0 * len(diff) / S:.2f}%)")
        if len(diff) == 0:
            continue
        print_sampled_list(
            diff, chunk,
            lambda s: f"s={s:>7d} : {state_str(s)}\n"
                      f"{'':13s}aware   -> {_act_str(A_['pi'][s], action_subsets)}\n"
                      f"{'':13s}unaware -> {_act_str(U_['pi'][s], action_subsets)}")


def print_cross_method_diffs(results, log_info, cfg, state_str):
    S = log_info["S"]
    action_subsets = log_info["action_subsets"]
    chunk = cfg.state_print_chunk
    pi_opt = results["OPT"]["pi"]

    print("\n" + "-" * 92)
    print("CROSS COMPARISON (i): OPTIMAL action vs each method, only states where they DIFFER")
    print("-" * 92)
    for method in UNIFORM_METHODS:
        if method == "OPT":
            continue
        val = results.get(method)
        if val is None:
            continue
        diff = np.where(val["pi"] != pi_opt)[0]
        print(f"\n[{method}] differs from OPT on {len(diff)}/{S} states "
              f"({100.0 * len(diff) / S:.2f}%)")
        if len(diff) == 0:
            continue
        print_sampled_list(
            diff, chunk,
            lambda s: f"s={s:>7d} : {state_str(s)}\n"
                      f"{'':13s}OPT      -> {_act_str(pi_opt[s], action_subsets)}\n"
                      f"{'':13s}{method:<9s}-> {_act_str(val['pi'][s], action_subsets)}")

    avail = [m for m in UNIFORM_METHODS if results.get(m) is not None]
    pi_mat = np.stack([results[m]["pi"] for m in avail], axis=0)
    disagree = np.where((pi_mat != pi_mat[0:1, :]).any(axis=0))[0]
    print("\n" + "-" * 92)
    print(f"CROSS COMPARISON (ii): all methods side by side, only states where NOT all agree "
          f"({len(disagree)}/{S} states, {100.0 * len(disagree) / S:.2f}%)")
    print("-" * 92)
    if len(disagree) == 0:
        print("    (all methods agree on every state)")
        return
    w = max(10, 13 * log_info["M"])
    print("    " + f"{'s':>9s} : {'state':<{w}s} | " + " ".join(f"{SHORT_NAME[m]:>7s}" for m in avail))
    print_sampled_list(
        disagree, chunk,
        lambda s: f"s={s:>7d} : {state_str(s):<{w}s} | "
                  + " ".join(f"{results[m]['pi'][s]:>7d}" for m in avail))


def _make_result(pi, V, V_opt, pi_star, n_active_per_action, R):
    idle_count = R - n_active_per_action[pi]
    return dict(
        subopt=100.0 * (V - V_opt) / V_opt,
        idle_count=idle_count,
        pi=pi, V=V, avg=V.mean(), sum=V.sum(),
        delta_mu=100.0 * (V.mean() - V_opt.mean()) / V_opt.mean(),
        pct_matches_opt=100.0 * np.mean(pi == pi_star),
    )


# =====================================================================
# One problem instance
# =====================================================================
def run_cell_uniform(cfg: ExperimentConfig, D, alpha, switch_frac, instance_seed, verbose=True):
    """Draws ONE random instance determined by instance_seed.  Returns
    (results, log_info); results[method] is None if the method is unavailable
    (e.g. ClosedForm/Theorem when costs are state dependent)."""
    rng = np.random.default_rng(instance_seed)
    N_list = resolve_N_list(cfg, rng)
    M = len(N_list)
    beta = cfg.beta
    alpha_per_machine = resolve_alpha_per_machine(alpha, M, cfg.various_alpha_multipliers)
    switch_frac_per_machine = resolve_switch_frac_per_machine(switch_frac, M)

    machine_params = []
    for N, alpha_m, sf_m in zip(N_list, alpha_per_machine, switch_frac_per_machine):
        p = sample_glazebrook_p(N, rng)
        C0, K0 = draw_base_costs(cfg.cost_scheme, D, alpha_m, cfg.C_width, rng)
        C_vec = C0 * cost_profile(cfg.C_profile, N, cfg.C_growth, cfg.normalise_profiles)
        K_vec = K0 * cost_profile(cfg.K_profile, N, cfg.K_growth, cfg.normalise_profiles)
        cbar_ = c_bar(K0, p[0], p[1], beta)       # reference scale only if state dependent
        C_switch = sf_m * cbar_
        machine_params.append(dict(N=N, p=p, C=C_vec, K=K_vec, C0=C0, K0=K0,
                                   C_switch=C_switch, c_bar=cbar_, alpha_m=alpha_m,
                                   switch_frac_m=sf_m, state_dep=cfg.state_dependent))

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
            extra = " (reference c_bar; closed forms not applicable)" if mp["state_dep"] else ""
            indexability_warnings.append(
                f"machine {m_idx+1}: C_switch={mp['C_switch']:.3f} >= c_bar={mp['c_bar']:.3f} "
                f"(switch_frac_m={mp['switch_frac_m']:.3f}){extra} -- "
                f"ClosedForm/Theorem indexability not guaranteed here.")

        mp_unaware = dict(mp)
        mp_unaware["C_switch"] = 0.0

        methods_aware, _ = all_method_indices(mp, beta)
        methods_unaware, _ = all_method_indices(mp_unaware, beta)
        methods_aware_by_machine.append(methods_aware)
        methods_unaware_by_machine.append(methods_unaware)

        for method in INDEX_METHODS:
            W_aware[method].append(methods_aware[method])
            W_unaware[method].append(methods_unaware[method])

        # ---- numerical indexability check (aware & unaware) ----
        ok_a, w_a = check_indexability(P0, P1, c0, c1, methods_aware["VI"], beta)
        P0u, P1u, c0u, c1u, _ = build_extended_mdp(
            mp["N"], np.append(mp["p"], 0.0), mp["C"], mp["K"], 0.0)
        ok_u, w_u = check_indexability(P0u, P1u, c0u, c1u, methods_unaware["VI"], beta)
        if not ok_a:
            indexability_warnings.append(
                f"machine {m_idx+1}: NOT indexable (aware), first violation at w={w_a:.3f}")
        if not ok_u:
            indexability_warnings.append(
                f"machine {m_idx+1}: NOT indexable (unaware), first violation at w={w_u:.3f}")

        # ---- ClosedForm vs Theorem must agree (only for constant costs) ----
        if not mp["state_dep"]:
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
                results[name] = None
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
    """Regenerate ONE instance from a transcript header (same cfg, cell and instance_seed)."""
    return run_cell_uniform(cfg, D, alpha, switch_frac, instance_seed, verbose=verbose)


def _log_instance_uniform(D, alpha, switch_frac, cfg: ExperimentConfig, log_info, results):
    M = log_info["M"]
    R = cfg.R
    sf_label = switch_frac_cell_label(switch_frac)
    print("=" * 92)
    print(f"D={D}  alpha={alpha}  switch_frac={sf_label}  R={R}  M={M}  beta={cfg.beta}  "
          f"N_list={log_info['N_list']}  instance_seed={log_info['instance_seed']}")
    print(f"cost_scheme={cfg.cost_scheme}  K:{cfg.K_profile}(g={cfg.K_growth})  "
          f"C:{cfg.C_profile}(g={cfg.C_growth})  normalise={cfg.normalise_profiles}  "
          f"state_dependent={cfg.state_dependent}")
    print("=" * 92)

    for w in log_info["indexability_warnings"]:
        print(f"  [WARNING] {w}")
    for w in log_info["equivalence_warnings"]:
        print(f"  [WARNING] {w}")

    if cfg.verbose_indices:
        for m, mp in enumerate(log_info["machine_params"]):
            inside = mp["C_switch"] < mp["c_bar"]
            print(f"\nMachine {m + 1}: N={mp['N']}  C0={mp['C0']:.3f}  K0={mp['K0']:.3f}  "
                  f"(alpha_m={mp['alpha_m']:.3f})  switch_frac_m={mp['switch_frac_m']:.3f}  "
                  f"C_switch(real)={mp['C_switch']:.3f}  c_bar(ref)={mp['c_bar']:.3f}  "
                  f"[{'INSIDE' if inside else 'OUTSIDE'} ref domain]")
            if mp["state_dep"]:
                print(f"    K(x) = {np.round(mp['K'], 3)}")
                print(f"    C(x) = {np.round(mp['C'], 3)}")

            print("  -- AWARE indices (computed with the REAL C_switch above) --")
            for method in INDEX_METHODS:
                print(f"    {method:10s}: {np.round(log_info['methods_aware_by_machine'][m][method], 4)}")
            print("  -- UNAWARE indices (computed pretending C_switch=0) --")
            for method in INDEX_METHODS:
                print(f"    {method:10s}: {np.round(log_info['methods_unaware_by_machine'][m][method], 4)}")

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
                   delta_mu=val["delta_mu"], pct_matches_OPT=val["pct_matches_opt"])
        row.update({c: v for c, v in zip(idle_cols, idle_pct)})
        rows.append(row)
    inst_df = pd.DataFrame(rows)
    print("\n" + "-" * 92)
    print("THIS INSTANCE -- cost suboptimality (%) and repairman idleness")
    print("-" * 92)
    print("  q1/median/q3/max : per-state delta_pi(s) = 100 (V_pi(s)-V*(s))/V*(s)")
    print("  delta_mu         : Delta_mu(pi) = 100 (J(pi)-J(pi*))/J(pi*), J = avg_s V (uniform start)")
    print(f"  pct_k_idle       : % of the {S} joint states with exactly k of R={R} repairmen idle")
    print(inst_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if cfg.verbose_policies:
        print("\nAction legend (action index -> machines serviced, 1-based):")
        for a, combo in enumerate(log_info["action_subsets"]):
            print(f"    action {a}: service {_format_action_combo(combo)}"
                  f"   [{len(combo)} repairmen working, {R - len(combo)} idle]")

        state_str = make_state_str_fn(L_list, log_info["labels_by_machine"])
        chunk = cfg.state_print_chunk
        if cfg.print_state_legend:
            print(f"\nState legend (joint state index s -> local state of machine 1 | ... | machine {M}):")
            print("  local state (x,f): x = damage level, f = 1 if attended in the previous period "
                  "(only for x=0).")
            print("  Machine 1 is the slowest varying coordinate.  Local state spaces: "
                  + ", ".join(f"m{m+1}: {L_list[m]} states" for m in range(M)))
            if S > 3 * chunk:
                print(f"  (showing first {chunk}, middle {chunk}, last {chunk} of {S} states)")
            print_sampled_rows(S, chunk, lambda s: f"s={s:>7d} : {state_str(s)}")

        if cfg.print_state_action_tables:
            pi_star = log_info["pi_star"]
            action_subsets = log_info["action_subsets"]
            print("\nOPT policy in original form (state -> machines serviced):")
            print_sampled_rows(
                S, chunk,
                lambda s: f"s={s:>7d} : {state_str(s)}  ->  a={pi_star[s]:>3d}  "
                          f"service {_format_action_combo(action_subsets[pi_star[s]])}")

            avail = [m for m in UNIFORM_METHODS if results.get(m) is not None]
            print("\nSide-by-side action INDEX of every method (CF=ClosedForm, Thm=Theorem, "
                  "Pkg=Package, _A=aware, _U=unaware):")
            wd = max(10, 13 * M)
            header = f"{'s':>9s} : {'state':<{wd}s} | " + " ".join(f"{SHORT_NAME[m]:>7s}" for m in avail)
            print("    " + header)
            print_sampled_rows(
                S, chunk,
                lambda s: f"s={s:>7d} : {state_str(s):<{wd}s} | " + " ".join(
                    f"{results[m]['pi'][s]:>7d}" for m in avail))

        if cfg.print_action_diffs:
            print_aware_vs_unaware_diffs(results, log_info, cfg, state_str)
            print_cross_method_diffs(results, log_info, cfg, state_str)

        avg_opt = results["OPT"]["avg"]
        print(f"\nsum(V_opt) = {results['OPT']['sum']:.3f}   avg_s V_opt(s) = {avg_opt:.4f}")
        print("  (V is an expected discounted COST; smaller is better. avg_s V = J_mu(pi) is the "
              "expected cost from a uniformly random initial joint state; Delta_mu is its gap "
              "to OPT in percent.)")
        for method in UNIFORM_METHODS:
            if method == "OPT":
                continue
            val = results.get(method)
            if val is None:
                print(f"sum(V_{method}) = n/a (method unavailable)")
                continue
            print(f"sum(V_{method}) = {val['sum']:.3f}   avg_s V_{method}(s) = {val['avg']:.4f}   "
                  f"(gap vs OPT: {val['avg'] - avg_opt:+.4f};  Delta_mu = {val['delta_mu']:.4f}%)")

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
                  f"skipping full pi/V dump)")

    print("\nDUPLICATION CHECK (do joint POLICIES from different index formulas match?):")
    for tag, suffix in (("AWARE", "_Aware"), ("UNAWARE", "_Unaware")):
        cf = results.get(f"ClosedForm{suffix}")
        th = results.get(f"Theorem{suffix}")
        vi = results.get(f"VI{suffix}")
        pi_ = results.get(f"PI{suffix}")
        pkg = results.get(f"Package{suffix}")
        if cf is not None and th is not None:
            print(f"  [{tag}] ClosedForm policy == Theorem policy? {np.array_equal(cf['pi'], th['pi'])}")
        if vi is not None and pi_ is not None:
            print(f"  [{tag}] VI policy == PI policy?              {np.array_equal(vi['pi'], pi_['pi'])}")
        if vi is not None and pkg is not None:
            print(f"  [{tag}] VI policy == Package policy?         {np.array_equal(vi['pi'], pkg['pi'])}")
        if cf is not None and vi is not None:
            n_diff = int(np.sum(cf["pi"] != vi["pi"]))
            print(f"  [{tag}] ClosedForm vs VI: differ on {n_diff}/{S} states ({100 * n_diff / S:.1f}%)")

    print("\nSanity check V_method >= V_opt (no policy beats true OPT):")
    for method in UNIFORM_METHODS:
        if method == "OPT":
            continue
        val = results.get(method)
        if val is None:
            continue
        print(f"  {method:18s}: {bool(np.all(val['subopt'] >= -1e-6))}")
    print()


# =====================================================================
#  n_reps instances per cell, POOLED, then quantiled
# =====================================================================
def run_uniform_glazebrook_table(cfg: ExperimentConfig, verbose=True):
    """For every (switch_frac, D, alpha) cell draw cfg.n_reps instances (instance k uses
    seed cfg.seed0 + k).  Returns
      subopt_df : pooled per-state suboptimality quartiles + idleness, all methods
      avg_df    : avg_s V(s) and Delta_mu summaries over instances, all methods."""
    R = cfg.R
    subopt_rows, avg_rows = [], []
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

                idle_cols = [f"pct_{k}_idle" for k in range(R + 1)]
                for method in UNIFORM_METHODS:
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
                                   pct_idle=idle_pct[R],
                                   mean_pct_matches_OPT=float(np.mean(pooled_match[method])),
                                   n_pooled=len(subopt_all))
                        row.update({c: v for c, v in zip(idle_cols, idle_pct)})
                        subopt_rows.append(row)
                        if verbose and method != "OPT":
                            idle_str = "/".join(f"{v:.1f}" for v in idle_pct)
                            print(f"  CELL SUMMARY [{method}] switch_frac={sf_label} D={D} "
                                  f"alpha={alpha}  subopt%[{q1:.3f},{med:.3f},{q3:.3f}] "
                                  f"max={subopt_all.max():.3f}  idle% (k=0..{R}) = {idle_str}  "
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
                            mean_delta_mu=dmus.mean(), n_reps=len(vals)))

    return pd.DataFrame(subopt_rows), pd.DataFrame(avg_rows)


# =====================================================================
#  Aggregate printouts
# =====================================================================
def print_subopt_table_style(df, cfg, method, switch_frac_label):
    sub = df[(df["Method"] == method) & (df["switch_frac"] == switch_frac_label)]
    if sub.empty:
        return
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {switch_frac_label}   "
          f"(cost suboptimality %, per-state, quartile triple / pct states with ALL {cfg.R} repairmen idle)")
    print("=" * 100)
    col_width = 22
    print("D".rjust(6) + "".join(str(a).rjust(col_width) for a in cfg.alpha_grid))
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
            line1 += f"{q1:.3f} {med:.3f} {q3:.3f}".rjust(col_width)
            line2 += f"{idle:.2f}".rjust(col_width)
        print(line1)
        print(line2)
    print()


def print_idle_table_style(df, cfg, method, switch_frac_label):
    sub = df[(df["Method"] == method) & (df["switch_frac"] == switch_frac_label)]
    if sub.empty:
        return
    R = cfg.R
    idle_cols = [f"pct_{k}_idle" for k in range(R + 1)]
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {switch_frac_label}   "
          f"(% of joint states with k idle repairmen, k = 0 / ... / {R}; R={R})")
    print("=" * 100)
    col_width = 6 * (R + 1) + 2
    print("D".rjust(6) + "".join(str(a).rjust(col_width) for a in cfg.alpha_grid))
    for D in cfg.D_grid:
        line = str(D).rjust(6)
        for alpha in cfg.alpha_grid:
            row = sub[(sub["D"] == D) & (sub["alpha"] == alpha)]
            if row.empty or pd.isna(row[idle_cols[0]].values[0]):
                line += "n/a".rjust(col_width)
            else:
                line += "/".join(f"{row[c].values[0]:.1f}" for c in idle_cols).rjust(col_width)
        print(line)
    print()


def print_avg_table_style(df, cfg, method, switch_frac_label, col="mean",
                          what="mean of avg_s V^pi(s) (expected discounted COST, uniform initial state)"):
    sub = df[(df["Method"] == method) & (df["switch_frac"] == switch_frac_label)]
    if sub.empty:
        return
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {switch_frac_label}   ({what} over n_reps instances)")
    print("=" * 100)
    col_width = 16
    print("D".rjust(6) + "".join(str(a).rjust(col_width) for a in cfg.alpha_grid))
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


def print_all_tables(subopt_df, avg_df, cfg):
    # (title, printer, skip OPT?)
    blocks = [
        ("per-state cost suboptimality % (pooled quartiles)",
         lambda m, sf: print_subopt_table_style(subopt_df, cfg, m, sf), True),
        (f"idle repairmen: % of joint states with k = 0..{cfg.R} idle",
         lambda m, sf: print_idle_table_style(subopt_df, cfg, m, sf), False),
        ("avg_s V(s) (expected discounted cost), uniform initial state",
         lambda m, sf: print_avg_table_style(avg_df, cfg, m, sf), False),
        ("ratio of means Delta_mu(pi) [%]",
         lambda m, sf: print_avg_table_style(avg_df, cfg, m, sf, col="mean_delta_mu",
                                             what="mean Delta_mu(pi) in %"), True),
    ]
    for title, printer, skip_opt in blocks:
        print("\n\n" + "=" * 100 + f"\nAGGREGATE TABLES -- {title}\n" + "=" * 100)
        for method in UNIFORM_METHODS:
            if skip_opt and method == "OPT":
                continue
            for sf in cfg.switch_fracs:
                printer(method, switch_frac_cell_label(sf))


# =====================================================================
#  Self-test, single run, sweep
# =====================================================================
def selftest_numeric_vs_closed_form(beta=0.95, seed=0):
    """Constant profile: VI/PI must equal ClosedForm/Theorem (checks the refactor)."""
    rng = np.random.default_rng(seed)
    N = 6
    p = sample_glazebrook_p(N, rng)
    C = 20.0
    K = 4 * C
    cs = 0.5 * c_bar(K, p[0], p[1], beta)
    mp = dict(N=N, p=p, C=C, K=K, C0=C, K0=K, C_switch=cs, state_dep=False)
    m, _ = all_method_indices(mp, beta)
    for k in ("VI", "PI", "Theorem"):
        print(f"selftest max|{k} - ClosedForm| = {np.nanmax(np.abs(m[k] - m['ClosedForm'])):.2e}")
    if PKG_AVAILABLE:
        print(f"selftest max|Package - ClosedForm| = "
              f"{np.nanmax(np.abs(m['Package'] - m['ClosedForm'])):.2e}")


def run_one(cfg: ExperimentConfig):
    """Run a full grid for one config, tee output to cfg.output_txt."""
    old = sys.stdout
    logfile = open(cfg.output_txt, "w", encoding="utf-8")
    sys.stdout = TeeWriter(old, logfile)
    try:
        print(f"Run started: {datetime.now().isoformat()}")
        print(f"python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__}")
        print(f"markovianbandit package available: {PKG_AVAILABLE}")
        print(f"Config: {cfg}")
        print(f"state_dependent={cfg.state_dependent}  (ClosedForm/Theorem are NaN -> n/a when True)")
        print(f"REPRODUCIBILITY: instance k uses seed = seed0 + k = {cfg.seed0} + k.\n")
        subopt_df, avg_df = run_uniform_glazebrook_table(cfg, verbose=True)
        print_all_tables(subopt_df, avg_df, cfg)
        print(f"\nRun finished: {datetime.now().isoformat()}")
        print(f"Transcript written to: {cfg.output_txt}")
        return subopt_df, avg_df
    finally:
        sys.stdout = old
        logfile.close()


COST_MODELS = [
    dict(tag="const"),                                                        # baseline (= original model)
    dict(tag="Klin1",   K_profile="linear",  K_growth=1.0),
    dict(tag="Klin3",   K_profile="linear",  K_growth=3.0),
    dict(tag="Kconvex", K_profile="convex",  K_growth=3.0),
    dict(tag="Kconc",   K_profile="concave", K_growth=3.0),
    dict(tag="Kstep",   K_profile="step",    K_growth=2.0),
    dict(tag="Clin1",   C_profile="linear",  C_growth=1.0),
    dict(tag="Klin1_Clin1", K_profile="linear", K_growth=1.0, C_profile="linear", C_growth=1.0),
    dict(tag="Klin1_Clin3", K_profile="linear", K_growth=1.0, C_profile="linear", C_growth=3.0),  # C outgrows K
    dict(tag="Klin3_norm", K_profile="linear", K_growth=3.0, normalise_profiles=True),             # shape only
    dict(tag="Kdec0.5", K_profile="decreasing", K_growth=0.5),                # stress test (non-physical)
    # other cost generators, constant profiles
    dict(tag="randratio",   cost_scheme="random_ratio"),
    dict(tag="independent", cost_scheme="independent"),
    dict(tag="lognormal",   cost_scheme="lognormal"),
    dict(tag="fixed",       cost_scheme="fixed"),
]


def run_sweep(base: ExperimentConfig, cost_models=COST_MODELS):
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    all_sub, all_avg = [], []
    for cm in cost_models:
        tag = cm["tag"]
        cfg = replace(base, **{k: v for k, v in cm.items() if k != "tag"},
                      output_txt=f"Discounted_cost_repairman_{tag}_{stamp}.txt")
        sub, avg = run_one(cfg)
        sub["cost_model"] = tag
        avg["cost_model"] = tag
        all_sub.append(sub)
        all_avg.append(avg)

    sub_all = pd.concat(all_sub, ignore_index=True)
    avg_all = pd.concat(all_avg, ignore_index=True)
    sub_all.to_csv(f"subopt_all_{stamp}.csv", index=False)
    avg_all.to_csv(f"avg_all_{stamp}.csv", index=False)

    summ = (avg_all[avg_all.Method != "OPT"]
            .groupby(["cost_model", "Method"])["mean_delta_mu"].mean().unstack("Method"))
    print("\nMean Delta_mu [%] over all (D, alpha, switch_frac) cells, per cost model:")
    print(summ.round(3).to_string())
    summ.to_csv(f"summary_delta_mu_{stamp}.csv")
    return sub_all, avg_all


def main():
    # "sweep"  -> every entry of COST_MODELS with paired seeds (same seed0)
    # "single" -> one pinned config (edit below; set K_/C_ profiles freely)
    # "random" -> random M / N_list / R (meta_seed printed for replay)
    RUN_MODE = "single"

    selftest_numeric_vs_closed_form()

    if RUN_MODE == "sweep":
        base = ExperimentConfig(
            N_list=[7, 7, 7, 7], R=2, beta=0.95, n_reps=3,
            D_grid=[0, 10, 20, 50], alpha_grid=[1.5, 4.0, "Various"],
            various_alpha_multipliers=[1.5, 3.0, 4.5, 6.0], C_width=25.0,
            switch_fracs=[0.5, 1.2], seed0=1000,
            verbose_indices=True, verbose_policies=False,   # many runs -> keep logs small
            print_state_legend=False, print_state_action_tables=False,
            print_action_diffs=False,
        )
        run_sweep(base)

    elif RUN_MODE == "single":
        cfg = ExperimentConfig(
            N_list=[7, 7, 7, 7], R=2, beta=0.95, n_reps=3,
            D_grid=[0, 10, 20, 50], alpha_grid=[1.5, 4.0, "Various"],
            various_alpha_multipliers=[1.5, 3.0, 4.5, 6.0], C_width=25.0,
            switch_fracs=[0.5, [0.3, 0.6, 0.8, 0.2], [0.3, 0.6, 0.8, 1.2], 1.2],
            seed0=1000,
            cost_scheme="paper", K_profile="linear", K_growth=1.0,
            C_profile="const", C_growth=0.0,
            verbose_policies=True, print_state_legend=False,
            print_state_action_tables=True, state_print_chunk=100, print_action_diffs=True,
        )
        run_one(cfg)

    elif RUN_MODE == "random":
        cfg = make_random_config(
            M_range=(2, 5), N_range=(4, 8), meta_seed=None, beta=0.95, n_reps=10,
            D_grid=[0, 10, 20, 25, 30, 40, 50], alpha_grid=[1.5, 2.0, 3.0, 4.0, 5.0],
            C_width=25.0, switch_fracs=[0.0, 0.5, 1.2, 3.0],
            K_profile="linear", K_growth=1.0,
        )
        run_one(cfg)
    else:
        raise ValueError(RUN_MODE)


if __name__ == "__main__":
    main()
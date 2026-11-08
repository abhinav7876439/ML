"""
Machine repairman problem with switching costs -- DISCOUNTED COST, perfect repair.

  * Whittle indices five ways (ClosedForm, Theorem, VI, PI, Package).  VI / PI are
    computed from the DEFINITION  W(s) = inf{W : passive optimal at s}  (grid scan of
    the exact single machine problem + bisection), so they stay correct OUTSIDE the
    proven domain c_switch <= c_bar.  ClosedForm / Theorem are only valid inside it.
  * Indexability diagnostics (per machine, aware and unaware):
      - numerical indexability (nested passive sets on a W grid, exact Q values)
      - closed-form monotonicity  W(0,0)<=W(0,1)<=W(1,0)<=...
      - verification of the closed-form CANDIDATE (Verification Lemma at breakpoints)
      - closed form vs numerical index
    plus indexability_study(): sweep c_switch/c_bar to see what happens beyond c_bar.
  * Benchmarks: uniform mu_L and reachable mu_R (L_R = at most R flagged machines),
    Delta_{mu_L}, Delta_{mu_R}, per-state eps pooled over L_R AND over all of L,
    benchmark-disagreement flags.
  * Lagrangian lower bound LB(s) = sup_{W>=0}[ sum_m V_m(s_m;W) - W R/(1-beta) ]
    (relaxation + decoupling + weak duality), gap of aware / unaware / greedy to LB,
    weak-duality check LB <= V*, duality gap.  The same bound is also recomputed using ONLY the
    breakpoints (indices) of PI / ClosedForm / Theorem, without the safety grid.
    Large-M route (no joint matrices): LB on sampled states of L_R vs Monte Carlo cost.
  * Aggregated validation: do the five methods give the same joint policy?
    (counts over ALL instances, split inside / outside c_bar)
  * Action mismatch printouts (OPT vs method, side by side) and aggregated mismatch rates.
    The per-instance aware-vs-unaware lists are available but OFF by default
    (SHOW_AWARE_VS_UNAWARE_LISTS) -- the aggregated aware!=unaware rates are always printed.

  The CLOSED-FORM index policies (ClosedForm_Aware / ClosedForm_Unaware) -- the novelty of the
  work -- appear in ALL aggregate outputs.

  NEW in this version
  -------------------
  * FIX: latex_perftable no longer crashes (r.iloc[0].median was the pandas METHOD, not the column).
  * Per-instance table now also reports, for every policy,
        DminV = 100 (min_s V^pi - min_s V*) / min_s V*
        DmaxV = 100 (max_s V^pi - max_s V*) / max_s V*
    over L (suffix _L) and over L_R (suffix _R), placed LEFT (min) and RIGHT (max) of Dmu_L / Dmu_R.
  * Per-instance printout of the joint state space size, joint action space size, shapes of the
    joint transition matrices and of the joint one-period cost vectors.
"""

import sys, platform, secrets
from datetime import datetime
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
    SCIPY = True
except ImportError:
    SCIPY = False
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False

# try:
#     import markovianbandit as bandit
#     PKG_AVAILABLE = True
#     _PKG_IMPORT_ERR = None
# except Exception as e:          # not only ImportError: catches errors inside the package too
#     PKG_AVAILABLE = False
#     _PKG_IMPORT_ERR = f"{type(e).__name__}: {e}"

# per-instance lists "aware action vs unaware action" (not part of the reference output)
SHOW_AWARE_VS_UNAWARE_LISTS = False
_PKG_WARNED = set()


# =====================================================================
# 1. Single machine MDP (costs >= 0).  Local order: [(0,1),(0,0),(1,0),...,(N,0)]
# =====================================================================
def build_extended_mdp(N, p, C, K, C_switch):
    """p has length N+1 with p[N]=0.  Active cost C from (0,1), C+C_switch elsewhere
    (perfect repair -> always to (0,1)).  Passive cost K(1-p_x)."""
    S = N + 2
    P0, P1 = np.zeros((S, S)), np.zeros((S, S))
    c0, c1 = np.zeros(S), np.full(S, C + C_switch)
    c1[0] = C
    P1[:, 0] = 1.0
    for s in range(S):
        x = 0 if s <= 1 else s - 1
        px = p[x]
        P0[s, 1] += 1 - px
        if px > 0 and x < N:
            P0[s, 2 if x == 0 else s + 1] = px
        c0[s] = K * (1 - px)
    labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N + 1)]
    return P0, P1, c0, c1, labels


# =====================================================================
# 2. Batched exact solvers (many charges W at once)
# =====================================================================
def batch_vi(P0, P1, c0, c1, Ws, beta, tol=1e-13, max_iter=20000):
    """Convergence tolerance 1e-13 so that VI indices agree with PI/closed form to ~1e-8."""
    Ws = np.atleast_1d(np.asarray(Ws, float))
    V = np.zeros((len(Ws), len(c0)))
    c1w = c1[None, :] + Ws[:, None]
    for _ in range(max_iter):
        Vn = np.minimum(c0 + beta * V @ P0.T, c1w + beta * V @ P1.T)
        d = np.abs(Vn - V).max()
        V = Vn
        if d < tol * (1 + np.abs(V).max()):
            break
    return V, c0 + beta * V @ P0.T, c1w + beta * V @ P1.T


def batch_pi(P0, P1, c0, c1, Ws, beta, max_iter=200):
    Ws = np.atleast_1d(np.asarray(Ws, float))
    nW, S = len(Ws), len(c0)
    c1w = c1[None, :] + Ws[:, None]
    pi = np.zeros((nW, S), dtype=int)
    I = np.eye(S)[None]
    for _ in range(max_iter):
        Ppi = np.where(pi[:, :, None] == 0, P0[None], P1[None])
        cpi = np.where(pi == 0, c0[None], c1w)
        V = np.linalg.solve(I - beta * Ppi, cpi[..., None])[..., 0]
        Q0, Q1 = c0 + beta * V @ P0.T, c1w + beta * V @ P1.T
        new = (Q1 < Q0 - 1e-12 * (1 + np.abs(Q0))).astype(int)
        if np.array_equal(new, pi):
            break
        pi = new
    return V, Q0, Q1


def solve_vi_discounted(P0, P1, c0, c1, w, beta):
    V, Q0, Q1 = batch_vi(P0, P1, c0, c1, [w], beta)
    return V[0], Q0[0], Q1[0]


def solve_pi_discounted(P0, P1, c0, c1, w, beta):
    V, Q0, Q1 = batch_pi(P0, P1, c0, c1, [w], beta)
    return V[0], Q0[0], Q1[0]


def policy_evaluation_discounted(pi, P0, P1, c0, c1, w, beta):
    S = len(pi)
    Ppi = np.where(np.asarray(pi)[:, None] == 0, P0, P1)
    cpi = np.where(np.asarray(pi) == 0, c0, c1 + w)
    return np.linalg.solve(np.eye(S) - beta * Ppi, cpi)


# =====================================================================
# 3. Index from the DEFINITION  W(s) = inf{W : Q0(s;W) <= Q1(s;W)}
# =====================================================================
def make_W_grid(C, K, cs, beta, n_grid):
    """Active is optimal everywhere for W <= -(C+cs); passive everywhere for W >= K/(1-beta)."""
    lo = -(C + cs) - 1.0
    mid = 1.5 * (K + cs) + 1.0
    hi = max(K / (1 - beta) + 1.0, mid + 1.0)
    return np.unique(np.concatenate([np.linspace(lo, mid, n_grid), np.linspace(mid, hi, 12)[1:]]))


def scan_tau(mach, beta, Ws, solver):
    fn = batch_pi if solver == "pi" else batch_vi
    _, Q0, Q1 = fn(mach["P0"], mach["P1"], mach["c0"], mach["c1"], Ws, beta)
    return Q0 - Q1                       # passive optimal  <=>  tau <= 0


def indices_from_scan(mach, beta, Ws, solver="pi", tau=None, n_bisect=45):
    tol = 1e-9 if solver == "pi" else 1e-8
    if tau is None:
        tau = scan_tau(mach, beta, Ws, solver)
    pas = tau <= tol
    S = tau.shape[1]
    W, lo, hi, todo = np.full(S, np.nan), np.zeros(S), np.zeros(S), np.zeros(S, bool)
    for s in range(S):
        if not pas[:, s].any():
            W[s] = np.inf
            continue
        g = int(np.argmax(pas[:, s]))
        if g == 0:
            W[s] = Ws[0]
            continue
        lo[s], hi[s], todo[s] = Ws[g - 1], Ws[g], True
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        passive = np.diag(scan_tau(mach, beta, mid, solver)) <= tol   # state s at its own W
        hi = np.where(todo & passive, mid, hi)
        lo = np.where(todo & ~passive, mid, lo)
    W[todo] = hi[todo]
    return W, tau


# ---- closed forms ----------------------------------------------------
def compute_G_H(p, beta):
    N = len(p)
    H, G = np.zeros(N + 1), np.zeros(N + 2)
    prod = 1.0
    for k in range(N + 1):
        H[k] = beta ** (k + 1) * prod
        if k < N:
            prod *= p[k]
    for k in range(N + 1):
        G[k + 1] = G[k] + (1 - (p[k] if k < N else 0.0)) * H[k]
    return G, H


def closed_form_indices(p, beta, K, C, C_switch):
    """[W(0,1), W(0,0), W(1,0..N,0)] -- derivation 1 (Lemma ordering form)."""
    N = len(p)
    G, H = compute_G_H(p, beta)
    w00 = (1 - p[0]) * K - C - C_switch * (1 - beta)
    den = beta * (1 - G[1]) - H[1]
    w01 = (1 - beta) * (K * G[1] + C_switch * H[1]) / den - C if abs(den) > 1e-15 else np.inf
    wp = np.zeros(N + 1)
    wp[0] = w00
    for k in range(1, N + 1):
        pk = p[k] if k < N else 0.0
        d = (1 - pk * beta) * (1 - G[k]) - (1 - pk) * H[k]
        wp[k] = K - C - C_switch - K * pk * (1 - beta) / d if abs(d) > 1e-15 else np.inf
    return np.concatenate(([w01], wp))


def theorem_indices(p, beta, K, C, C_switch):
    """Derivation 2: Theorem (thm:main) formulas."""
    N = len(p)
    G, _ = compute_G_H(p, beta)
    W = np.zeros(N + 2)
    W[0] = (1 - p[0]) * K - C + beta * p[0] * C_switch
    W[1] = (1 - p[0]) * K - C - (1 - beta) * C_switch
    for k in range(1, N + 1):
        pk = p[k] if k < N else 0.0
        num = 1 - G[k + 1] - pk * (1 - beta * G[k])
        den = 1 - G[k + 1] - beta * pk * (1 - G[k])
        W[1 + k] = K * num / den - C - C_switch
    return W


def c_bar(K, p0, p1, beta):
    return K * (p0 - p1) / (1 + beta * (p0 - p1))


def all_method_indices(mp, beta, n_grid=400, use_vi=True, use_pkg=True):
    N, p, C, K, cs = mp["N"], mp["p"], mp["C"], mp["K"], mp["C_switch"]
    P0, P1, c0, c1, labels = build_extended_mdp(N, np.append(p, 0.0), C, K, cs)
    mach = dict(P0=P0, P1=P1, c0=c0, c1=c1)
    S = N + 2
    Ws = make_W_grid(C, K, cs, beta, n_grid)
    W_pi, tau = indices_from_scan(mach, beta, Ws, "pi")
    W_vi = indices_from_scan(mach, beta, Ws, "vi")[0] if use_vi else np.full(S, np.nan)
    W_pkg = np.full(S, np.nan)
    if PKG_AVAILABLE and use_pkg:
        try:   # package is reward based: reward = -cost, same index
            W_pkg = np.array(bandit.restless_bandit_from_P0P1_R0R1(P0, P1, -c0, -c1)
                             .whittle_indices(discount=beta))
        except Exception as e:
            if "pkg" not in _PKG_WARNED:
                _PKG_WARNED.add("pkg")
                print(f"  [WARNING] markovianbandit call failed ({type(e).__name__}: {e}); "
                      f"Package indices will be reported as n/a.")
    W = {"ClosedForm": closed_form_indices(p, beta, K, C, cs), "VI": W_vi, "PI": W_pi,
         "Package": W_pkg, "Theorem": theorem_indices(p, beta, K, C, cs)}
    return dict(W=W, labels=labels, mach=mach, Ws=Ws, tau_pi=tau)


# =====================================================================
# 4. Indexability / monotonicity diagnostics
# =====================================================================
def chain_order(S):
    return [1, 0] + list(range(2, S))           # (0,0),(0,1),(1,0),(2,0),...


def check_monotone(W, labels, tol=1e-7):
    o = chain_order(len(W))
    for a, b in zip(o[:-1], o[1:]):
        if W[a] > W[b] + tol:
            return False, f"Monotonicity broken: {labels[a]} ({W[a]:.4f}) > {labels[b]} ({W[b]:.4f})"
    return True, ""


def indexability_from_tau(Ws, tau, labels, tol=1e-9):
    """Indexable <=> passive sets P(W) nested on the grid, using EXACT optimal Q's.
    This is a property of the machine, not of any candidate formula."""
    pas = tau <= tol
    errs = []
    for s in range(tau.shape[1]):
        col = pas[:, s]
        if not col.any():
            errs.append(f"{labels[s]}: never passive up to W={Ws[-1]:.1f}")
            continue
        f = int(np.argmax(col))
        bad = np.where(~col[f:])[0]
        if bad.size:
            i = f + bad[0]
            errs.append(f"{labels[s]}: passive from W={Ws[f]:.4f} but active again at W={Ws[i]:.4f} "
                        f"(Q0-Q1={tau[i, s]:.3e})")
    if pas[0].any():
        errs.append("passive set not empty at W_min")
    return len(errs) == 0, errs


def verify_candidate(mach, Wc, beta, labels, tol=1e-7, max_err=6):
    """Verification Lemma applied to a CANDIDATE index vector: for charges at / around every
    breakpoint, the policy 'passive iff Wc(s) <= W' must satisfy the Bellman optimality
    conditions.  (cost convention: passive optimal iff Q0 <= Q1)."""
    if not np.all(np.isfinite(Wc)):
        return False, ["non-finite candidate index"]
    pts = np.unique(Wc)
    d = 1e-4
    tests = [pts[0] - 1.0, pts[-1] + 1.0] + [w + e for w in pts for e in (-d, 0.0, d)]
    tests += [(a + b) / 2 for a, b in zip(pts[:-1], pts[1:])]
    errs = []
    for w in sorted(tests):
        passive = Wc <= w
        V = policy_evaluation_discounted((~passive).astype(int), mach["P0"], mach["P1"],
                                         mach["c0"], mach["c1"], w, beta)
        Q0 = mach["c0"] + beta * mach["P0"] @ V
        Q1 = mach["c1"] + w + beta * mach["P1"] @ V
        for s in range(len(Wc)):
            if passive[s] and Q1[s] < Q0[s] - tol * (1 + abs(Q0[s])):
                errs.append(f"W={w:.6f}, {labels[s]}: passive prescribed but active cheaper (Q1<Q0)")
            if (not passive[s]) and Q0[s] < Q1[s] - tol * (1 + abs(Q0[s])):
                errs.append(f"W={w:.6f}, {labels[s]}: active prescribed but passive cheaper (Q0<Q1)")
    return len(errs) == 0, errs[:max_err]


def machine_diagnostics(mp, res, beta):
    W, labels = res["W"], res["labels"]
    inside = mp["C_switch"] <= mp["c_bar"] * (1 + 1e-12)
    idx_ok, idx_errs = indexability_from_tau(res["Ws"], res["tau_pi"], labels)
    cf_ok, cf_msg = check_monotone(W["ClosedForm"], labels)
    nm_ok, nm_msg = check_monotone(W["PI"], labels)
    ver_ok, ver_errs = verify_candidate(res["mach"], W["ClosedForm"], beta, labels)
    fin = np.isfinite(W["ClosedForm"]) & np.isfinite(W["PI"])
    d_cf = float(np.max(np.abs(W["ClosedForm"] - W["PI"])[fin])) if fin.any() else np.nan
    fin2 = np.isfinite(W["ClosedForm"]) & np.isfinite(W["Theorem"])
    d_th = float(np.max(np.abs(W["ClosedForm"] - W["Theorem"])[fin2])) if fin2.any() else np.nan
    return dict(inside=inside, indexable=idx_ok, idx_errs=idx_errs, cf_mono=cf_ok, cf_msg=cf_msg,
                num_mono=nm_ok, num_msg=nm_msg, cf_verified=ver_ok, ver_errs=ver_errs,
                d_cf_num=d_cf, d_cf_thm=d_th, cf_eq_num=bool(d_cf < 1e-4), cf_eq_thm=bool(d_th < 1e-6))


def indexability_study(beta=0.95, N=7, n_machines=100, n_grid=300, seed=777,
                       fracs=(0, .25, .5, .75, .9, 1.0, 1.1, 1.25, 1.5, 2, 3, 5, 10)):
    """What happens as c_switch/c_bar grows?  Per machine, using exact Q values."""
    rng = np.random.default_rng(seed)
    rows, examples = [], {}
    for i in range(n_machines):
        D, a = rng.choice([0, 10, 20, 50]), rng.choice([1.5, 3.0, 4.0, 5.0])
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + 25)
        K = a * C
        cb = c_bar(K, p[0], p[1], beta)
        for f in fracs:
            mp = dict(N=N, p=p, C=C, K=K, C_switch=f * cb, c_bar=cb)
            res = all_method_indices(mp, beta, n_grid, use_vi=False, use_pkg=False)
            dg = machine_diagnostics(mp, res, beta)
            Wn = res["W"]["PI"]
            j = int(np.sum(Wn[2:] < Wn[0] - 1e-9))           # #(k,0) ranked below (0,1)
            rows.append(dict(frac=f, indexable=dg["indexable"], cf_mono=dg["cf_mono"],
                             cf_verified=dg["cf_verified"], cf_eq_num=dg["cf_eq_num"],
                             num_mono=dg["num_mono"], j=j,
                             W00_le_W10=bool(Wn[1] <= Wn[2] + 1e-9)))
            if not dg["indexable"] and f not in examples:
                examples[f] = dg["idx_errs"][:2]
    df = pd.DataFrame(rows).groupby("frac").agg(
        n=("indexable", "size"), indexable=("indexable", "mean"), cf_mono=("cf_mono", "mean"),
        cf_verified=("cf_verified", "mean"), cf_eq_num=("cf_eq_num", "mean"),
        num_chain_monotone=("num_mono", "mean"), W00_le_W10=("W00_le_W10", "mean"),
        mean_j=("j", "mean"), max_j=("j", "max")).reset_index()
    print("=" * 100)
    print(f"INDEXABILITY STUDY  (N={N}, beta={beta}, {n_machines} random machines per row; "
          "fractions of the numbers of machines)")
    print("  frac = c_switch / c_bar.   indexable: nested passive sets (exact Q).  cf_*: closed-form "
          "candidate.\n  num_chain_monotone: numerical W(0,0)<=W(0,1)<=W(1,0)<=...   "
          "j = #damaged states (k,0) whose index is below W(0,1) (j=0 is the draft's order)")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for f, e in examples.items():
        print(f"  first non-indexable example at frac={f}: {e}")
    print()
    return df


# =====================================================================
# 5. Joint MDP (sparse) + vectorised policies
# =====================================================================
def _kron(A, B):
    if SCIPY:
        A = A if sp.issparse(A) else sp.csr_matrix(A)
        return sp.kron(A, sp.csr_matrix(B), format="csr")
    return np.kron(A, B)


def _rowmask(P, mask):
    if SCIPY and sp.issparse(P):
        return sp.diags(mask.astype(float)) @ P
    return mask[:, None] * P


def enumerate_actions(M, R):
    """All subsets of <= R machines.  lut[bitmask] -> action index."""
    subsets = []
    for k in range(min(R, M) + 1):
        subsets += list(combinations(range(M), k))
    lut = np.full(1 << M, -1, dtype=int)
    for a, combo in enumerate(subsets):
        lut[sum(1 << m for m in combo)] = a
    return subsets, lut


def build_joint_mdp_hetero(machine_locals, R):
    M = len(machine_locals)
    L_list = [ml["P0"].shape[0] for ml in machine_locals]
    subsets, lut = enumerate_actions(M, R)
    P_joint, C_joint = [], []
    for combo in subsets:
        Pa, Cg = None, np.zeros(tuple(L_list))
        for m in range(M):
            k = "1" if m in combo else "0"
            mat = machine_locals[m]["P" + k]
            Pa = mat if Pa is None else _kron(Pa, mat)
            b = [1] * M
            b[m] = L_list[m]
            Cg = Cg + machine_locals[m]["c" + k].reshape(b)
        if SCIPY and not sp.issparse(Pa):
            Pa = sp.csr_matrix(Pa)
        P_joint.append(Pa)
        C_joint.append(Cg.reshape(-1))
    return P_joint, C_joint, L_list, subsets, lut


def evaluate_joint_policy(pi, P_joint, C_joint, beta, tol=1e-12, max_iter=100000):
    S = len(pi)
    P_pi, c_pi = None, np.zeros(S)
    for a in range(len(C_joint)):
        m = pi == a
        if m.any():
            part = _rowmask(P_joint[a], m)
            P_pi = part if P_pi is None else P_pi + part
            c_pi += np.where(m, C_joint[a], 0.0)
    V = np.zeros(S)
    for _ in range(max_iter):
        Vn = c_pi + beta * (P_pi @ V)
        d = np.abs(Vn - V).max()
        V = Vn
        if d <= tol * (1 + np.abs(V).max()):
            break
    return V


def solve_joint_vi(P_joint, C_joint, beta, tol=1e-10, max_iter=20000):
    """VI, then policy-evaluation polish + a few PI steps so V* is exact."""
    S, A = len(C_joint[0]), len(C_joint)
    V = np.zeros(S)
    for _ in range(max_iter):
        Vn = np.min([C_joint[a] + beta * (P_joint[a] @ V) for a in range(A)], axis=0)
        d = np.abs(Vn - V).max()
        V = Vn
        if d < tol * (1 + np.abs(V).max()):
            break
    Qs = np.stack([C_joint[a] + beta * (P_joint[a] @ V) for a in range(A)])
    pi = Qs.argmin(0)
    for _ in range(10):
        V = evaluate_joint_policy(pi, P_joint, C_joint, beta)
        Qs = np.stack([C_joint[a] + beta * (P_joint[a] @ V) for a in range(A)])
        cur = Qs[pi, np.arange(S)]
        new = np.where(cur <= Qs.min(0) + 1e-10 * (1 + np.abs(V)), pi, Qs.argmin(0))
        if np.array_equal(new, pi):
            break
        pi = new
    return V, pi


def joint_states(L_list):
    return np.stack(np.unravel_index(np.arange(int(np.prod(L_list))), tuple(L_list)), axis=1)


def damage_level(L, N):
    lv = np.zeros(L, dtype=int)
    lv[2:] = np.arange(1, L - 1)
    return lv


def index_actions(W_list, x, R):
    """Attend the (<=R) largest POSITIVE indices; ties -> lowest machine index."""
    P, M = x.shape
    w = np.stack([W_list[m][x[:, m]] for m in range(M)], axis=1)
    order = np.argsort(-w, axis=1, kind="stable")[:, :min(R, M)]
    rows = np.arange(P)[:, None]
    act = np.zeros((P, M), dtype=bool)
    act[rows, order] = w[rows, order] > 0
    return act


def greedy_actions(levels, x, R):
    """Top-R machines by current damage level; never idles; ties -> lowest index."""
    P, M = x.shape
    lv = np.stack([levels[m][x[:, m]] for m in range(M)], axis=1)
    order = np.argsort(-lv, axis=1, kind="stable")[:, :min(R, M)]
    act = np.zeros((P, M), dtype=bool)
    act[np.arange(P)[:, None], order] = True
    return act


def act_code(act):
    return (act.astype(np.int64) << np.arange(act.shape[1])).sum(1)


def sample_glazebrook_p(N, rng):
    return np.sort(rng.uniform(0.0, 1.0, N))[::-1]


# =====================================================================
# 6. Benchmarks  (mu_L uniform on L, mu_R uniform on L_R)
# =====================================================================
def reachable_mask(L_list, R):
    """L_R = {s : #machines flagged (x,1) <= R}.  Perfect repair: flag only in local index 0."""
    M = len(L_list)
    flags = np.zeros(tuple(L_list), dtype=int)
    for m in range(M):
        f = np.zeros(L_list[m], dtype=int)
        f[0] = 1
        b = [1] * M
        b[m] = L_list[m]
        flags = flags + f.reshape(b)
    return flags.reshape(-1) <= R


def make_result(pi, V, V_opt, pi_star, n_active, R, mask_R, LB=None):
    idle = R - n_active[pi]
    eps = 100.0 * (V - V_opt) / V_opt
    aL, aR, oL, oR = V.mean(), V[mask_R].mean(), V_opt.mean(), V_opt[mask_R].mean()
    out = dict(pi=pi, V=V, sum=V.sum(), subopt_LR=eps[mask_R], subopt_ALL=eps,
               idle_LR=idle[mask_R], idle_ALL=idle, avg_L=aL, avg_R=aR,
               delta_mu_L=100 * (aL - oL) / oL, delta_mu_R=100 * (aR - oR) / oR,
               match_LR=100 * np.mean((pi == pi_star)[mask_R]), match_ALL=100 * np.mean(pi == pi_star))
    # min / max over s of V^pi, relative to the same quantity for V*  (L = all states, R = L_R)
    rel = lambda a, b: 100.0 * (a - b) / b
    out.update(
        min_V_L=float(V.min()), max_V_L=float(V.max()),
        min_V_R=float(V[mask_R].min()), max_V_R=float(V[mask_R].max()),
        delta_min_L=rel(V.min(), V_opt.min()),
        delta_max_L=rel(V.max(), V_opt.max()),
        delta_min_R=rel(V[mask_R].min(), V_opt[mask_R].min()),
        delta_max_R=rel(V[mask_R].max(), V_opt[mask_R].max()))
    if LB is not None:
        g = 100.0 * (V - LB) / LB
        out.update(gapLB_LR=g[mask_R], gapLB_ALL=g,
                   gapLB_J_L=100 * (aL - LB.mean()) / LB.mean(),
                   gapLB_J_R=100 * (aR - LB[mask_R].mean()) / LB[mask_R].mean())
    return out


def benchmark_disagreements(results, tol=1e-9):
    """Pairs whose ORDER by J differs between mu_L and mu_R (Remark 'why mu must be stated')."""
    names = [m for m in UNIFORM_METHODS if m != "OPT" and results.get(m) is not None]
    fl = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]], results[names[j]]
            dL, dR = a["avg_L"] - b["avg_L"], a["avg_R"] - b["avg_R"]
            if abs(dL) > tol * abs(a["avg_L"]) and abs(dR) > tol * abs(a["avg_R"]) and dL * dR < 0:
                fl.append((names[i], names[j]))
    return fl


def action_gap(P_joint, C_joint, V_opt, beta, pi):
    """Extra cost of taking pi(s) once, then acting optimally.
    0 if pi(s) is optimal (ties included)."""
    Q = np.stack([C_joint[a] + beta * (P_joint[a] @ V_opt) for a in range(len(C_joint))])
    return Q[pi, np.arange(len(pi))] - V_opt


# =====================================================================
# 7. Lagrangian lower bound  (relaxation -> decoupling -> weak duality)
# =====================================================================
def lagrangian_candidates(W_aware_pi, K_list, beta, n_extra=40):
    """W>=0 at which D(s;W) is evaluated.  Under indexability the sup is at W=0 or at an index
    (Prop. dual-bound(c)); an extra grid is a safety net.  ANY W>=0 gives a VALID bound."""
    W_hi = max(K_list) / (1 - beta)
    c = [0.0]
    for Wm in W_aware_pi:
        c += [float(w) for w in Wm if np.isfinite(w) and 0 < w <= W_hi]
    top = max(c) * 1.05 if len(c) > 1 else 1.0
    c += list(np.linspace(0, top, n_extra))
    return np.unique(np.round(c, 10))


def breakpoint_candidates(W_list, K_list, beta):
    """W=0 plus the method's OWN breakpoints in (0, K_max/(1-beta)] -- NO safety grid."""
    W_hi = max(K_list) / (1 - beta)
    c = [0.0]
    for Wm in W_list:
        c += [float(w) for w in Wm if np.isfinite(w) and 0 < w <= W_hi]
    return np.unique(np.round(c, 10))


def lagrangian_lower_bound(machine_locals, cand_W, states_idx, beta, R):
    """LB(s) = max_W [ sum_m V_m(s_m;W) - W R/(1-beta) ]  using ONLY single-machine problems."""
    n, M = states_idx.shape
    Vtab = [batch_pi(ml["P0"], ml["P1"], ml["c0"], ml["c1"], cand_W, beta)[0] for ml in machine_locals]
    LB, argW = np.full(n, -np.inf), np.zeros(n)
    for ci, w in enumerate(cand_W):
        Dv = sum(Vtab[m][ci][states_idx[:, m]] for m in range(M)) - w * R / (1 - beta)
        upd = Dv > LB
        LB[upd], argW[upd] = Dv[upd], w
    return LB, argW


def lower_bound_from_breakpoints(machine_locals, W_list, K_list, states_idx, beta, R):
    cand = breakpoint_candidates(W_list, K_list, beta)
    return lagrangian_lower_bound(machine_locals, cand, states_idx, beta, R)[0]


# =====================================================================
# 8. Monte Carlo evaluation (no joint matrices)
# =====================================================================
def simulate_cost(machine_locals, action_fn, start_idx, beta, n_paths, seed, tol=1e-3):
    n, M = start_idx.shape
    x = np.repeat(start_idx, n_paths, axis=0)
    Pn = x.shape[0]
    cum0 = [np.cumsum(ml["P0"], 1) for ml in machine_locals]
    cum1 = [np.cumsum(ml["P1"], 1) for ml in machine_locals]
    Ls = [ml["P0"].shape[0] for ml in machine_locals]
    cmax = sum(max(ml["c0"].max(), ml["c1"].max()) for ml in machine_locals)
    T = int(np.ceil(np.log(tol * (1 - beta) / cmax) / np.log(beta)))
    rng = np.random.default_rng(seed)               # same seed => common random numbers
    total, disc = np.zeros(Pn), 1.0
    for _ in range(T):
        act = action_fn(x)
        u = rng.random((Pn, M))
        nx, cost = np.empty_like(x), np.zeros(Pn)
        for m in range(M):
            ml, xm, a = machine_locals[m], x[:, m], act[:, m]
            cost += np.where(a, ml["c1"][xm], ml["c0"][xm])
            n0 = np.minimum((cum0[m][xm] < u[:, [m]]).sum(1), Ls[m] - 1)
            n1 = np.minimum((cum1[m][xm] < u[:, [m]]).sum(1), Ls[m] - 1)
            nx[:, m] = np.where(a, n1, n0)
        total += disc * cost
        disc *= beta
        x = nx
    est = total.reshape(n, n_paths)
    return est.mean(1), est.std(1, ddof=1) / np.sqrt(n_paths), est


def sample_reachable_states(L_list, R, n, rng):
    out = []
    while len(out) < n:
        s = np.stack([rng.integers(0, L, size=4 * n) for L in L_list], axis=1)
        out.extend(s[(s == 0).sum(1) <= R].tolist())
    return np.array(out[:n])


# =====================================================================
# 9. Config
# =====================================================================
VARIOUS_ALPHA_MULTIPLIERS = [1.5, 3.0, 4.5]
AlphaSpec = Union[float, str, List[float], Tuple[float, ...]]
SwitchFracSpec = Union[float, List[float], Tuple[float, ...]]


def resolve_alpha_per_machine(alpha, M, various=None):
    if isinstance(alpha, str):
        if alpha != "Various":
            raise ValueError(f"Unknown alpha spec: {alpha!r}")
        if various is None or len(various) != M:
            raise ValueError(f'"Various" needs {M} multipliers (cfg.various_alpha_multipliers)')
        return [float(a) for a in various]
    if isinstance(alpha, (list, tuple)):
        if len(alpha) != M:
            raise ValueError(f"alpha list length {len(alpha)} != M={M}")
        return [float(a) for a in alpha]
    return [float(alpha)] * M


def resolve_switch_frac_per_machine(sf, M):
    if isinstance(sf, (list, tuple)):
        if len(sf) != M:
            raise ValueError(f"switch_frac list length {len(sf)} != M={M}")
        return [float(s) for s in sf]
    return [float(sf)] * M


def switch_frac_cell_label(sf):
    """Always a STRING ('0.5', '1.01', 'mixed_0.3-0.6-0.8-0.2') so DataFrame columns / groupby
    keys never mix floats and strings."""
    if isinstance(sf, (list, tuple)):
        return "mixed_" + "-".join(f"{float(s):g}" for s in sf)
    return f"{float(sf):g}"


INDEX_METHODS = ["ClosedForm", "VI", "PI", "Package", "Theorem"]
UNIFORM_METHODS = (["OPT"] + [f"{m}_Aware" for m in INDEX_METHODS]
                   + [f"{m}_Unaware" for m in INDEX_METHODS] + ["Greedy"])
_SHORT = {"ClosedForm": "CF", "VI": "VI", "PI": "PI", "Package": "Pkg", "Theorem": "Thm"}
SHORT_NAME = {"OPT": "OPT", "Greedy": "Greedy"}
for _m in INDEX_METHODS:
    SHORT_NAME[f"{_m}_Aware"] = f"{_SHORT[_m]}_A"
    SHORT_NAME[f"{_m}_Unaware"] = f"{_SHORT[_m]}_U"

# Methods shown in the aggregate tables by default.  ClosedForm is the novelty of the work, so it
# is shown next to the definition-based PI index (VI / Package / Theorem coincide with CF or PI
# and are covered by the aggregated validation table).
DEFAULT_TABLE_METHODS = ["ClosedForm_Aware", "ClosedForm_Unaware",
                         "PI_Aware", "PI_Unaware", "Greedy"]

# pairs of policies whose action mismatch rate (% of states) is aggregated
MISMATCH_PAIRS = {
    "aware!=unaware":       ("PI_Aware", "PI_Unaware"),
    "aware!=OPT":           ("PI_Aware", "OPT"),
    "unaware!=OPT":         ("PI_Unaware", "OPT"),
    "greedy!=OPT":          ("Greedy", "OPT"),
    "CFaware!=CFunaware":   ("ClosedForm_Aware", "ClosedForm_Unaware"),
    "CFaware!=OPT":         ("ClosedForm_Aware", "OPT"),
    "CFunaware!=OPT":       ("ClosedForm_Unaware", "OPT"),
    "CFaware!=PIaware":     ("ClosedForm_Aware", "PI_Aware"),
    "CFunaware!=PIunaware": ("ClosedForm_Unaware", "PI_Unaware"),
}


@dataclass
class ExperimentConfig:
    N_list: Optional[List[int]] = field(default_factory=lambda: [6, 6, 6])
    M: int = 3
    N_range: Tuple[int, int] = (3, 6)
    M_range: Tuple[int, int] = (2, 5)
    R: int = 1
    beta: float = 0.95
    n_reps: int = 10
    D_grid: List[float] = field(default_factory=lambda: [0, 10, 20, 25, 30, 40, 50])
    alpha_grid: List[AlphaSpec] = field(default_factory=lambda: [1.5, 2.0, 3.0, 4.0, 5.0, "Various"])
    C_width: float = 25.0
    various_alpha_multipliers: Optional[Union[List[float], Callable]] = None
    switch_fracs: List[SwitchFracSpec] = field(default_factory=lambda: [0.0])
    seed0: int = 1000
    # index computation
    n_grid: int = 400            # W-grid size for the definition-based (scan) indices
    use_vi: bool = True
    # lower bound
    compute_LB: bool = True
    lb_extra_grid: int = 40
    # which methods appear in the aggregate tables: list, or "all".
    table_methods: Union[str, List[str]] = field(default_factory=lambda: list(DEFAULT_TABLE_METHODS))
    # logging
    verbose_indices: bool = True
    verbose_policies: bool = True
    verbose_joint_shapes: bool = True
    verbose_machine_matrices: bool = True
    policy_dump_max_states: int = 1000
    print_state_legend: bool = False
    print_state_action_tables: bool = True
    state_print_chunk: int = 100   # # first / middle / last this many states
    print_action_diffs: bool = True
    diff_scope: str = "LR"       # which states the mismatch LISTS use: "LR" or "ALL"
    output_txt: Optional[str] = None

    def __post_init__(self):
        if self.N_list is not None:
            self.M = len(self.N_list)
        if "Various" in self.alpha_grid:
            v = self.various_alpha_multipliers
            if callable(v):
                v = list(v(self.M))
            if v is None and self.M == 3:
                v = list(VARIOUS_ALPHA_MULTIPLIERS)
            if v is None or len(v) != self.M:
                raise ValueError('"Various" needs cfg.various_alpha_multipliers of length M')
            self.various_alpha_multipliers = list(v)
        if self.table_methods != "all":
            bad = [m for m in self.table_methods if m not in UNIFORM_METHODS]
            if bad:
                raise ValueError(f"unknown table_methods {bad}; valid: {UNIFORM_METHODS}")
        if self.output_txt is None:
            self.output_txt = f"Discounted_cost_repairman_results_{datetime.now():%Y-%m-%d_%H%M%S}.txt"

    def methods_for_tables(self):
        return UNIFORM_METHODS if self.table_methods == "all" else list(self.table_methods)


def resolve_N_list(cfg, rng):
    if cfg.N_list is not None:
        return list(cfg.N_list)
    return [int(rng.integers(cfg.N_range[0], cfg.N_range[1] + 1)) for _ in range(cfg.M)]


def make_random_config(*, M_range=(2, 5), N_range=(3, 8), R_max=None, seed0=None, meta_seed=None,
                       various_alpha_multipliers=None, **overrides):
    if meta_seed is None:
        meta_seed = secrets.randbits(31)
    r = np.random.default_rng(meta_seed)
    M = int(r.integers(M_range[0], M_range[1] + 1))
    N_list = [int(r.integers(N_range[0], N_range[1] + 1)) for _ in range(M)]
    R = int(r.integers(1, (R_max or M) + 1))
    s0 = seed0 if seed0 is not None else int(r.integers(0, 2 ** 31 - 1))
    cfg = ExperimentConfig(N_list=N_list, R=R, seed0=s0,
                           various_alpha_multipliers=various_alpha_multipliers, **overrides)
    print(f"[random experiment] meta_seed={meta_seed} -> M={M}, N_list={N_list}, R={R}, seed0={s0}")
    return cfg


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, d):
        for s in self.streams:
            s.write(d)

    def flush(self):
        for s in self.streams:
            s.flush()


# =====================================================================
# 10. Instance builder (shared by exact and large-M routes)
# =====================================================================
def build_instance(cfg, D, alpha, switch_frac, instance_seed):
    rng = np.random.default_rng(instance_seed)
    N_list = resolve_N_list(cfg, rng)
    M, beta = len(N_list), cfg.beta
    alpha_pm = resolve_alpha_per_machine(alpha, M, cfg.various_alpha_multipliers)
    sf_pm = resolve_switch_frac_per_machine(switch_frac, M)
    inst = dict(N_list=N_list, M=M, machine_params=[], machine_locals=[], labels=[],
                W={"Aware": {m: [] for m in INDEX_METHODS}, "Unaware": {m: [] for m in INDEX_METHODS}},
                diag={"Aware": [], "Unaware": []}, warnings=[], instance_seed=instance_seed)
    for i, (N, a_m, sf_m) in enumerate(zip(N_list, alpha_pm, sf_pm)):
        p = sample_glazebrook_p(N, rng)
        C = rng.uniform(D, D + cfg.C_width)
        K = a_m * C
        pp = np.append(p, 0.0)
        cb = c_bar(K, pp[0], pp[1], beta)
        mp = dict(N=N, p=p, C=C, K=K, C_switch=sf_m * cb, c_bar=cb, alpha_m=a_m, switch_frac_m=sf_m)
        mp_un = dict(mp, C_switch=0.0)
        res = {"Aware": all_method_indices(mp, beta, cfg.n_grid, cfg.use_vi),
               "Unaware": all_method_indices(mp_un, beta, cfg.n_grid, cfg.use_vi)}
        inst["machine_params"].append(mp)
        inst["machine_locals"].append(res["Aware"]["mach"])      # TRUE system: real C_switch
        inst["labels"].append(res["Aware"]["labels"])
        for tag, mpx in (("Aware", mp), ("Unaware", mp_un)):
            dg = machine_diagnostics(mpx, res[tag], beta)
            inst["diag"][tag].append(dg)
            for method in INDEX_METHODS:
                inst["W"][tag][method].append(res[tag]["W"][method])
        if mp["C_switch"] > mp["c_bar"] * (1 + 1e-12):
            inst["warnings"].append(f"machine {i+1}: C_switch={mp['C_switch']:.3f} > c_bar={mp['c_bar']:.3f} "
                                    f"-- OUTSIDE proven domain; ClosedForm/Theorem not guaranteed.")
    return inst


# =====================================================================
# 11. Printing helpers
# =====================================================================
def _fmt_combo(c):
    if not c:
        return "(idle)"
    return ("machine " if len(c) == 1 else "machines ") + str(tuple(m + 1 for m in c))


def _act_str(a, subsets):
    return f"a={int(a):>3d}  service {_fmt_combo(subsets[int(a)])}"


def sampled_state_blocks(S, chunk):
    if chunk <= 0 or S <= 3 * chunk:
        return [list(range(S))]
    mid = S // 2 - chunk // 2
    return [list(range(chunk)), list(range(mid, mid + chunk)), list(range(S - chunk, S))]


def print_sampled_rows(S, chunk, row_fn, indent="    "):
    prev = -1
    for b in sampled_state_blocks(S, chunk):
        if prev >= 0 and b[0] > prev + 1:
            print(f"{indent}   ...   (states {prev + 1}..{b[0] - 1} not shown)")
        for s in b:
            print(indent + row_fn(s))
        prev = b[-1]


def print_sampled_list(idx, chunk, row_fn, indent="    "):
    n = len(idx)
    if n == 0:
        print(f"{indent}(none)")
        return
    blocks = [(0, n)] if (chunk <= 0 or n <= 3 * chunk) else \
        [(0, chunk), (n // 2 - chunk // 2, n // 2 - chunk // 2 + chunk), (n - chunk, n)]
    prev = None
    for lo, hi in blocks:
        if prev is not None and lo > prev:
            print(f"{indent}   ...   ({lo - prev} differing states not shown)")
        for s in idx[lo:hi]:
            print(indent + row_fn(int(s)))
        prev = hi


def make_state_str_fn(L_list, labels):
    shape = tuple(L_list)

    def f(s):
        i = np.unravel_index(s, shape)
        return "[" + " | ".join(labels[m][i[m]] for m in range(len(L_list))) + "]"
    return f


def idle_pct(idle, R):
    return 100.0 * np.bincount(idle, minlength=R + 1)[:R + 1] / len(idle)


def _pf(x):
    return "PASS" if x else "FAIL"


def print_machine_matrices(m, mp, mach, labels):
    """Single-machine transition matrices P0 (passive), P1 (active) and one-period
    cost vectors c0, c1, labelled by local state.  mach = TRUE system (real C_switch)."""
    P0, P1, c0, c1 = mach["P0"], mach["P1"], mach["c0"], mach["c1"]
    print(f"\n  SINGLE-MACHINE MDP of machine {m + 1}  "
          f"(rows = current local state, columns = next local state)")

    print(f"  P0  passive (a=0): row s -> damage x->x+1 w.p. p_x, else breakdown -> (0,0)")
    print(pd.DataFrame(P0, index=labels, columns=labels)
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n  P1  active (a=1): every state -> (0,1) w.p. 1  (perfect repair)")
    print(pd.DataFrame(P1, index=labels, columns=labels)
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n  ONE-PERIOD COSTS  c0 = passive K(1-p_x),  c1 = active C (from (0,1)) or C+C_switch (elsewhere)")
    c1_un = c1.copy()
    c1_un[1:] = mp["C"]                       # unaware model: C_switch = 0
    cost_df = pd.DataFrame({"c0 (passive)": c0,
                            "c1 (active, real)": c1,
                            "c1 (active, unaware)": c1_un}, index=labels)
    print(cost_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"  check: P0 rows sum to 1: {np.allclose(P0.sum(1), 1)}   "
          f"P1 rows sum to 1: {np.allclose(P1.sum(1), 1)}   "
          f"all costs >= 0: {bool((c0 >= 0).all() and (c1 >= 0).all())}")


def print_machine_diagnostics(m, mp, dga, dgu):
    print(f"\nMachine {m + 1}: N={mp['N']}  C={mp['C']:.3f}  K={mp['K']:.3f}  (alpha_m={mp['alpha_m']})  "
          f"switch_frac_m={mp['switch_frac_m']:.3f}  C_switch={mp['C_switch']:.3f}  c_bar={mp['c_bar']:.3f}  "
          f"[{'INSIDE' if dga['inside'] else 'OUTSIDE'} domain]")
    for tag, dg in (("AWARE", dga), ("UNAWARE", dgu)):
        print(f"  [{tag}] numerical indexability (exact Q, nested passive sets): {_pf(dg['indexable'])}")
        for e in dg["idx_errs"][:3]:
            print(f"        {e}")
        print(f"  [{tag}] discounted monotonicity (closed form): {_pf(dg['cf_mono'])}"
              + (f"   {dg['cf_msg']}" if not dg['cf_mono'] else ""))
        print(f"  [{tag}] discounted monotonicity (numerical):   {_pf(dg['num_mono'])}"
              + (f"   {dg['num_msg']}" if not dg['num_mono'] else ""))
        print(f"  [{tag}] verification of closed-form candidate:  {_pf(dg['cf_verified'])}")
        for e in dg["ver_errs"][:3]:
            print(f"        {e}")
        print(f"  [{tag}] max|ClosedForm-PI|={dg['d_cf_num']:.3g}   max|ClosedForm-Theorem|={dg['d_cf_thm']:.3g}")


# ---- action mismatch printouts --------------------------------------
def print_aware_vs_unaware_diffs(results, log, cfg, state_str, scope_idx):
    S, subsets, chunk = log["S"], log["action_subsets"], cfg.state_print_chunk
    print("\n" + "-" * 92)
    print(f"STATES WHERE AWARE and UNAWARE ACTIONS DIFFER (lists over scope {cfg.diff_scope})")
    print("-" * 92)
    for method in INDEX_METHODS:
        A_, U_ = results.get(f"{method}_Aware"), results.get(f"{method}_Unaware")
        if A_ is None or U_ is None:
            print(f"\n[{method}] n/a")
            continue
        d_all = A_["pi"] != U_["pi"]
        d = scope_idx[d_all[scope_idx]]
        print(f"\n[{method}] differ on {d_all.sum()}/{S} states ({100 * d_all.mean():.2f}% of L); "
              f"{int(d_all[log['mask_R']].sum())}/{int(log['mask_R'].sum())} in L_R")
        print_sampled_list(d, chunk, lambda s: f"s={s:>7d} : {state_str(s)}\n{'':13s}aware   -> "
                           f"{_act_str(A_['pi'][s], subsets)}\n{'':13s}unaware -> {_act_str(U_['pi'][s], subsets)}")


def print_cross_method_diffs(results, log, cfg, state_str, scope_idx):
    S, subsets, chunk = log["S"], log["action_subsets"], cfg.state_print_chunk
    pi_opt = results["OPT"]["pi"]
    print("\n" + "-" * 92)
    print("CROSS COMPARISON (i): OPT action vs each method, only states where they DIFFER")
    print("-" * 92)
    for method in UNIFORM_METHODS:
        v = results.get(method)
        if method == "OPT" or v is None:
            continue
        d_all = v["pi"] != pi_opt
        d = scope_idx[d_all[scope_idx]]
        print(f"\n[{method}] differs from OPT on {d_all.sum()}/{S} states ({100 * d_all.mean():.2f}%); "
              f"in L_R: {100 * d_all[log['mask_R']].mean():.2f}%")
        print_sampled_list(d, chunk, lambda s: f"s={s:>7d} : {state_str(s)}\n{'':13s}OPT      -> "
                           f"{_act_str(pi_opt[s], subsets)}\n{'':13s}{method:<9s}-> {_act_str(v['pi'][s], subsets)}")


# ---- Whittle indices of every method, legends, policy/value tables ------
def print_index_table(m, labels, inst):
    """Whittle index of machine m from EACH method, aware and unaware, one table."""
    rows = {}
    for tag in ("Aware", "Unaware"):
        for method in INDEX_METHODS:
            rows[f"{tag[0]}:{method}"] = inst["W"][tag][method][m]
    df = pd.DataFrame(rows, index=labels).T
    print(f"\n  WHITTLE INDICES of machine {m + 1}, every method (columns = local states; "
          f"A = aware: real C_switch, U = unaware: C_switch=0; nan = method unavailable)")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    for tag in ("Aware", "Unaware"):
        w = inst["W"][tag]["PI"][m]
        order = np.argsort(w, kind="stable")
        print(f"  index order ({tag}, PI, lowest -> highest): " + " < ".join(labels[i] for i in order))


def print_action_legend(log, cfg):
    R = cfg.R
    print("\n" + "-" * 92)
    print("ACTION LEGEND (action index -> machines serviced, 1-based)")
    print("-" * 92)
    for a, c in enumerate(log["action_subsets"]):
        print(f"    action {a}: service {_fmt_combo(c)}   [{len(c)} repairmen working, {R - len(c)} idle]")


def print_state_legend(log, cfg, state_str):
    L_list, M, S, mask_R = log["L_list"], log["M"], log["S"], log["mask_R"]
    labels = log["inst"]["labels"]
    print("\n" + "-" * 92)
    print("STATE LEGEND (joint state index s -> local state of machine 1 | ... | machine M)")
    print("-" * 92)
    print("  local state (x,a_(t-1)): x = damage level, a_(t-1) = 1 iff the machine was attended in the previous period")
    print("  (under perfect repair only (0,1) carries a_(t-1)=1), else a_(t-1)=0.  Machine 1 is the slowest varying coordinate.")
    for m in range(M):
        print(f"  machine {m + 1} ({L_list[m]} states), local index -> label: "
              + ", ".join(f"{i}:{lab}" for i, lab in enumerate(labels[m])))
    print(f"  'Y' marks states of L_R (at most R={cfg.R} machines flagged); '.' marks states outside L_R.")
    if S > 3 * cfg.state_print_chunk:
        print(f"  (showing first / middle / last {cfg.state_print_chunk} of {S} states)")
    print_sampled_rows(S, cfg.state_print_chunk,
                       lambda s: f"s={s:>7d} {'Y' if mask_R[s] else '.'} : {state_str(s)}")


def print_policy_value_table(results, log, cfg, state_str, idx=None, title=""):
    """ONE table: state | for every policy  'action index | value function V^pi(s)'.
    idx=None -> first/middle/last states of L; otherwise only the listed states."""
    S, mask_R, chunk = log["S"], log["mask_R"], cfg.state_print_chunk
    avail = [m for m in UNIFORM_METHODS if results.get(m) is not None]
    w = max(10, 13 * log["M"])
    print("\n" + "-" * 92)
    print(title)
    print("-" * 92)
    print("  cell = action index | value function V^pi(s) (expected discounted cost, smaller is better);")
    print("  'Y' = state in L_R.  Abbreviations: OPT = true optimum, CF = ClosedForm, Thm = Theorem, "
          "Pkg = Package, _A = switching-aware index, _U = unaware index.")
    print("    " + f"{'s':>9s} R : {'state':<{w}s} | " + " ".join(f"{SHORT_NAME[m]:>12s}" for m in avail))

    def row(s):
        cells = " ".join(f"{int(results[m]['pi'][s]):>2d}|{results[m]['V'][s]:>9.3f}" for m in avail)
        return f"s={s:>7d} {'Y' if mask_R[s] else '.'} : {state_str(s):<{w}s} | {cells}"
    if idx is None:
        print_sampled_rows(S, chunk, row)
    else:
        print_sampled_list(idx, chunk, row)


# =====================================================================
# 12. Exact route (small M): OPT, LB, all policies
# =====================================================================
def validation_record(results, inst):
    rec = {}
    for tag in ("Aware", "Unaware"):
        ref = results.get(f"PI_{tag}")
        agree, ndiff, idxdiff = {}, {}, {}
        for m in INDEX_METHODS:
            if m == "PI":
                continue
            v = results.get(f"{m}_{tag}")
            if v is None or ref is None:
                agree[m], ndiff[m], idxdiff[m] = None, None, np.nan
                continue
            agree[m] = bool(np.array_equal(v["pi"], ref["pi"]))
            ndiff[m] = int(np.sum(v["pi"] != ref["pi"]))
            dd = [np.max(np.abs(a[np.isfinite(a) & np.isfinite(b)] - b[np.isfinite(a) & np.isfinite(b)]), initial=0.0)
                  for a, b in zip(inst["W"][tag][m], inst["W"][tag]["PI"])]
            idxdiff[m] = float(max(dd))
        rec[tag] = dict(agree=agree, ndiff=ndiff, idxdiff=idxdiff,
                        all5=all(v for v in agree.values() if v is not None),
                        pkg=agree.get("Package") is not None)
    rec["inside"] = all(d["inside"] for d in inst["diag"]["Aware"])
    return rec


def run_cell_exact(cfg, D, alpha, switch_frac, instance_seed, verbose=True):
    inst = build_instance(cfg, D, alpha, switch_frac, instance_seed)
    beta, R, M = cfg.beta, cfg.R, inst["M"]
    locs = inst["machine_locals"]
    P_joint, C_joint, L_list, subsets, lut = build_joint_mdp_hetero(locs, R)
    n_active = np.array([len(c) for c in subsets])
    S = int(np.prod(L_list))
    V_opt, pi_star = solve_joint_vi(P_joint, C_joint, beta)
    states, mask_R = joint_states(L_list), reachable_mask(L_list, R)

    LB = argW = None
    lb_bp = None
    if cfg.compute_LB:
        K_list = [mp["K"] for mp in inst["machine_params"]]
        cand = lagrangian_candidates(inst["W"]["Aware"]["PI"], K_list, beta, cfg.lb_extra_grid)
        LB, argW = lagrangian_lower_bound(locs, cand, states, beta, R)
        # bound from each method's OWN breakpoints only (no safety grid), aware indices
        raw = {name: lower_bound_from_breakpoints(locs, inst["W"]["Aware"][name], K_list, states, beta, R)
               for name in ("PI", "ClosedForm", "Theorem")}
        ref = raw["PI"]
        lb_bp = {name: dict(J=float(lbm.mean()), maxdiff=float(np.max(np.abs(lbm - ref))),
                            valid=bool(np.all(lbm <= V_opt + 1e-7 * (1 + V_opt))))
                 for name, lbm in raw.items()}

    results = {"OPT": make_result(pi_star, V_opt, V_opt, pi_star, n_active, R, mask_R, LB)}
    cache = {}

    def evaluate(pi):
        k = pi.tobytes()
        if k not in cache:
            cache[k] = evaluate_joint_policy(pi, P_joint, C_joint, beta)
        return cache[k]

    for tag in ("Aware", "Unaware"):
        for method in INDEX_METHODS:
            Wl = inst["W"][tag][method]
            if any(np.any(np.isnan(w)) for w in Wl):
                results[f"{method}_{tag}"] = None
                continue
            pi = lut[act_code(index_actions(Wl, states, R))]
            results[f"{method}_{tag}"] = make_result(pi, evaluate(pi), V_opt, pi_star, n_active, R, mask_R, LB)
    levels = [damage_level(L_list[m], inst["N_list"][m]) for m in range(M)]
    pg = lut[act_code(greedy_actions(levels, states, R))]
    results["Greedy"] = make_result(pg, evaluate(pg), V_opt, pi_star, n_active, R, mask_R, LB)

    # ---- action gap: is each action optimal UP TO TIES? --------------------
    for name, v in results.items():
        if v is None:
            continue
        g = action_gap(P_joint, C_joint, V_opt, beta, v["pi"])
        ok = g <= 1e-9 * (1 + V_opt)              # optimal up to ties
        v["action_gap"] = g
        v["match_tol_ALL"] = 100 * ok.mean()
        v["match_tol_LR"] = 100 * ok[mask_R].mean()
        bad = g[mask_R & ~ok]                      # real mismatches inside L_R
        v["gap_bad_median"] = float(np.median(bad)) if bad.size else 0.0
        v["gap_bad_max"] = float(bad.max()) if bad.size else 0.0

    log = dict(inst=inst, M=M, S=S, L_list=L_list, action_subsets=subsets, mask_R=mask_R, pi_star=pi_star,
               V_opt=V_opt, LB=LB, argW=argW, lb_bp=lb_bp, instance_seed=instance_seed,
               P_joint=P_joint, C_joint=C_joint, validation=validation_record(results, inst))
    if LB is not None:
        log["lb_valid"] = bool(np.all(LB <= V_opt + 1e-7 * (1 + V_opt)))
        log["dual_gap"] = 100.0 * (V_opt - LB) / LB
    if verbose:
        log_instance(D, alpha, switch_frac, cfg, log, results)
    return results, log


def replay_instance(cfg, D, alpha, switch_frac, instance_seed, verbose=True):
    return run_cell_exact(cfg, D, alpha, switch_frac, instance_seed, verbose)


def log_instance(D, alpha, switch_frac, cfg, log, results):
    inst, M, R, S, mask_R = log["inst"], log["M"], cfg.R, log["S"], log["mask_R"]
    L_list, subsets = log["L_list"], log["action_subsets"]
    print("=" * 92)
    print(f"D={D}  alpha={alpha}  switch_frac={switch_frac_cell_label(switch_frac)}  R={R}  M={M}  "
          f"beta={cfg.beta}  N_list={inst['N_list']}  instance_seed={log['instance_seed']}")
    print("=" * 92)
    for w in inst["warnings"]:
        print(f"  [WARNING] {w}")
    for m, mp in enumerate(inst["machine_params"]):
        print_machine_diagnostics(m, mp, inst["diag"]["Aware"][m], inst["diag"]["Unaware"][m])
        if cfg.verbose_machine_matrices:
            print_machine_matrices(m, mp, inst["machine_locals"][m], inst["labels"][m])
        if cfg.verbose_indices:
            print_index_table(m, inst["labels"][m], inst)
    print(f"\nJoint state space size |L| = {S} = " + "*".join(map(str, L_list))
          + f"   |L_R| = {int(mask_R.sum())}  ({100 * mask_R.mean():.1f}% of L; at most R={R} machines flagged)")
    if cfg.verbose_joint_shapes:
        #print(f"Joint state space size: {S} = " + "*".join(map(str, L_list)))
        print(f"Joint action space size: {len(subsets)}  (R={R} repairmen, M={M} machines)")
        print(f"Joint transition matrices: {[tuple(P.shape) for P in log['P_joint']]}")
        print(f"Joint one-period cost vectors: {[tuple(c.shape) for c in log['C_joint']]}")

    # ---------------- ONE table: eps quartiles, Delta_mu (+min/max of V), match-with-OPT, matchTol, badGap, idle %, gap to LB
    rows = []
    for method in UNIFORM_METHODS:
        v = results.get(method)
        if v is None:
            rows.append(dict(Method=method))
            continue
        r = dict(Method=method)
        for sc in ("LR", "ALL"):
            e = v[f"subopt_{sc}"]
            r.update({f"{sc}_q1": np.percentile(e, 25), f"{sc}_med": np.percentile(e, 50),
                      f"{sc}_q3": np.percentile(e, 75), f"{sc}_max": e.max()})
        # min-V ratio LEFT of Dmu, max-V ratio RIGHT of Dmu (for L and for L_R)
        r.update(DminV_L=v["delta_min_L"], Dmu_L=v["delta_mu_L"], DmaxV_L=v["delta_max_L"],
                 DminV_R=v["delta_min_R"], Dmu_R=v["delta_mu_R"], DmaxV_R=v["delta_max_R"],
                 matchLR=v["match_LR"], matchALL=v["match_ALL"],
                 matchTolLR=v.get("match_tol_LR", np.nan), matchTolALL=v.get("match_tol_ALL", np.nan),
                 badGapMedian=v.get("gap_bad_median", np.nan), badGapMax=v.get("gap_bad_max", np.nan))
        for sc in ("LR", "ALL"):                      # % of states with exactly k idle repairmen
            ip = idle_pct(v[f"idle_{sc}"], R)
            r.update({f"{sc}_{k}idle": ip[k] for k in range(R + 1)})
        if log["LB"] is not None:
            r.update(gapLB_JL=v["gapLB_J_L"], gapLB_JR=v["gapLB_J_R"])
        rows.append(r)
    print("\n" + "-" * 92)
    print("THIS INSTANCE: per-state eps(s)=100(V_pi-V*)/V*  pooled over L_R  vs  over ALL of L;")
    print("               Delta_muL / Delta_muR = ratio of means under mu_L / mu_R, Delta_mu(pi) = 100 (J(pi)-J(pi*))/J(pi*);  gapLB_J* = 100(J(pi)-J(LB))/J(LB)")
    print("               DminV_* / DmaxV_* = 100(min_s V^pi - min_s V*)/min_s V*  and the same with max_s;"
          " _L over all of L, _R over L_R")
    print("               match*    = % of states where the action equals the OPT action (exact index match)")
    print("               matchTol* = % of states where the action is optimal UP TO TIES (Q-gap <= 1e-9 relative)")
    print("               badGap*   = one-step extra cost (absolute, cost units) on the REAL mismatches in L_R")
    print(f"               *_kidle   = % of states with exactly k of the R={R} repairmen idle "
          f"(k={R}: nobody works), over L_R / ALL")
    print("-" * 92)
    with pd.option_context("display.width", 400, "display.max_columns", None):
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    fl = benchmark_disagreements(results)
    print(f"benchmark disagreement (order of J differs between mu_L and mu_R): {'NONE' if not fl else fl}")

    if log["LB"] is not None:
        LB, V_opt = log["LB"], log["V_opt"]
        dg = log["dual_gap"]
        print("\n" + "-" * 92)
        print("LAGRANGIAN LOWER BOUND  LB(s) = max_{W>=0}[ sum_m V_m(s_m;W) - W R/(1-beta) ]  (single machine problems only)")
        print(f"  weak duality LB <= V* at every state ?  {log['lb_valid']}")
        print(f"  duality gap 100(V*-LB)/LB : L_R q1/med/q3/max = {np.percentile(dg[mask_R], 25):.3f}/"
              f"{np.percentile(dg[mask_R], 50):.3f}/{np.percentile(dg[mask_R], 75):.3f}/{dg[mask_R].max():.3f}"
              f"   ALL median={np.median(dg):.3f}  J-gap muL={100 * (V_opt.mean() - LB.mean()) / LB.mean():.3f}%"
              f" muR={100 * (V_opt[mask_R].mean() - LB[mask_R].mean()) / LB[mask_R].mean():.3f}%")
        for tag in ("Aware", "Unaware"):
            for base in ("ClosedForm", "PI"):
                v = results.get(f"{base}_{tag}")
                if v is not None:
                    g = v["gapLB_LR"]
                    print(f"  {base + '_' + tag:19s} gap-to-LB per-state (L_R) q1/med/q3 = {np.percentile(g, 25):.3f}/"
                          f"{np.percentile(g, 50):.3f}/{np.percentile(g, 75):.3f}   J-gap muL={v['gapLB_J_L']:.3f}% "
                          f"muR={v['gapLB_J_R']:.3f}%   (= policy gap to V* + duality gap)")
        g = results["Greedy"]["gapLB_LR"]
        print(f"  {'Greedy':19s} gap-to-LB per-state (L_R) q1/med/q3 = {np.percentile(g, 25):.3f}/"
              f"{np.percentile(g, 50):.3f}/{np.percentile(g, 75):.3f}")
        if log.get("lb_bp"):
            print("  Bound using ONLY each method's own breakpoints (no safety grid), aware indices:")
            for name, o in log["lb_bp"].items():
                print(f"    LB with {name:<10s} breakpoints only: J={o['J']:.5f}  "
                      f"max|LB-LB_PI|={o['maxdiff']:.2e}  valid={o['valid']}")

    state_str = make_state_str_fn(L_list, inst["labels"])
    avail = [m for m in UNIFORM_METHODS if results.get(m) is not None]
    if cfg.verbose_policies:
        print_action_legend(log, cfg)
        if cfg.print_state_legend:
            print_state_legend(log, cfg, state_str)
        if cfg.print_state_action_tables:
            print_policy_value_table(results, log, cfg, state_str, None,
                                     "ALL POLICIES AND THEIR VALUE FUNCTIONS, state by state (first / middle / last states)")
    if cfg.print_action_diffs:
        scope_idx = np.where(mask_R)[0] if cfg.diff_scope == "LR" else np.arange(S)
        mat = np.stack([results[m]["pi"] for m in avail])
        dis_all = (mat != mat[0:1]).any(0)
        dis = scope_idx[dis_all[scope_idx]]
        print_policy_value_table(
            results, log, cfg, state_str, dis,
            f"STATES WHERE THE POLICIES DIFFER (not all methods agree): {int(dis_all.sum())}/{S} states "
            f"({100 * dis_all.mean():.2f}% of L; {int(dis_all[mask_R].sum())}/{int(mask_R.sum())} in L_R); "
            f"rows listed over scope {cfg.diff_scope}")
        if SHOW_AWARE_VS_UNAWARE_LISTS:
            print_aware_vs_unaware_diffs(results, log, cfg, state_str, scope_idx)
        print_cross_method_diffs(results, log, cfg, state_str, scope_idx)

    ao = results["OPT"]["avg_L"]
    print("\n" + "-" * 92)
    print("VALUE SUMMARY  (V = expected discounted COST; J_muL = avg_s V(s) = expected cost from a UNIFORMLY random start,")
    print("                J_muR = same average over L_R only; Delta = 100 (J - J_OPT)/J_OPT)")
    print("-" * 92)
    o = results["OPT"]
    print(f"sum(V_OPT) = {o['sum']:.3f}   J_muL(OPT) = {ao:.4f}   J_muR(OPT) = {o['avg_R']:.4f}")
    print(f"{'':18s} min V*: L={o['min_V_L']:.4f} R={o['min_V_R']:.4f}   "
          f"max V*: L={o['max_V_L']:.4f} R={o['max_V_R']:.4f}")
    for method in UNIFORM_METHODS:
        v = results.get(method)
        if method == "OPT":
            continue
        if v is None:
            print(f"  {method:18s} n/a (method unavailable)")
            continue
        print(f"  {method:18s} sum(V) = {v['sum']:.3f}  J_muL = {v['avg_L']:.4f}  J_muR = {v['avg_R']:.4f}  "
              f"Delta_muL = {v['delta_mu_L']:.4f}%  Delta_muR = {v['delta_mu_R']:.4f}%")
        print(f"  {'':18s} min V: L={v['min_V_L']:.4f} R={v['min_V_R']:.4f}   "
              f"max V: L={v['max_V_L']:.4f} R={v['max_V_R']:.4f}   "
              f"DminV_L={v['delta_min_L']:.4f}% DmaxV_L={v['delta_max_L']:.4f}% "
              f"DminV_R={v['delta_min_R']:.4f}% DmaxV_R={v['delta_max_R']:.4f}%")
    if S <= cfg.policy_dump_max_states:
        print(f"\nOptimal (DP) policy pi_star, one action per joint state (S={S}):\n    {log['pi_star'].tolist()}")
        print(f"V_opt ({S} states): {np.round(log['V_opt'], 3).tolist()}")
        for method in UNIFORM_METHODS:
            v = results.get(method)
            if method == "OPT" or v is None:
                continue
            print(f"{method:18s} policy pi ({S} states): {v['pi'].tolist()}")
            print(f"{method:18s} V        ({S} states): {np.round(v['V'], 3).tolist()}")
    else:
        print(f"\n(S={S} > policy_dump_max_states={cfg.policy_dump_max_states}: full pi / V list dump skipped; "
              f"the state-by-state tables above show them)")

    print("\nVALIDATION (do the index methods give the same joint policy?  reference = PI)")
    for tag in ("Aware", "Unaware"):
        rec = log["validation"][tag]
        print(f"  [{tag.upper()}] all available methods identical: {rec['all5']}   details: "
              + ", ".join(f"{m}:{('n/a' if a is None else ('same' if a else str(rec['ndiff'][m]) + ' diff'))}"
                          for m, a in rec["agree"].items()))
    print("\nSanity: V_method >= V_opt everywhere (no policy beats OPT):")
    for method in UNIFORM_METHODS:
        v = results.get(method)
        if method != "OPT" and v is not None:
            print(f"  {method:18s}: {bool(np.all(v['subopt_ALL'] >= -1e-6))}")
    print()


# =====================================================================
# 13. Large-M route: no joint matrices, no OPT.  LB vs Monte Carlo cost
# =====================================================================
def run_cell_bound(cfg, D, alpha, switch_frac, instance_seed, n_states=100, n_paths=100,
                   method="PI", verbose=True):
    """method = which index family drives the Index_* policies: 'PI' (definition-based) or
    'ClosedForm' (the closed-form expressions of the paper)."""
    inst = build_instance(cfg, D, alpha, switch_frac, instance_seed)
    beta, R, M, locs = cfg.beta, cfg.R, inst["M"], inst["machine_locals"]
    L_list = [ml["P0"].shape[0] for ml in locs]
    rng = np.random.default_rng(instance_seed + 10 ** 6)
    states = sample_reachable_states(L_list, R, n_states, rng)
    cand = lagrangian_candidates(inst["W"]["Aware"]["PI"], [mp["K"] for mp in inst["machine_params"]],
                                 beta, cfg.lb_extra_grid)
    LB, _ = lagrangian_lower_bound(locs, cand, states, beta, R)
    levels = [damage_level(L_list[m], inst["N_list"][m]) for m in range(M)]
    pols = {"Index_Aware": lambda x: index_actions(inst["W"]["Aware"][method], x, R),
            "Index_Unaware": lambda x: index_actions(inst["W"]["Unaware"][method], x, R),
            "Greedy": lambda x: greedy_actions(levels, x, R)}
    out = {"instance_seed": instance_seed, "LB_mean": LB.mean()}
    ests = {}
    for name, fn in pols.items():
        mean, se, est = simulate_cost(locs, fn, states, beta, n_paths, instance_seed + 2 * 10 ** 6)
        ests[name] = est
        gap = 100 * (mean - LB) / LB
        out[name] = dict(J_gap=100 * (mean.mean() - LB.mean()) / LB.mean(), J=mean.mean(),
                         J_se=np.sqrt(np.sum(se ** 2)) / n_states, gap_q=np.percentile(gap, [25, 50, 75]))
    d = ests["Index_Unaware"] - ests["Index_Aware"]            # paired, common random numbers
    out["unaware_minus_aware"] = (d.mean(), np.sqrt(np.sum((d.std(1, ddof=1) / np.sqrt(n_paths)) ** 2)) / n_states)
    if verbose:
        print(f"[bound] seed={instance_seed} M={M} R={R} switch_frac={switch_frac_cell_label(switch_frac)} "
              f"D={D} alpha={alpha} index={method}  mean LB={LB.mean():.3f}  ({n_states} states of L_R x {n_paths} paths)")
        for name in pols:
            o = out[name]
            print(f"   {name:14s} J={o['J']:.3f}+-{o['J_se']:.3f}  gap-to-LB(J)={o['J_gap']:.2f}%  "
                  f"per-state q1/med/q3={o['gap_q'][0]:.2f}/{o['gap_q'][1]:.2f}/{o['gap_q'][2]:.2f}%")
        print(f"   J(unaware)-J(aware) = {out['unaware_minus_aware'][0]:.3f} +- {out['unaware_minus_aware'][1]:.3f}")
    return out


def run_bound_study(cfg, cells, n_reps=3, **kw):
    rows, k = [], 0
    for sf in cfg.switch_fracs:
        for (D, alpha) in cells:
            for _ in range(n_reps):
                k += 1
                o = run_cell_bound(cfg, D, alpha, sf, cfg.seed0 + 50000 + k, **kw)
                rows.append(dict(switch_frac=switch_frac_cell_label(sf), D=D, alpha=alpha,
                                 gap_aware=o["Index_Aware"]["J_gap"], gap_unaware=o["Index_Unaware"]["J_gap"],
                                 gap_greedy=o["Greedy"]["J_gap"], unaware_minus_aware=o["unaware_minus_aware"][0]))
    df = pd.DataFrame(rows).groupby(["switch_frac", "D", "alpha"]).mean(numeric_only=True).reset_index()
    print("\n" + "=" * 100)
    print("LARGE-M STUDY: gap of J(policy) to the Lagrangian lower bound (%), mean over reps (muR states)")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def mc_vs_exact_check(cfg, D, alpha, switch_frac, seed, n_states=20, n_paths=400):
    """Sanity check of the Monte Carlo evaluator against exact V on a small instance."""
    results, log = run_cell_exact(cfg, D, alpha, switch_frac, seed, verbose=False)
    inst, R = log["inst"], cfg.R
    rng = np.random.default_rng(1)
    st = sample_reachable_states(log["L_list"], R, n_states, rng)
    flat = np.ravel_multi_index(st.T, tuple(log["L_list"]))
    mean, se, _ = simulate_cost(inst["machine_locals"], lambda x: index_actions(inst["W"]["Aware"]["PI"], x, R),
                                st, cfg.beta, n_paths, 5)
    z = (mean - results["PI_Aware"]["V"][flat]) / se
    print(f"[MC check] PI_Aware: max |z| = {np.abs(z).max():.2f} over {n_states} states "
          f"(should be ~<3.5);  mean rel. error = {np.mean(np.abs(mean / results['PI_Aware']['V'][flat] - 1)) * 100:.3f}%")


# =====================================================================
# 14. Sweep + aggregation
# =====================================================================
def run_uniform_glazebrook_table(cfg, verbose=True):
    R = cfg.R
    sub_rows, avg_rows, lb_rows, bench_rows, val_rows, diag_rows, mis_rows = [], [], [], [], [], [], []
    counter = 0
    for sf in cfg.switch_fracs:
        sfl = switch_frac_cell_label(sf)
        for D in cfg.D_grid:
            for alpha in cfg.alpha_grid:
                pool = {m: {k: [] for k in ("LR", "ALL", "idLR", "idALL", "gLR", "gALL", "JL", "JR", "dL", "dR",
                                            "mLR", "mALL", "gJL", "gJR")} for m in UNIFORM_METHODS}
                n_bench = 0
                for rep in range(cfg.n_reps):
                    counter += 1
                    seed = cfg.seed0 + counter
                    print(f"\n########## switch_frac={sfl}  D={D}  alpha={alpha}  rep={rep + 1}/{cfg.n_reps}  "
                          f"instance_seed={seed} ##########")
                    res, log = run_cell_exact(cfg, D, alpha, sf, seed, verbose=verbose)
                    for m, v in res.items():
                        if v is None:
                            continue
                        P = pool[m]
                        P["LR"].append(v["subopt_LR"]); P["ALL"].append(v["subopt_ALL"])
                        P["idLR"].append(v["idle_LR"]); P["idALL"].append(v["idle_ALL"])
                        P["JL"].append(v["avg_L"]); P["JR"].append(v["avg_R"])
                        P["dL"].append(v["delta_mu_L"]); P["dR"].append(v["delta_mu_R"])
                        P["mLR"].append(v["match_LR"]); P["mALL"].append(v["match_ALL"])
                        if "gapLB_LR" in v:
                            P["gLR"].append(v["gapLB_LR"]); P["gALL"].append(v["gapLB_ALL"])
                            P["gJL"].append(v["gapLB_J_L"]); P["gJR"].append(v["gapLB_J_R"])
                    n_bench += bool(benchmark_disagreements(res))
                    # validation + diagnostics records
                    vr = log["validation"]
                    for tag in ("Aware", "Unaware"):
                        inside = vr["inside"] if tag == "Aware" else True
                        val_rows.append(dict(switch_frac=sfl, tag=tag, inside=inside, all5=vr[tag]["all5"],
                                             pkg=vr[tag]["pkg"],
                                             **{f"same_{m}": a for m, a in vr[tag]["agree"].items()},
                                             **{f"idxdiff_{m}": d for m, d in vr[tag]["idxdiff"].items()}))
                        for dg in log["inst"]["diag"][tag]:
                            diag_rows.append(dict(switch_frac=sfl, tag=tag, inside=dg["inside"],
                                                  indexable=dg["indexable"], cf_mono=dg["cf_mono"],
                                                  num_mono=dg["num_mono"], cf_verified=dg["cf_verified"],
                                                  cf_eq_num=dg["cf_eq_num"], cf_eq_thm=dg["cf_eq_thm"]))
                    # action mismatch rates: PI, Greedy AND ClosedForm (aware / unaware)
                    mk = log["mask_R"]
                    for name, (a_, b_) in MISMATCH_PAIRS.items():
                        if res.get(a_) is None or res.get(b_) is None:
                            continue
                        x, y = res[a_]["pi"], res[b_]["pi"]
                        mis_rows.append(dict(switch_frac=sfl, D=D, alpha=alpha, pair=name,
                                             LR=100 * np.mean((x != y)[mk]), ALL=100 * np.mean(x != y)))
                bench_rows.append(dict(switch_frac=sfl, D=D, alpha=alpha, n_reps=cfg.n_reps,
                                       n_benchmark_disagree=n_bench))
                for m in UNIFORM_METHODS:
                    P = pool[m]
                    if not P["LR"]:
                        continue
                    for sc in ("LR", "ALL"):
                        e, idl = np.concatenate(P[sc]), np.concatenate(P["id" + sc])
                        ip = idle_pct(idl, R)
                        q1, md, q3 = np.percentile(e, [25, 50, 75])
                        row = dict(D=D, alpha=alpha, switch_frac=sfl, Method=m, scope=sc, q1=q1, median=md,
                                   q3=q3, max=e.max(), pct_idle=ip[R], match_OPT=float(np.mean(P["m" + sc])),
                                   n_pooled=len(e))
                        row.update({f"pct_{k}_idle": v for k, v in enumerate(ip)})
                        sub_rows.append(row)
                        if P["g" + sc]:
                            g = np.concatenate(P["g" + sc])
                            lb_rows.append(dict(D=D, alpha=alpha, switch_frac=sfl, Method=m, scope=sc,
                                                q1=np.percentile(g, 25), median=np.percentile(g, 50),
                                                q3=np.percentile(g, 75), J_gap_muL=np.mean(P["gJL"]),
                                                J_gap_muR=np.mean(P["gJR"])))
                    avg_rows.append(dict(D=D, alpha=alpha, switch_frac=sfl, Method=m,
                                         J_muL=np.mean(P["JL"]), J_muR=np.mean(P["JR"]),
                                         Delta_muL=np.mean(P["dL"]), Delta_muR=np.mean(P["dR"]),
                                         Delta_muL_med=np.median(P["dL"]), Delta_muR_med=np.median(P["dR"])))
                    if verbose and m != "OPT":
                        for sc in ("LR", "ALL"):
                            r = [x for x in sub_rows[-2:] if x["scope"] == sc][0]
                            print(f"  CELL [{m}] sf={sfl} D={D} alpha={alpha} scope={sc} "
                                  f"eps%[{r['q1']:.3f},{r['median']:.3f},{r['q3']:.3f}] max={r['max']:.3f} "
                                  f"all-idle%={r['pct_idle']:.1f} (pooled {r['n_pooled']} states)")
    return dict(subopt=pd.DataFrame(sub_rows), avg=pd.DataFrame(avg_rows), lb=pd.DataFrame(lb_rows),
                bench=pd.DataFrame(bench_rows), val=pd.DataFrame(val_rows), diag=pd.DataFrame(diag_rows),
                mis=pd.DataFrame(mis_rows))


def _outside_domain_note(method, sf):
    """ClosedForm for the AWARE problem is proven only for c_switch <= c_bar (switch_frac <= 1)."""
    if method != "ClosedForm_Aware":
        return ""
    body = sf[len("mixed_"):] if sf.startswith("mixed_") else sf
    vals = [float(x) for x in body.split("-")]
    if max(vals) > 1.0:
        return "   [NOTE: c_switch > c_bar for some machine -> OUTSIDE proven domain of the closed form]"
    return ""


def _cell_table(df, cfg, sf, method, title, value_fn, width, scope=None):
    sub = df[(df["Method"] == method) & (df["switch_frac"] == sf)] if "Method" in df else df
    if scope is not None:
        sub = sub[sub["scope"] == scope]
    if sub.empty:
        return
    print("=" * 100)
    print(f"Method = {method}   switch_frac = {sf}   {title}{_outside_domain_note(method, sf)}")
    print("=" * 100)
    print("D".rjust(6) + "".join(str(a).rjust(width) for a in cfg.alpha_grid))
    for D in cfg.D_grid:
        line = str(D).rjust(6)
        for a in cfg.alpha_grid:
            r = sub[(sub["D"] == D) & (sub["alpha"] == a)]
            line += (value_fn(r.iloc[0]) if not r.empty else "n/a").rjust(width)
        print(line)
    print()


def print_aggregates(cfg, T):
    methods = [m for m in cfg.methods_for_tables() if m != "OPT"]
    sfs = [switch_frac_cell_label(s) for s in cfg.switch_fracs]
    sub, avg, lb = T["subopt"], T["avg"], T["lb"]
    for sc, desc in (("LR", "pooled over L_R ONLY (states that can recur)"),
                     ("ALL", "pooled over ALL of L (incl. states never revisited)")):
        print("\n" + "#" * 100 + f"\n# per-state eps(s) {desc}\n" + "#" * 100)
        for m in methods:
            for sf in sfs:
                _cell_table(sub, cfg, sf, m, f"eps% q1/med/q3  [scope {sc}]",
                            lambda r: f"{r['q1']:.2f} {r['median']:.2f} {r['q3']:.2f}", 22, sc)
                _cell_table(sub, cfg, sf, m, f"% states with ALL R={cfg.R} repairmen idle  [scope {sc}]",
                            lambda r: f"{r['pct_idle']:.1f}", 12, sc)
    for col, ttl in (("Delta_muL", "Delta_{mu_L}  (uniform benchmark) %"),
                     ("Delta_muR", "Delta_{mu_R}  (reachable benchmark, over L_R) %")):
        print("\n" + "#" * 100 + f"\n# {ttl}\n" + "#" * 100)
        for m in methods:
            for sf in sfs:
                _cell_table(avg, cfg, sf, m, f"mean {col}", lambda r, c=col: f"{r[c]:.3f}", 14)
    if not lb.empty:
        for sc in ("LR", "ALL"):
            print("\n" + "#" * 100 + f"\n# gap of policy cost to the LAGRANGIAN LOWER BOUND, per-state, scope {sc}\n" + "#" * 100)
            for m in methods + ["OPT"]:
                for sf in sfs:
                    _cell_table(lb, cfg, sf, m, f"100(V-LB)/LB q1/med/q3 [scope {sc}]  (OPT row = duality gap)",
                                lambda r: f"{r['q1']:.2f} {r['median']:.2f} {r['q3']:.2f}", 22, sc)

    # -------- scope / benchmark comparison -----------------------------
    print("\n" + "=" * 100)
    print("SCOPE & BENCHMARK COMPARISON (averages over all cells of a switch_frac)")
    print("=" * 100)
    rows = []
    for m in methods:
        for sf in sfs:
            a, b = sub[(sub.Method == m) & (sub.switch_frac == sf)], avg[(avg.Method == m) & (avg.switch_frac == sf)]
            if a.empty:
                continue
            rows.append(dict(Method=m, switch_frac=sf,
                             eps_med_LR=a[a.scope == "LR"]["median"].mean(), eps_med_ALL=a[a.scope == "ALL"]["median"].mean(),
                             eps_q3_LR=a[a.scope == "LR"]["q3"].mean(), eps_q3_ALL=a[a.scope == "ALL"]["q3"].mean(),
                             eps_max_LR=a[a.scope == "LR"]["max"].max(), eps_max_ALL=a[a.scope == "ALL"]["max"].max(),
                             Delta_muL=b.Delta_muL.mean(), Delta_muR=b.Delta_muR.mean()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("BENCHMARK DISAGREEMENT (instances where mu_L and mu_R rank two heuristics differently)")
    print("=" * 100)
    print(T["bench"].groupby("switch_frac", sort=False)[["n_reps", "n_benchmark_disagree"]].sum().to_string())

    print("\n" + "=" * 100)
    print("ACTION MISMATCH RATES (% of states; mean over all instances)")
    print("  PI-based pairs, Greedy, and the CLOSED-FORM index (CF) vs OPT / vs PI / aware vs unaware")
    print("  CFaware!=PIaware and CFunaware!=PIunaware: how often the closed form picks a different action")
    print("  than the definition-based index (0 inside the proven domain c_switch <= c_bar).")
    print("=" * 100)
    print(T["mis"].groupby(["switch_frac", "pair"], sort=False)[["LR", "ALL"]].mean()
          .to_string(float_format=lambda x: f"{x:.2f}"))

    # -------- aggregated validation ------------------------------------
    val, dg = T["val"], T["diag"]
    print("\n" + "=" * 100)
    print("AGGREGATED VALIDATION: do ClosedForm, Theorem, VI, PI, Package return the SAME joint policy?")
    print("  (one row per switch_frac x aware/unaware x inside/outside the proven domain c_switch<=c_bar;")
    print("   Package column counts only instances where markovianbandit was available)")
    print("=" * 100)
    g = val.groupby(["switch_frac", "tag", "inside"], sort=False)
    rows = []
    for k, d in g:
        r = dict(switch_frac=k[0], tag=k[1], inside=k[2], n_inst=len(d), n_all_same=int(d["all5"].sum()),
                 pct_all_same=100 * d["all5"].mean(), n_pkg=int(d["pkg"].sum()))
        for m in ("ClosedForm", "Theorem", "VI", "Package"):
            r[f"max|W-W_PI|_{_SHORT[m]}"] = d[f"idxdiff_{m}"].max()
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2e}" if abs(x) < 0.01 and x != 0 else f"{x:.2f}"))

    print("\n" + "=" * 100)
    print("AGGREGATED INDEXABILITY / MONOTONICITY DIAGNOSTICS (per MACHINE, counts)")
    print("=" * 100)
    cols = ["indexable", "cf_mono", "num_mono", "cf_verified", "cf_eq_num", "cf_eq_thm"]
    keys = ["switch_frac", "tag", "inside"]
    t = dg.groupby(keys, sort=False)[cols].sum()
    t.insert(0, "n_machines", dg.groupby(keys, sort=False).size())
    print(t.to_string())


def latex_perftable(sub, cfg, sf_label, methods_labels, scope="LR"):
    """Rows for \\begin{perftable}{label}{frac}{n} ... \\end{perftable}.  Needs 6 alpha columns."""
    if len(cfg.alpha_grid) != 6:
        return f"% skipped: perftable needs 6 alpha columns, have {len(cfg.alpha_grid)}\n"
    out = [f"\\begin{{perftable}}{{tab:sw_{sf_label}}}{{{sf_label}}}{{{cfg.n_reps}}}"]
    for method, label in methods_labels:
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}}\\\\")
        for D in cfg.D_grid:
            cells = []
            for a in cfg.alpha_grid:
                r = sub[(sub.Method == method) & (sub.D == D) & (sub.alpha == a) &
                        (sub.switch_frac == sf_label) & (sub.scope == scope)]
                if r.empty:
                    cells.append("--")
                else:
                    # NOTE: access by key.  r.iloc[0].median is the pandas Series METHOD, not the column.
                    row = r.iloc[0]
                    cells.append(f"\\pc{{{row['q1']:.2f}}}{{{row['median']:.2f}}}{{{row['q3']:.2f}}}"
                                 f"{{{row['pct_idle']:.1f}}}")
            out.append(f"\\perfrow{{{D}}}{{" + "}{".join(cells) + "}")
        out.append("\\midrule")
    out.append("\\end{perftable}")
    return "\n".join(out) + "\n"


LATEX_METHODS = [("ClosedForm_Aware", "Closed-form index (aware)"),
                 ("ClosedForm_Unaware", "Closed-form index (unaware)"),
                 ("PI_Aware", "Index policy PI (aware)"),
                 ("PI_Unaware", "Index policy PI (unaware)"),
                 ("Greedy", "Greedy")]


# =====================================================================
# 15. main
# =====================================================================
def main():
    cfg = ExperimentConfig(
        N_list=[7, 7, 7, 7], R=2, beta=0.95, n_reps=5,
        D_grid=[0, 10, 20, 25, 30, 40, 50], alpha_grid=[1.5, 2.0, 3.0, 4.0, 5.0, "Various"],
        various_alpha_multipliers=[1.5, 3.0, 4.5, 6.0], C_width=25.0,
        switch_fracs=[0.5, 0.99, 1.01, [0.3, 0.6, 0.8, 0.2], [0.3, 0.6, 0.8, 1.2], 1.2],
        seed0=1000, state_print_chunk=20, policy_dump_max_states=1000,
        print_state_legend=False, print_state_action_tables=True,
        table_methods=["ClosedForm_Aware", "ClosedForm_Unaware",
                       "PI_Aware", "PI_Unaware", "Greedy"])
    RUN_INDEXABILITY_STUDY, RUN_EXACT, RUN_BOUND_DEMO, RUN_MC_CHECK = True, True, True, True

    old = sys.stdout
    logfile = open(cfg.output_txt, "w", encoding="utf-8")
    sys.stdout = TeeWriter(old, logfile)
    try:
        print(f"Run started: {datetime.now().isoformat()}")
        print(f"python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} "
              f"| scipy sparse: {SCIPY} | markovianbandit: {PKG_AVAILABLE}")
        print(f"Config: {cfg}\n")

        if RUN_INDEXABILITY_STUDY:
            indexability_study(cfg.beta, N=7, n_machines=100, n_grid=300)

        if RUN_EXACT:
            T = run_uniform_glazebrook_table(cfg, verbose=True)
            print("\n\n" + "=" * 100 + "\nAGGREGATE RESULTS\n" + "=" * 100)
            print_aggregates(cfg, T)
            tex = "".join(latex_perftable(T["subopt"], cfg, switch_frac_cell_label(s), LATEX_METHODS)
                          for s in cfg.switch_fracs if not isinstance(s, (list, tuple)))
            with open(cfg.output_txt.replace(".txt", ".tex"), "w", encoding="utf-8") as f:
                f.write(tex)
            print("\nLaTeX rows written to", cfg.output_txt.replace(".txt", ".tex"))
            if RUN_MC_CHECK:
                mc_vs_exact_check(cfg, 20, 4.0, 0.5, cfg.seed0 + 1)

        if RUN_BOUND_DEMO:       # M too large for the joint MDP: LB vs Monte Carlo
            big = ExperimentConfig(N_list=[7] * 8, R=3, beta=0.95, n_reps=1, D_grid=[20], alpha_grid=[4.0],
                                   switch_fracs=[0.5, 1.2], seed0=cfg.seed0, n_grid=200)
            # index policies driven by the CLOSED-FORM indices (the novelty); use method="PI" for the
            # definition-based ones.  Outside the domain (switch_frac 1.2) the closed form may deviate.
            run_bound_study(big, cells=[(20, 4.0), (50, 1.5)], n_reps=2, n_states=60, n_paths=60,
                            method="ClosedForm")

        print(f"\nRun finished: {datetime.now().isoformat()}\nTranscript: {cfg.output_txt}")
    finally:
        sys.stdout = old
        logfile.close()


if __name__ == "__main__":
    main()
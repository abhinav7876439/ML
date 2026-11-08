"""
lagrange_index_policy.py  --  add-on for Repairman_discounted.py (put both files in the same folder)

Contents
--------
 1. joint_size_report / print_size_table : how big is the exact joint DP (states, actions, nonzeros,
                                           memory) and a ROUGH time estimate, for any (N_list, R).
 2. benchmark_exact_dp                   : really builds + solves the joint DP for M = 2,3,... and times it
                                           (stops gracefully on MemoryError).
 3. Lagrange index policy (LIP)          : Avrachenkov-Borkar-Shah (2025) and Brown-Smith (2020).
        cost-minimisation convention of Repairman_discounted.py, charge W >= 0 for activating a machine:
            Q0_m(x;W) = c0 + beta P0 V_m(.;W)            (passive)
            Q1_m(x;W) = c1 + W + beta P1 V_m(.;W)        (active, pays the charge W)
            tau_m(x;W) = Q0 - Q1                         (>0  <=>  activating is strictly better at charge W)
        Lagrangian dual   D(s;W) = sum_m V_m(s_m;W) - W R/(1-beta),   LB(s) = max_{W>=0} D(s;W) <= V*(s)
        W*(s) = argmax_W D(s;W)           (the same W for every machine: ONE scalar, no indexability needed)
        LIP priority of machine m in state x   :  a_m = Q0 - (c1 + beta P1 V) = tau_m(x;W*) + W*
                                                  (value of activating, charge excluded; = Brown-Smith index (13))
        LIP action : activate the (at most R) machines with the largest a_m, only those with a_m > 0.
    Variants implemented (all deterministic, so their value function can be evaluated EXACTLY):
        LIP_reopt_*  : W*(s) recomputed at the current joint state s (= 're-optimisation' of Brown-Smith 6.4)
        LIP_muL_*    : ONE fixed W maximising the mean dual over all of L      (single scalar, as in ABS 2025)
        LIP_muR_*    : ONE fixed W maximising the mean dual over L_R
        suffix _index   : ties in a_m broken by lowest machine index
        suffix _whittle : ties in a_m broken by the (aware) Whittle index
    IMPORTANT LIMITATIONS
      * Ties at a_m are generic at the optimal W* (kink of the dual).  Brown-Smith break them with an optimal
        MIXED policy (proportional assignment / Markov mixing, their EC1.2).  EC1.2 is not in the PDF we have, so
        here only deterministic tie-breakers are implemented.  Random tie-breaking is known to be able to destroy
        the sqrt(N) rate.
      * Brown-Smith use TIME-VARYING multipliers lambda_t (tighter bound, cutting-plane method).  Here one
        time-invariant W is used (Whittle / Bertsimas-Nino-Mora first-order relaxation).
 4. run_lip_exact / run_lip_study        : small M: OPT vs Whittle vs LIP vs Greedy, gap to the lower bound.
 5. run_lip_large_M                      : large M (no joint matrices): Monte Carlo cost of Whittle / LIP /
                                           Greedy versus the Lagrangian lower bound, paired differences.

NOTHING in this file has been executed by the author of this answer; time numbers printed by
joint_size_report are back-of-envelope estimates, benchmark_exact_dp measures the real ones on YOUR machine.
"""

import math
import time
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from Repairman_discounted import (
    ExperimentConfig, build_instance, build_joint_mdp_hetero, solve_joint_vi,
    evaluate_joint_policy, joint_states, reachable_mask, index_actions, greedy_actions,
    act_code, damage_level, make_result, idle_pct, batch_pi, lagrangian_candidates,
    simulate_cost, sample_reachable_states)


# =====================================================================
# helpers
# =====================================================================
def make_cfg(M, R, N=7, beta=0.95, n_grid=200, seed0=1000):
    """Minimal ExperimentConfig for these experiments (alpha_grid has no 'Various', so any M works)."""
    return ExperimentConfig(N_list=[N] * M, R=R, beta=beta, n_reps=1, D_grid=[20], alpha_grid=[4.0],
                            switch_fracs=[0.5], seed0=seed0, n_grid=n_grid, use_vi=False,
                            compute_LB=True)


# =====================================================================
# 1. size / time of the exact joint DP
# =====================================================================
def joint_size_report(N_list, R, beta=0.95, tol=1e-10, nnz_per_sec=5e8, flops_per_sec=2e9):
    """Exact counts for the explicit Kronecker/sparse DP of Repairman_discounted.py, plus rough times.

    * joint states            S = prod (N_m + 2)
    * joint actions           A = sum_{k<=R} C(M,k)
    * nonzeros of action a    = prod_m nnz(P^{a_m}_m),  nnz(P0_m) = 2 N_m + 3 (all 0<p<1),  nnz(P1_m) = N_m + 2
    * explicit sparse memory  ~ 12 bytes per nonzero (float64 + int32 index) + A cost vectors of length S
    * VI sweeps               ~ ln(tol)/ln(beta)  (about 450 for tol=1e-10, beta=0.95)
    * time estimates          ROUGH: nnz_per_sec = sparse matvec throughput, flops_per_sec = dense tensordot
    """
    M = len(N_list)
    L = [N + 2 for N in N_list]
    S = 1
    for l in L:
        S *= l
    A = sum(comb(M, k) for k in range(min(R, M) + 1))
    nnz_p = [2 * N + 3 for N in N_list]
    nnz_a = L
    nnz_total = 0
    for k in range(min(R, M) + 1):
        for combo in combinations(range(M), k):
            cs, t = set(combo), 1
            for m in range(M):
                t *= nnz_a[m] if m in cs else nnz_p[m]
            nnz_total += t
    sweeps = math.ceil(math.log(tol) / math.log(beta))
    return dict(M=M, R=R, S=S, A=A, nnz_total=nnz_total,
                mem_sparse_GB=(12 * nnz_total + 8 * A * S) / 1e9,
                mem_matrixfree_GB=4 * 8 * S / 1e9,
                sweeps=sweeps,
                t_VI_sparse_min=sweeps * nnz_total / nnz_per_sec / 60,
                t_VI_matrixfree_min=sweeps * A * M * 2 * max(L) * S / flops_per_sec / 60)


def print_size_table(Ms=(3, 4, 5, 6, 7, 8, 9, 10), N=7, R_of_M=lambda M: 2, beta=0.95):
    rows = [joint_size_report([N] * M, min(R_of_M(M), M), beta) for M in Ms]
    df = pd.DataFrame(rows)
    print("=" * 110)
    print(f"EXACT JOINT DP SIZE  (N={N} -> {N + 2} local states, beta={beta}).  Times are ROUGH estimates "
          "of value-iteration sweeps ONLY\n(the full per-instance routine - polishing, ~12 policy evaluations, LB - "
          "costs roughly 2-4x more).  sparse = what Repairman_discounted.py does.")
    print("=" * 110)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.3g}"))
    return df


def benchmark_exact_dp(Ms=(2, 3, 4, 5), N=7, R=2, seed=1, beta=0.95):
    """Build + solve the real joint DP for increasing M and time it.  Add M=6 only with >= 16 GB RAM."""
    rows = []
    for M in Ms:
        cfg = make_cfg(M, min(R, M), N=N, beta=beta)
        est = joint_size_report(cfg.N_list, cfg.R, beta)
        row = dict(M=M, R=cfg.R, S=est["S"], A=est["A"], nnz=est["nnz_total"],
                   est_VI_min=est["t_VI_sparse_min"])
        try:
            inst = build_instance(cfg, 20, 4.0, 0.5, seed)
            t0 = time.perf_counter()
            P_joint, C_joint, L_list, subsets, lut = build_joint_mdp_hetero(inst["machine_locals"], cfg.R)
            t1 = time.perf_counter()
            V, pi = solve_joint_vi(P_joint, C_joint, beta)
            t2 = time.perf_counter()
            row.update(t_build_s=t1 - t0, t_solve_s=t2 - t1, mean_Vstar=float(V.mean()))
            del P_joint, C_joint
        except MemoryError:
            row.update(t_build_s=np.nan, t_solve_s=np.nan, mean_Vstar=np.nan)
            print(f"  [MemoryError at M={M}]")
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return pd.DataFrame(rows)


# =====================================================================
# 3. Lagrange index policy
# =====================================================================
def lip_tables(machine_locals, cand_W, beta):
    """tau_m[ci, x] = Q0 - Q1 at charge cand_W[ci]   (Q1 includes the charge)."""
    tabs = []
    for ml in machine_locals:
        _, Q0, Q1 = batch_pi(ml["P0"], ml["P1"], ml["c0"], ml["c1"], cand_W, beta)
        tabs.append(Q0 - Q1)
    return tabs


def lagrangian_dual_argmax(machine_locals, cand_W, states_idx, beta, R, weights=None):
    """For every joint state s in states_idx:
         LB(s)      = max_{W in cand} D(s;W),   D(s;W) = sum_m V_m(s_m;W) - W R/(1-beta)
         ci_star(s) = index (into cand_W) of the maximiser            (W*(s) = cand_W[ci_star])
       weights: optional dict name -> weight vector over states; returns dbar[name][ci] = sum_s w_s D(s;W_ci),
       whose argmax is the single fixed multiplier for that state distribution."""
    n, M = states_idx.shape
    Vtab = [batch_pi(ml["P0"], ml["P1"], ml["c0"], ml["c1"], cand_W, beta)[0] for ml in machine_locals]
    weights = weights or {}
    LB = np.full(n, -np.inf)
    ci_star = np.zeros(n, dtype=int)
    dbar = {k: np.zeros(len(cand_W)) for k in weights}
    for ci, w in enumerate(cand_W):
        Dv = sum(Vtab[m][ci][states_idx[:, m]] for m in range(M)) - w * R / (1 - beta)
        for k, wv in weights.items():
            dbar[k][ci] = float(np.dot(wv, Dv))
        upd = Dv > LB
        LB[upd], ci_star[upd] = Dv[upd], ci
    return LB, ci_star, dbar


def whittle_tie_key(W_list, x):
    """Secondary sort key: Whittle index of each machine's current local state (larger = served first)."""
    return np.stack([np.nan_to_num(np.asarray(W_list[m], float), posinf=1e12, neginf=-1e12)[x[:, m]]
                     for m in range(x.shape[1])], axis=1)


def lip_actions(tau_tabs, cand_W, ci_vec, x, R, tol=1e-9, tie_key=None):
    """Lagrange index policy.
       x: (P, M) local states;  ci_vec: (P,) index of the multiplier used at each state.
       priority a_m = tau_m(x_m; W) + W = Q0 - (c1 + beta P1 V)  (charge excluded);
       activate the <= R largest priorities, only if a_m > tol.  Ties (equal a_m up to 1e-8) are
       broken by tie_key (larger first), then by lowest machine index."""
    P, M = x.shape
    Wv = np.asarray(cand_W)[ci_vec]
    tau = np.stack([tau_tabs[m][ci_vec, x[:, m]] for m in range(M)], axis=1)
    adv = tau + Wv[:, None]
    prim = -np.round(adv, 8)
    sec = np.zeros_like(adv) if tie_key is None else -np.asarray(tie_key, float)
    order = np.lexsort((sec, prim), axis=1)[:, :min(R, M)]
    rows = np.arange(P)[:, None]
    act = np.zeros((P, M), dtype=bool)
    act[rows, order] = adv[rows, order] > tol
    return act


def summarize(results, R):
    rows = []
    for name, v in results.items():
        if v is None:
            continue
        e = v["subopt_LR"]
        rows.append(dict(Method=name, LR_med=np.percentile(e, 50), LR_q3=np.percentile(e, 75), LR_max=e.max(),
                         DminV_L=v["delta_min_L"], Dmu_L=v["delta_mu_L"], DmaxV_L=v["delta_max_L"],
                         DminV_R=v["delta_min_R"], Dmu_R=v["delta_mu_R"], DmaxV_R=v["delta_max_R"],
                         matchLR=v["match_LR"], allIdleLR=idle_pct(v["idle_LR"], R)[R],
                         gapLB_JL=v["gapLB_J_L"], gapLB_JR=v["gapLB_J_R"]))
    return pd.DataFrame(rows)


# =====================================================================
# 4. exact route (small M): OPT vs Whittle vs LIP vs Greedy, gap to the lower bound
# =====================================================================
def run_lip_exact(cfg, D, alpha, switch_frac, seed, verbose=True):
    inst = build_instance(cfg, D, alpha, switch_frac, seed)
    beta, R, M = cfg.beta, cfg.R, inst["M"]
    locs = inst["machine_locals"]
    P_joint, C_joint, L_list, subsets, lut = build_joint_mdp_hetero(locs, R)
    n_active = np.array([len(c) for c in subsets])
    S = int(np.prod(L_list))
    V_opt, pi_star = solve_joint_vi(P_joint, C_joint, beta)
    states, mask_R = joint_states(L_list), reachable_mask(L_list, R)

    K_list = [mp["K"] for mp in inst["machine_params"]]
    Wpi_A, Wpi_U = inst["W"]["Aware"]["PI"], inst["W"]["Unaware"]["PI"]
    cand = lagrangian_candidates(Wpi_A, K_list, beta, cfg.lb_extra_grid)
    weights = {"muL": np.full(S, 1.0 / S), "muR": mask_R / mask_R.sum()}
    LB, ci_s, dbar = lagrangian_dual_argmax(locs, cand, states, beta, R, weights)
    ci_fix = {k: int(np.argmax(v)) for k, v in dbar.items()}
    tabs = lip_tables(locs, cand, beta)

    cache = {}

    def evaluate(pi):
        k = pi.tobytes()
        if k not in cache:
            cache[k] = evaluate_joint_policy(pi, P_joint, C_joint, beta)
        return cache[k]

    pols = {"Whittle_Aware": lut[act_code(index_actions(Wpi_A, states, R))],
            "Whittle_Unaware": lut[act_code(index_actions(Wpi_U, states, R))]}
    for tie in ("index", "whittle"):
        key = whittle_tie_key(Wpi_A, states) if tie == "whittle" else None
        pols[f"LIP_reopt_{tie}"] = lut[act_code(lip_actions(tabs, cand, ci_s, states, R, tie_key=key))]
        for k, ci in ci_fix.items():
            pols[f"LIP_{k}_{tie}"] = lut[act_code(lip_actions(tabs, cand, np.full(S, ci), states, R,
                                                              tie_key=key))]
    levels = [damage_level(L_list[m], inst["N_list"][m]) for m in range(M)]
    pols["Greedy"] = lut[act_code(greedy_actions(levels, states, R))]

    results = {"OPT": make_result(pi_star, V_opt, V_opt, pi_star, n_active, R, mask_R, LB)}
    for name, pi in pols.items():
        results[name] = make_result(pi, evaluate(pi), V_opt, pi_star, n_active, R, mask_R, LB)

    df = summarize(results, R)
    if verbose:
        Ws = cand[ci_s]
        print("=" * 100)
        print(f"LIP vs Whittle vs OPT   M={M} R={R} N_list={inst['N_list']} D={D} alpha={alpha} "
              f"switch_frac={switch_frac} seed={seed}   |L|={S}  |L_R|={int(mask_R.sum())}")
        print("=" * 100)
        print(f"  W*(s) (state-dependent multiplier): zero in {100 * np.mean(Ws == 0):.1f}% of states "
              f"(constraint not binding there), median {np.median(Ws):.3f}, max {Ws.max():.3f}")
        print("  fixed multipliers: " + ", ".join(f"{k}: W={cand[c]:.4f}" for k, c in ci_fix.items()))
        print(f"  J_muL(OPT)={V_opt.mean():.4f}   J_muL(LB)={LB.mean():.4f}   "
              f"duality gap (muL) = {100 * (V_opt.mean() - LB.mean()) / LB.mean():.3f}%")
        print("  gapLB_J* = 100 (J(policy) - J(LB))/J(LB)  ~  Delta_mu + duality gap;  Delta/DminV/DmaxV as in "
              "Repairman_discounted.py")
        with pd.option_context("display.width", 300, "display.max_columns", None):
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return results, df


def run_lip_study(M=4, R=2, N=7, fracs=(0.5, 0.99, 1.2, 2.0), reps=3, D=20, alpha=4.0, seed0=1000):
    """Mean Delta_muR (%) of every policy for several switch_frac (same instances across switch_frac)."""
    cfg = make_cfg(M, R, N=N, seed0=seed0)
    rows = []
    for sf in fracs:
        for r in range(reps):
            res, _ = run_lip_exact(cfg, D, alpha, sf, seed0 + r, verbose=False)
            for name, v in res.items():
                if name != "OPT" and v is not None:
                    rows.append(dict(sf=sf, Method=name, Dmu_R=v["delta_mu_R"], Dmu_L=v["delta_mu_L"],
                                     gapLB_JR=v["gapLB_J_R"]))
    df = pd.DataFrame(rows).groupby(["sf", "Method"], sort=False).mean().reset_index()
    print("=" * 100)
    print(f"LIP STUDY  M={M} R={R} N={N} D={D} alpha={alpha}  reps={reps}   mean Delta_muR (%) by switch_frac")
    print("=" * 100)
    piv = df.pivot(index="Method", columns="sf", values="Dmu_R")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nmean gap to the Lagrangian lower bound, J_muR (%):")
    print(df.pivot(index="Method", columns="sf", values="gapLB_JR").to_string(float_format=lambda x: f"{x:.4f}"))
    return df


# =====================================================================
# 5. large M: Monte Carlo cost of Whittle / LIP / Greedy versus the Lagrangian lower bound
# =====================================================================
def lip_action_fn(tabs, Vtabs, cand, R, beta, mode="reopt", ci_fixed=None, tie_W=None):
    """Action function for simulate_cost.  mode='reopt': W*(x) recomputed at every visited joint state x
    (max over the candidate grid of the dual); mode='fixed': one multiplier cand[ci_fixed]."""
    pen = (np.asarray(cand) * R / (1 - beta))[:, None]

    def fn(x):
        P = x.shape[0]
        if mode == "reopt":
            Dm = sum(Vt[:, x[:, m]] for m, Vt in enumerate(Vtabs)) - pen      # (nW, P)
            ci = Dm.argmax(axis=0)
        else:
            ci = np.full(P, ci_fixed)
        key = None if tie_W is None else whittle_tie_key(tie_W, x)
        return lip_actions(tabs, cand, ci, x, R, tie_key=key)
    return fn


def run_lip_large_M(cfg, D, alpha, switch_frac, seed, n_states=60, n_paths=60, verbose=True):
    inst = build_instance(cfg, D, alpha, switch_frac, seed)
    beta, R, M, locs = cfg.beta, cfg.R, inst["M"], inst["machine_locals"]
    L_list = [ml["P0"].shape[0] for ml in locs]
    rng = np.random.default_rng(seed + 10 ** 6)
    states = sample_reachable_states(L_list, R, n_states, rng)
    K_list = [mp["K"] for mp in inst["machine_params"]]
    Wpi_A = inst["W"]["Aware"]["PI"]
    cand = lagrangian_candidates(Wpi_A, K_list, beta, cfg.lb_extra_grid)
    LB, _, dbar = lagrangian_dual_argmax(locs, cand, states, beta, R, {"mu": np.full(n_states, 1.0 / n_states)})
    ci_bar = int(np.argmax(dbar["mu"]))
    Vtabs = [batch_pi(ml["P0"], ml["P1"], ml["c0"], ml["c1"], cand, beta)[0] for ml in locs]
    tabs = lip_tables(locs, cand, beta)
    levels = [damage_level(L_list[m], inst["N_list"][m]) for m in range(M)]

    pols = {"Whittle_Aware": lambda x: index_actions(Wpi_A, x, R),
            "LIP_fixed_index": lip_action_fn(tabs, Vtabs, cand, R, beta, "fixed", ci_bar),
            "LIP_fixed_whittle": lip_action_fn(tabs, Vtabs, cand, R, beta, "fixed", ci_bar, Wpi_A),
            "LIP_reopt_index": lip_action_fn(tabs, Vtabs, cand, R, beta, "reopt"),
            "LIP_reopt_whittle": lip_action_fn(tabs, Vtabs, cand, R, beta, "reopt", None, Wpi_A),
            "Greedy": lambda x: greedy_actions(levels, x, R)}
    out, ests = {}, {}
    for name, fn in pols.items():
        mean, se, est = simulate_cost(locs, fn, states, beta, n_paths, seed + 2 * 10 ** 6)   # common random numbers
        ests[name] = est
        gap = 100 * (mean - LB) / LB
        out[name] = dict(J=mean.mean(), J_se=np.sqrt(np.sum(se ** 2)) / n_states,
                         gapLB_J=100 * (mean.mean() - LB.mean()) / LB.mean(), gap_q=np.percentile(gap, [25, 50, 75]))
    if verbose:
        print(f"[large M] seed={seed} M={M} R={R} switch_frac={switch_frac} D={D} alpha={alpha}  "
              f"mean LB={LB.mean():.3f}   fixed multiplier W={cand[ci_bar]:.4f}   "
              f"({n_states} states of L_R x {n_paths} paths)")
        for name, o in out.items():
            print(f"   {name:18s} J={o['J']:.3f}+-{o['J_se']:.3f}  gap-to-LB(J)={o['gapLB_J']:.3f}%  "
                  f"per-state q1/med/q3={o['gap_q'][0]:.2f}/{o['gap_q'][1]:.2f}/{o['gap_q'][2]:.2f}%")
        base = ests["Whittle_Aware"]
        for name in pols:
            if name == "Whittle_Aware":
                continue
            d = ests[name] - base                                     # paired, common random numbers
            se = np.sqrt(np.sum((d.std(1, ddof=1) / np.sqrt(n_paths)) ** 2)) / n_states
            print(f"   J({name}) - J(Whittle_Aware) = {d.mean():+.4f} +- {se:.4f}")
    return out


# =====================================================================
# main
# =====================================================================
def main():
    RUN_SIZES, RUN_BENCH, RUN_EXACT, RUN_STUDY, RUN_LARGE = True, True, True, True, True

    if RUN_SIZES:
        print_size_table(Ms=(3, 4, 5, 6, 7, 8, 9, 10), N=7, R_of_M=lambda M: 2)
        print_size_table(Ms=(6, 8, 10, 12), N=7, R_of_M=lambda M: M // 2)

    if RUN_BENCH:                       # real timings on this machine; add 6 only with >= 16 GB RAM
        print("\n" + "=" * 100 + "\nMEASURED exact-DP times (build + solve_joint_vi)\n" + "=" * 100)
        benchmark_exact_dp(Ms=(2, 3, 4, 5), N=7, R=2)

    if RUN_EXACT:
        cfg = make_cfg(M=4, R=2, N=7)
        run_lip_exact(cfg, 20, 4.0, 0.5, seed=1001)
        run_lip_exact(cfg, 20, 4.0, 1.2, seed=1001)        # outside the proven c_bar domain

    if RUN_STUDY:
        run_lip_study(M=4, R=2, N=7, fracs=(0.5, 0.99, 1.2, 2.0), reps=3)

    if RUN_LARGE:                       # M too large for the joint DP: Monte Carlo vs lower bound
        big = make_cfg(M=8, R=3, N=7, n_grid=200)
        for sf in (0.5, 1.2):
            run_lip_large_M(big, 20, 4.0, sf, seed=2001, n_states=60, n_paths=60)


if __name__ == "__main__":
    main()
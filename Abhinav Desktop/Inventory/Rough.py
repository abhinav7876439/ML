# import numpy as np
# import pandas as pd
# import markovianbandit as bandit

# # =========================
# # Model Setup
# # =========================
# max_inventory = 15
# inventory_levels = list(range(max_inventory + 1))
# a_prev_values = [0, 1]        # 0 = Idle, 1 = Producing
# actions = [0, 1]
# states = [(i, ap) for i in inventory_levels for ap in a_prev_values]
# state_index = {s: idx for idx, s in enumerate(states)}
# nS = len(states)
# ref_state = (0, 1)
# ref_idx = state_index[ref_state]

# # Cost and rate parameters
# h, p, l = 1.0, 2.0, 5.0
# gamma, lam, mu = 0.1, 0.6, 0.8
# switch_cost = 2.0

# # =========================
# # Continuous-time generator
# # =========================
# def get_rates(i, a):
#     r_up = mu if a == 1 else 0.0
#     r_down = lam + gamma * i
#     return r_up, r_down

# def build_generator(state, action):
#     """Build Q generator row for CTMC (nS x nS)"""
#     i, _ = state
#     r_up, r_down = get_rates(i, action)
#     row = np.zeros(nS)
    
#     # Identify indices of neighbors
#     s_up = state_index[(min(i+1, max_inventory), action)]
#     s_down = state_index[(max(i-1, 0), action)]
#     s_self = state_index[(i, action)]
    
#     row[s_up] = r_up
#     row[s_down] = r_down
#     row[s_self] = -(r_up + r_down)
    
#     return row

# def build_Q_matrices():
#     Q0 = np.zeros((nS, nS))
#     Q1 = np.zeros((nS, nS))
#     C0 = np.zeros(nS)
#     C1 = np.zeros(nS)
    
#     for s_idx, (i, ap) in enumerate(states):
#         Q0[s_idx, :] = build_generator((i, ap), 0)
#         Q1[s_idx, :] = build_generator((i, ap), 1)
#         base_cost = h*i + p*gamma*i
#         if i == 0:
#             base_cost += l*lam
#         C0[s_idx] = base_cost + (switch_cost if 0 != ap else 0.0)
#         C1[s_idx] = base_cost + (switch_cost if 1 != ap else 0.0)
#     return Q0, Q1, C0, C1

# # =========================
# # Stable Continuous-time RVI
# # =========================
# # def rvi_continuous_stable(Q0, Q1, C0, C1, ref_idx, tol=1e-12, max_iter=50000, eps=0.01, verbose=False):
# #     V = np.zeros(nS)
    
# #     for it in range(max_iter):
# #         V_old = V.copy()
# #         QV0 = C0 + Q0 @ V_old
# #         QV1 = C1 + Q1 @ V_old
# #         V_tilde = np.minimum(QV0, QV1)
# #         rho = V_tilde[ref_idx]
# #         V += eps * (V_tilde - rho - V)
        
# #         delta = np.max(np.abs(V - V_old))
# #         if verbose and (it % 500 == 0 or delta < tol):
# #             print(f"RVI iter {it}, delta={delta:.3e}, rho~={rho:.6f}")
# #         if delta < tol:
# #             break
    
# #     # Final policy
# #     QV0 = C0 + Q0 @ V
# #     QV1 = C1 + Q1 @ V
# #     policy = np.argmin(np.vstack([QV0, QV1]).T, axis=1)
    
# #     return V, policy, QV0, QV1, rho, it

# # =========================
# # Continuous-time RVI with subsidy
# # =========================
# def rvi_continuous_stable(
#     Q0, Q1,
#     C0, C1,
#     ref_idx,
#     subsidy=0.0,
#     tol=1e-10,
#     max_iter=50000,
#     eps=0.01,
#     verbose=False
# ):
#     """
#     Continuous-time Relative Value Iteration with subsidy on passive action.
#     Solves:
#         rho + V = min_a { c_a - w*1{a=0} + Q_a V }
#     """
#     V = np.zeros(nS)

#     for it in range(max_iter):
#         V_old = V.copy()

#         # Subsidized costs
#         C0_sub = C0 - subsidy

#         # Bellman operators
#         QV0 = C0_sub + Q0 @ V_old
#         QV1 = C1      + Q1 @ V_old

#         V_tilde = np.minimum(QV0, QV1)

#         # Relative normalization
#         rho = V_tilde[ref_idx]

#         # Stochastic-approximation update
#         V = V + eps * (V_tilde - rho - V)

#         delta = np.max(np.abs(V - V_old))
#         if verbose and (it % 1000 == 0 or delta < tol):
#             print(f"RVI iter {it}, delta={delta:.3e}, rho~={rho:.6f}")

#         if delta < tol:
#             break

#     # Final Q-values and policy
#     QV0 = C0_sub + Q0 @ V
#     QV1 = C1     + Q1 @ V
#     policy = np.argmin(np.vstack([QV0, QV1]).T, axis=1)

#     return V, policy, QV0, QV1, rho, it


# # =========================
# # Action gap Δ_w(s)
# # =========================
# def action_gap_continuous(Q0, Q1, C0, C1, s_idx, subsidy=0.0):
#     C0_sub = C0 - subsidy
#     V, policy, QV0, QV1, rho, _ = rvi_continuous_stable(Q0, Q1, C0_sub, C1, ref_idx)
#     delta = QV1[s_idx] - QV0[s_idx]
#     return delta, V

# # =========================
# # Bisection Whittle index
# # =========================
# def whittle_index_bisection_continuous(s_idx, w_low=-20, w_high=20, tol_w=1e-6, max_iter=50, verbose=False):
#     low, high = w_low, w_high
#     for it in range(max_iter):
#         mid = 0.5*(low + high)
#         delta, _ = action_gap_continuous(Q0, Q1, C0, C1, s_idx, subsidy=mid)
#         if delta > 0:
#             low = mid
#         else:
#             high = mid
#         if abs(high - low) < tol_w:
#             break
#     if verbose:
#         print(f"State {states[s_idx]} → Whittle index ≈ {(low+high)/2}")
#     return 0.5*(low + high)

# # =========================
# # Grid search for policy
# # =========================
# def scan_over_w_continuous(w_values):
#     policy_by_w = {}
#     Pb_by_w = {}
#     for w in w_values:
#         V, policy, QV0, QV1, rho, it = rvi_continuous_stable(Q0, Q1, C0-w, C1, ref_idx)
#         policy_by_w[w] = policy
#         Pb_by_w[w] = set(si for si in range(nS) if policy[si] == 0)
#     return Pb_by_w, policy_by_w

# # =========================
# # Package extraction
# # =========================
# def extract_P0_P1_R0_R1_package():
#     P0 = np.zeros((nS,nS))
#     P1 = np.zeros((nS,nS))
#     R0 = np.zeros(nS)
#     R1 = np.zeros(nS)
    
#     for si, (i, ap) in enumerate(states):
#         # Passive
#         row0 = build_generator((i, ap), 0)
#         P0[si,:] = row0
#         R0[si] = -C0[si]
#         # Active
#         row1 = build_generator((i, ap), 1)
#         P1[si,:] = row1
#         R1[si] = -C1[si]
    
#     # Normalize rows to get stochastic matrices
#     for i in range(nS):
#         s0_sum = np.sum(P0[i,:])
#         s1_sum = np.sum(P1[i,:])
#         if abs(s0_sum) > 0:
#             P0[i,:] /= s0_sum
#         if abs(s1_sum) > 0:
#             P1[i,:] /= s1_sum
#     return P0, P1, R0, R1

# # =========================
# # Main Execution
# # =========================
# Q0, Q1, C0, C1 = build_Q_matrices()

# # --- Grid search over w ---
# w_grid = np.linspace(-5, 5, 11)
# Pb_by_w, policy_by_w = scan_over_w_continuous(w_grid)

# print("\nPolicy for different w values:")
# for w in w_grid:
#     print(f"\nw = {w}")
#     policy_readable = [actions[a] for a in policy_by_w[w]]
#     print(policy_readable)

# # --- Bisection Whittle indices ---
# whittle_indices = {}
# for s_idx in range(nS):
#     whittle_indices[s_idx] = whittle_index_bisection_continuous(s_idx, w_low=-10, w_high=10)

# # --- Extract Whittle indices from package ---
# P0_pkg, P1_pkg, R0_pkg, R1_pkg = extract_P0_P1_R0_R1_package()
# model_pkg = bandit.restless_bandit_from_P0P1_R0R1(P0_pkg, P1_pkg, R0_pkg, R1_pkg)
# pkg_indices = model_pkg.whittle_indices()
# pkg_indices_dict = {i: pkg_indices[i] for i in range(len(pkg_indices))}

# # --- Comparison ---
# print("\nWhittle index comparison:")
# print(f"{'State':>10} | {'Bisection':>10} | {'Package':>10}")
# print("-"*35)
# for s_idx in range(nS):
#     print(f"{str(states[s_idx]):>10} | {whittle_indices[s_idx]:>10.6f} | {pkg_indices_dict[s_idx]:>10.6f}")

import numpy as np
import markovianbandit as bandit

# =====================================================
# 1. Model definition
# =====================================================
max_inventory = 15
inventory_levels = list(range(max_inventory + 1))
a_prev_values = [0, 1]

states = [(i, ap) for i in inventory_levels for ap in a_prev_values]
state_index = {s: i for i, s in enumerate(states)}
nS = len(states)

ref_state = (0, 1)
ref_idx = state_index[ref_state]

# Cost & rate parameters
h = 1.0
p = 2.0
l = 5.0
gamma = 0.1
lam = 0.6
mu = 0.8
switch_cost = 0.0   # >0 removes multichain issues

# =====================================================
# 2. Generator matrices and costs
# =====================================================
def build_generators_and_costs():
    Q0 = np.zeros((nS, nS))
    Q1 = np.zeros((nS, nS))
    C0 = np.zeros(nS)
    C1 = np.zeros(nS)

    for si, (i, ap) in enumerate(states):
        base_cost = h * i + p * gamma * i
        if i == 0:
            base_cost += l * lam

        C0[si] = base_cost + (switch_cost if ap == 1 else 0.0)
        C1[si] = base_cost + (switch_cost if ap == 0 else 0.0)

        # Passive dynamics
        down = lam + gamma * i
        if i > 0:
            sj = state_index[(i - 1, 0)]
            Q0[si, sj] += down
            Q0[si, si] -= down

        # Active dynamics
        if i < max_inventory:
            sj = state_index[(i + 1, 1)]
            Q1[si, sj] += mu
            Q1[si, si] -= mu

        if i > 0:
            sj = state_index[(i - 1, 1)]
            Q1[si, sj] += down
            Q1[si, si] -= down

    return Q0, Q1, C0, C1

# =====================================================
# 3. Stable continuous-time RVI (subsidy inside)
# =====================================================
def rvi_continuous(Q0, Q1, C0, C1, subsidy,
                   eps=0.01, tol=1e-10, max_iter=50000):
    V = np.zeros(nS)

    for _ in range(max_iter):
        V_old = V.copy()
        C0_sub = C0 - subsidy

        QV0 = C0_sub + Q0 @ V_old
        QV1 = C1 + Q1 @ V_old

        V_tilde = np.minimum(QV0, QV1)
        rho = V_tilde[ref_idx]
        V += eps * (V_tilde - rho - V)

        if np.max(np.abs(V - V_old)) < tol:
            break

    policy = np.argmin(np.vstack([QV0, QV1]).T, axis=1)
    return V, policy

# =====================================================
# 4. Grid-based Whittle indices (definition correct)
# =====================================================
def whittle_indices_grid(Q0, Q1, C0, C1, w_grid):
    Pb = {}
    for w in w_grid:
        _, policy = rvi_continuous(Q0, Q1, C0, C1, w)
        Pb[w] = set(np.where(policy == 0)[0])

    W = {}
    for s in range(nS):
        for w in w_grid:
            if s in Pb[w]:
                W[s] = w
                break
        else:
            W[s] = np.inf

    return W, Pb

# =====================================================
# 5. Policy-based bisection (FIXED)
# =====================================================
def whittle_index_bisection_policy(
    s_idx, w_low=-5.0, w_high=5.0, tol=1e-6
):
    def is_passive(w):
        _, policy = rvi_continuous(Q0, Q1, C0, C1, w)
        return policy[s_idx] == 0

    if is_passive(w_low):
        return w_low
    if not is_passive(w_high):
        return np.inf

    for _ in range(60):
        w_mid = 0.5 * (w_low + w_high)
        if is_passive(w_mid):
            w_high = w_mid
        else:
            w_low = w_mid
        if abs(w_high - w_low) < tol:
            break

    return 0.5 * (w_low + w_high)

# =====================================================
# 6. Indexability (monotonicity) check
# =====================================================
def check_indexability(Pb, w_grid):
    violations = []
    for i in range(len(w_grid) - 1):
        w1, w2 = w_grid[i], w_grid[i + 1]
        if not Pb[w2].issuperset(Pb[w1]):
            violations.append((w1, w2, Pb[w1] - Pb[w2]))
    return violations


# =====================================================
# 7. Package-based Whittle indices (NO tau)
# =====================================================
def whittle_indices_package(Q0, Q1, C0, C1):
    """
    Compute Whittle indices using the markovianbandit package.
    This uses discrete-time uniformization internally.
    """

    # Uniformize continuous-time generators
    P0 = np.eye(nS) + Q0
    P1 = np.eye(nS) + Q1

    R0 = C0
    R1 = C1

    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    W = model.whittle_indices()

    return {i: W[i] for i in range(nS)}

# =====================================================
# 8. Run everything
# =====================================================
if __name__ == "__main__":
    Q0, Q1, C0, C1 = build_generators_and_costs()

    w_grid = np.linspace(-5, 5, 101)

    print("\nComputing grid-based Whittle indices...")
    W_grid, Pb = whittle_indices_grid(Q0, Q1, C0, C1, w_grid)

    print("Checking indexability...")
    violations = check_indexability(Pb, w_grid)
    print("Indexable?", len(violations) == 0)

    print("\nComputing bisection Whittle indices...")
    W_bisect = {s: whittle_index_bisection_policy(s) for s in range(nS)}

    print("\nComputing package Whittle indices...")
    W_pkg = whittle_indices_package(Q0, Q1, C0, C1)

    print("\nState | Grid | Bisection | Package")
    print("-" * 55)
    for s in range(nS):
        print(f"{states[s]} | "
              f"{W_grid[s]:8.3f} | "
              f"{W_bisect[s]:8.3f} | "
              f"{W_pkg[s]:8.3f}")

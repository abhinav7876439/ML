import numpy as np
import pandas as pd
from datetime import datetime

#from Multiple_instances import closed_form_indices

# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")

# =====================================================================
# Extended MDP builder – IMPERFECT REPAIR (two-component state (k,j))
# =====================================================================
def build_extended_mdp_imperfect(N, p, q, c, K, c_switch):
    """
    State space: active family (k,1), k=0,...,N  and passive family (k,0), k=0,...,N.
    Indexing convention:
        idx = k            for state (k,1),  k = 0,...,N   (first N+1 slots)
        idx = (N+1) + k     for state (k,0),  k = 0,...,N   (next N+1 slots)
    """
    num_active = N + 1
    num_states = 2 * (N + 1)

    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    def act_idx(k):
        return k
    def pas_idx(k):
        return num_active + k

    state_labels = [f"({k},1)" for k in range(N + 1)] + [f"({k},0)" for k in range(N + 1)]

    # sanity check on repair-quality distributions
    for k in range(N + 1):
        qk = np.asarray(q[k], dtype=float)
        assert qk.shape[0] == k + 1, f"q[{k}] must have length {k+1}"
        assert np.isclose(qk.sum(), 1.0), f"q[{k}] must sum to 1 (got {qk.sum()})"

    # ---------------- ACTIVE ACTION (a = 1) — repair ----------
    for k in range(N + 1):
        qk = np.asarray(q[k], dtype=float)

        # from active family (k,1): repairman already present, cost c
        s = act_idx(k)
        for i in range(k + 1):
            P1[s, act_idx(i)] += qk[i]
        R1[s] = -c

        # from passive family (k,0): repairman newly assigned, cost c + c_switch
        s = pas_idx(k)
        for i in range(k + 1):
            P1[s, act_idx(i)] += qk[i]
        R1[s] = -(c + c_switch)

    # ---------------- PASSIVE ACTION (a = 0) — survive/breakdown
    for k in range(N + 1):
        for s in (act_idx(k), pas_idx(k)):
            if k < N:
                P0[s, pas_idx(k + 1)] = p[k]          # survive -> (k+1,0), cost 0
                P0[s, pas_idx(0)] = 1 - p[k]           # breakdown -> (0,0), cost K
                R0[s] = -(K * (1 - p[k]))
            else:
                # truncation boundary: forced breakdown
                P0[s, pas_idx(0)] = 1.0
                R0[s] = -K

    return P0, P1, R0, R1, state_labels

# =====================================================================
# Repair-quality distribution q_k(.)
# =====================================================================
def make_geometric_repair_dist(N, r=0.5):
    q = []
    for k in range(N + 1):
        if k == 0:
            q.append(np.array([1.0]))
            continue
        raw = np.array([r ** i for i in range(k)])
        raw = raw / raw.sum() * 0.8
        qk = np.append(raw, 1 - raw.sum())
        q.append(qk)
    return q

# =====================================================================
# 3. Closed-form indices (example implementation – REPLACE with your own
#    if you have a different derivation)
# =====================================================================
# def closed_form_indices(p, beta, K, C, C_switch):
#     """
#     Compute closed-form Whittle indices for the passive states (k,0).
#     p: survival probabilities, length N (p[0]..p[N-1]).
#     Returns (w01, w_pass) where w01 is the index for state (0,1)/(0,0)
#     and w_pass is an array of indices for states (0,0), (1,0), ..., (N,0).
#     """
#     N = len(p)                # p includes 0..N-1, terminal state N handled separately
#     # compute expected discounted cost if we never repair
#     # standard formula for a machine that breaks down with prob 1-p[k] each step
#     # This is a placeholder; in your sample the closed-form matches VI exactly,
#     # so you must use the actual derivation for your model.
#     # Here we just return NaN to indicate not implemented.
#     return np.nan, np.full(N+1, np.nan)  # N+1 passive states

# =====================================================================
# Discounted solvers (VI, PI)
# =====================================================================
def solve_vi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=10000):
    S = len(R0)
    V = np.zeros(S)
    for _ in range(max_iter):
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        V_new = np.maximum(Q0, Q1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    Q0 = R0 + beta * (P0 @ V)
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1

def policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta):
    S = len(pi)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])
    I = np.eye(S)
    V = np.linalg.solve(I - beta * P_pi, r_pi)
    return V

def solve_pi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=1000):
    S = len(R0)
    pi = np.zeros(S, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta)
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + beta * (P0 @ V)
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1

def whittle_index_discounted(state, solver, beta, P0, P1, R0, R1,
                             w_min=-1000.0, w_max=1000.0,
                             tol_w=1e-5, tol_mdp=1e-8, verbose=False):
    solve_f = solve_vi_discounted if solver == 'vi' else solve_pi_discounted
    def f(w):
        _, Q0, Q1 = solve_f(P0, P1, R0, R1, w, beta, tol_mdp)
        return Q0[state] - Q1[state]
    f_min = f(w_min)
    f_max = f(w_max)
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
            w_min = w_mid
            f_min = f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0

# =====================================================================
# Average-reward solvers (RVI, PI)
# =====================================================================
def solve_rvi(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=10000):
    n = len(R0)
    V = np.zeros(n)
    for _ in range(max_iter):
        Q0 = R0 + P0 @ V
        Q1 = R1 - w + P1 @ V
        V_tilde = np.maximum(Q0, Q1)
        delta = V_tilde[ref]
        V_new = V_tilde - delta
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q0 = R0 + P0 @ V
    Q1 = R1 - w + P1 @ V
    g = delta
    return V, g, Q0, Q1

def policy_evaluation_avg(pi, R0, R1, P0, P1, w, ref=0):
    n = len(R0)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(n)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(n)])
    A = np.zeros((n+1, n+1))
    A[:n, :n] = np.eye(n) - P_pi
    A[:n, -1] = 1.0
    A[n, ref] = 1.0
    b = np.zeros(n+1)
    b[:n] = r_pi
    sol = np.linalg.solve(A, b)
    h = sol[:n]
    g = sol[-1]
    return h, g

def solve_pi_avg(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=100):
    n = len(R0)
    pi = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        h, g = policy_evaluation_avg(pi, R0, R1, P0, P1, w, ref)
        Q0 = R0 + P0 @ h
        Q1 = R1 - w + P1 @ h
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + P0 @ h
    Q1 = R1 - w + P1 @ h
    return h, g, Q0, Q1

def whittle_index_bisection_avg(state, solver, R0, R1, P0, P1,
                                w_min=-2000.0, w_max=2000.0, tol_w=1e-10,
                                tol_mdp=1e-12, ref=0, verbose=False):
    solve_func = solve_rvi if solver == 'rvi' else solve_pi_avg
    def f(w):
        _, _, Q0, Q1 = solve_func(R0, R1, P0, P1, w, ref=ref, tol=tol_mdp)
        return Q0[state] - Q1[state]
    f_min = f(w_min)
    f_max = f(w_max)
    if f_min * f_max > 0:
        expand = 2.0
        while f_min * f_max > 0:
            if abs(f_min) < abs(f_max):
                w_min -= expand
                f_min = f(w_min)
            else:
                w_max += expand
                f_max = f(w_max)
            if abs(w_min) > 1e4 or abs(w_max) > 1e4:
                raise RuntimeError(f"Could not bracket index for state {state}")
    for _ in range(60):
        w_mid = (w_min + w_max) / 2.0
        f_mid = f(w_mid)
        if abs(f_mid) < tol_w:
            return w_mid
        if f_mid * f_min > 0:
            w_min = w_mid
            f_min = f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0

# =====================================================================
# Verification & helper routines
# =====================================================================
def check_monotonicity(indices, state_labels, expected_order):
    for i in range(1, len(expected_order)):
        s_prev = expected_order[i-1]
        s_curr = expected_order[i]
        if indices[s_curr] <= indices[s_prev] + 1e-8:
            msg = (f"Monotonicity broken: {state_labels[s_prev]} "
                   f"({indices[s_prev]:.4f}) >= {state_labels[s_curr]} "
                   f"({indices[s_curr]:.4f})")
            return False, msg
    return True, "Indices are strictly increasing in the expected state ordering."

def verify_discounted_threshold(indices, P0, P1, R0, R1, beta, state_labels,
                                tol=1e-5, w_eps=1e-6):
    S = len(indices)
    unique_indices = sorted(set(indices))
    test_Ws = []
    for lam in unique_indices:
        test_Ws.append(lam - w_eps)
        test_Ws.append(lam)
        test_Ws.append(lam + w_eps)
    if unique_indices:
        test_Ws.append(unique_indices[0] - 10.0)
    test_Ws = sorted(set(test_Ws))

    failures = []
    for W in test_Ws:
        pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, W, beta)
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - W) + beta * (P1 @ V)
        for s in range(S):
            if pi[s] == 0:
                if Q1[s] > Q0[s] + tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: passive but Q1 > Q0")
                if abs(V[s] - Q0[s]) > tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: V != Q0")
            else:
                if Q0[s] > Q1[s] + tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: active but Q0 > Q1")
                if abs(V[s] - Q1[s]) > tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: V != Q1")
    if not failures:
        return True, f"Discounted verification passed for {len(test_Ws)} test W values."
    else:
        return False, f"Discounted verification FAILED: first errors:\n" + "\n".join(failures[:10])

def verify_average_threshold(indices, P0, P1, R0, R1, state_labels, tol=1e-5, w_eps=1e-6):
    S = len(indices)
    unique_indices = sorted(set(indices))
    test_Ws = []
    for lam in unique_indices:
        test_Ws.append(lam - w_eps)
        test_Ws.append(lam)
        test_Ws.append(lam + w_eps)
    if unique_indices:
        test_Ws.append(unique_indices[0] - 10.0)
    test_Ws = sorted(set(test_Ws))

    failures = []
    for W in test_Ws:
        pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
        h, g = policy_evaluation_avg(pi, R0, R1, P0, P1, W, ref=0)
        Q0 = R0 + P0 @ h
        Q1 = R1 - W + P1 @ h
        for s in range(S):
            if pi[s] == 0:
                if Q1[s] > Q0[s] + tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: passive but Q1 > Q0")
                if abs(h[s] + g - Q0[s]) > tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: h+g != Q0")
            else:
                if Q0[s] > Q1[s] + tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: active but Q0 > Q1")
                if abs(h[s] + g - Q1[s]) > tol:
                    failures.append(f"W={W:.6f}, {state_labels[s]}: h+g != Q1")
    if not failures:
        return True, f"Average-cost verification passed for {len(test_Ws)} test W values."
    else:
        return False, f"Average-cost verification FAILED: first errors:\n" + "\n".join(failures[:10])

# =====================================================================
# Main experiment loop
# =====================================================================
def run_experiments():
    # Parameter ranges (adjust as needed)
    C_vals = [5.0, 10.0, 20.0, 400.0, 600.0, 1200.0]
    K_vals = [500.0, 1000.0, 2000.0]
    C_switch_vals = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 75.0, 100.0,
                     200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0]
    beta_vals = [0.9, 0.95, 0.9999]

    # Transition matrices
    N = 5
    num_random_matrices = 3
    np.random.seed(42)
    trans_matrices = []
    for _ in range(num_random_matrices):
        p = np.sort(np.random.uniform(0.0, 1.0, N))[::-1]
        p_full = np.append(p, 0.0)
        trans_matrices.append(p_full)

    # Repair distribution for all runs
    q = make_geometric_repair_dist(N, r=0.5)

    total_combos = (len(trans_matrices) * len(C_vals) * len(K_vals) *
                    len(C_switch_vals) * len(beta_vals))
    instance_counter = 0

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_filename = f"Whittle_Experiment_Results_{now_str}.txt"

    with open(out_filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("WHITTLE INDICES: CLOSED-FORM vs VI/PI vs PACKAGE\n")
        f.write("WITH MONOTONICITY AND INDEXABILITY VERIFICATION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameter ranges:\n")
        f.write(f"  C (activation cost)        : {C_vals}\n")
        f.write(f"  K (penalty cost)           : {K_vals}\n")
        f.write(f"  C_switch (switching cost)  : {C_switch_vals}\n")
        f.write(f"  beta (discount factor)     : {beta_vals}\n")
        f.write(f"  Number of transition matrices : {len(trans_matrices)}\n")
        f.write(f"  Total combinations            : {total_combos}\n")
        f.write("\n" + "#" * 80 + "\n\n")

        for mat_idx, p_full in enumerate(trans_matrices, start=1):
            N_current = len(p_full) - 1
            p_str = np.array2string(p_full, precision=4, separator=', ')
            f.write(f"TRANSITION MATRIX {mat_idx}\n")
            f.write(f"Survival probabilities: {p_str}\n")
            f.write("#" * 80 + "\n\n")

            for C in C_vals:
                for K in K_vals:
                    for C_switch in C_switch_vals:
                        for beta in beta_vals:
                            instance_counter += 1
                            print(f"[{instance_counter}/{total_combos}] "
                                  f"C={C}, K={K}, Csw={C_switch}, beta={beta}, mat={mat_idx}",
                                  flush=True)

                            # Build MDP
                            P0, P1, R0, R1, state_labels = build_extended_mdp_imperfect(
                                N_current, p_full[:-1], q, C, K, C_switch)
                            expected_order = [1, 0] + list(range(2, len(state_labels)))
                            p_raw = p_full[:-1]

                            # ========== Instance header ==========
                            f.write("=" * 80 + "\n")
                            f.write(f"INSTANCE {instance_counter}\n")
                            f.write("=" * 80 + "\n")
                            f.write(f"C={C}, K={K}, C_switch={C_switch}, beta={beta}\n")
                            f.write(f"Survival probabilities: {p_str}\n")
                            f.write("-" * 80 + "\n\n")

                            f.write("Passive transition matrix:\n")
                            f.write(pd.DataFrame(P0, index=state_labels, columns=state_labels)
                                    .to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
                            f.write("Passive reward vector:\n")
                            f.write(pd.DataFrame(R0, index=state_labels, columns=["Passive"])
                                    .to_string(float_format=lambda x: f"{x:.4f}") + "\n\n")
                            f.write("Active transition matrix:\n")
                            f.write(pd.DataFrame(P1, index=state_labels, columns=state_labels)
                                    .to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
                            f.write("Active reward vector:\n")
                            f.write(pd.DataFrame(R1, index=state_labels, columns=["Active"])
                                    .to_string(float_format=lambda x: f"{x:.4f}") + "\n\n")

                            # ==================== DISCOUNTED ====================
                            f.write(f"--- Discounted Whittle Indices (beta={beta}) ---\n")
                            disc_results = {}
                            for method in ['vi', 'pi']:
                                indices = []
                                for s in range(len(R0)):
                                    try:
                                        lam = whittle_index_discounted(
                                            s, method, beta, P0, P1, R0, R1,
                                            w_min=-2000.0, w_max=2000.0,
                                            tol_w=1e-5, tol_mdp=1e-8)
                                        indices.append(lam)
                                    except:
                                        indices.append(np.nan)
                                disc_results[method] = indices

                            # # Closed-form
                            # try:
                            #     w01_cf, w_pass_cf = closed_form_indices(p_raw, beta, K, C, C_switch)
                            #     closed = [w01_cf] + list(w_pass_cf)
                            # except:
                            #     closed = [np.nan] * len(state_labels)

                            # Package
                            if PKG_AVAILABLE:
                                try:
                                    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                                    pkg_disc = model.whittle_indices(discount=beta)
                                except:
                                    pkg_disc = [np.nan] * len(state_labels)
                            else:
                                pkg_disc = [np.nan] * len(state_labels)

                            df_disc = pd.DataFrame({
                                #'ClosedForm': closed,
                                'VI': disc_results['vi'],
                                'PI': disc_results['pi'],
                                'Package': pkg_disc
                            }, index=state_labels)
                            f.write(df_disc.to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")

                            # Monotonicity
                            mono_ok, mono_msg = check_monotonicity(disc_results['vi'], state_labels, expected_order)
                            f.write("Discounted monotonicity (closed-form): " + ("PASS" if mono_ok else "FAIL") + "\n")
                            f.write(mono_msg + "\n\n")

                            # Verification
                            disc_ok, disc_msg = verify_discounted_threshold(disc_results['vi'], P0, P1, R0, R1, beta, state_labels)
                            f.write("Discounted indexability verification: " + ("PASS" if disc_ok else "FAIL") + "\n")
                            f.write(disc_msg + "\n\n")

                            # ==================== AVERAGE COST ====================
                            f.write("--- Average-Cost Whittle Indices ---\n")
                            avg_results = {}
                            for method in ['rvi', 'pi']:
                                indices = []
                                for s in range(len(R0)):
                                    try:
                                        lam = whittle_index_bisection_avg(
                                            s, method, R0, R1, P0, P1,
                                            w_min=-2000.0, w_max=2000.0,
                                            tol_w=1e-8, tol_mdp=1e-12)
                                        indices.append(lam)
                                    except:
                                        indices.append(np.nan)
                                avg_results[method] = indices

                            if PKG_AVAILABLE:
                                try:
                                    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                                    pkg_avg = model.whittle_indices()
                                except:
                                    pkg_avg = [np.nan] * len(state_labels)
                            else:
                                pkg_avg = [np.nan] * len(state_labels)

                            df_avg = pd.DataFrame({
                                'RVI': avg_results['rvi'],
                                'PI': avg_results['pi'],
                                'Package': pkg_avg
                            }, index=state_labels)
                            f.write(df_avg.to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")

                            # Monotonicity (using RVI indices)
                            mono_avg_ok, mono_avg_msg = check_monotonicity(avg_results['rvi'], state_labels, expected_order)
                            f.write("Average-cost monotonicity (RVI): " + ("PASS" if mono_avg_ok else "FAIL") + "\n")
                            f.write(mono_avg_msg + "\n\n")

                            # Verification
                            avg_ok, avg_msg = verify_average_threshold(avg_results['rvi'], P0, P1, R0, R1, state_labels)
                            f.write("Average-cost indexability verification: " + ("PASS" if avg_ok else "FAIL") + "\n")
                            f.write(avg_msg + "\n\n")

    print(f"\nAll experiments completed. Results saved to '{out_filename}'.")

if __name__ == "__main__":
    run_experiments()
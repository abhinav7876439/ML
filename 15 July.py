# import numpy as np
# import pandas as pd
# from datetime import datetime

# # Attempt package import
# try:
#     import markovianbandit as bandit
#     PKG_AVAILABLE = True
# except ImportError:
#     PKG_AVAILABLE = False
#     print("markovianbandit not installed; package column will be NaN.")

# # =====================================================================
# # 1. Extended MDP builder (same as before)
# # =====================================================================
# def build_extended_mdp_imperfect(N, p, q, c, K, c_switch):
#     num_active = N + 1
#     num_states = 2 * (N + 1)
#     P0 = np.zeros((num_states, num_states))
#     P1 = np.zeros((num_states, num_states))
#     R0 = np.zeros(num_states)
#     R1 = np.zeros(num_states)

#     def act_idx(k): return k
#     def pas_idx(k): return num_active + k

#     state_labels = [f"({k},1)" for k in range(N + 1)] + [f"({k},0)" for k in range(N + 1)]

#     for k in range(N + 1):
#         qk = np.asarray(q[k], dtype=float)
#         s_active = act_idx(k)
#         for i in range(k + 1):
#             P1[s_active, act_idx(i)] += qk[i]
#         R1[s_active] = -c

#         s_passive = pas_idx(k)
#         for i in range(k + 1):
#             P1[s_passive, act_idx(i)] += qk[i]
#         R1[s_passive] = -(c + c_switch)

#     for k in range(N + 1):
#         for s in (act_idx(k), pas_idx(k)):
#             if k < N:
#                 P0[s, pas_idx(k + 1)] = p[k]
#                 P0[s, pas_idx(0)] = 1 - p[k]
#                 R0[s] = -(K * (1 - p[k]))
#             else:
#                 P0[s, pas_idx(0)] = 1.0
#                 R0[s] = -K
#     return P0, P1, R0, R1, state_labels

# def make_geometric_repair_dist(N, r=0.5):
#     q = []
#     for k in range(N + 1):
#         if k == 0:
#             q.append(np.array([1.0]))
#             continue
#         raw = np.array([r ** i for i in range(k)])
#         raw = raw / raw.sum() * 0.8
#         qk = np.append(raw, 1 - raw.sum())
#         q.append(qk)
#     return q

# # Placeholder closed-form function (replace with your own)
# def closed_form_indices(p, beta, K, C, C_switch):
#     # Replace with actual closed-form computation.
#     # Returning NaN for illustration; your implementation should fill these.
#     return np.nan, np.full(len(p)+1, np.nan)   # (w01, w_passive[0..N])

# # =====================================================================
# # 2. Discounted and average solvers (unchanged from your code)
# # =====================================================================
# def solve_vi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=10000):
#     S = len(R0)
#     V = np.zeros(S)
#     for _ in range(max_iter):
#         Q0 = R0 + beta * (P0 @ V)
#         Q1 = (R1 - w) + beta * (P1 @ V)
#         V_new = np.maximum(Q0, Q1)
#         if np.max(np.abs(V_new - V)) < tol:
#             break
#         V = V_new
#     Q0 = R0 + beta * (P0 @ V)
#     Q1 = (R1 - w) + beta * (P1 @ V)
#     return V, Q0, Q1

# def policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta):
#     S = len(pi)
#     P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
#     r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])
#     V = np.linalg.solve(np.eye(S) - beta * P_pi, r_pi)
#     return V

# def solve_pi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=1000):
#     S = len(R0)
#     pi = np.zeros(S, dtype=int)
#     for _ in range(max_iter):
#         V = policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta)
#         Q0 = R0 + beta * (P0 @ V)
#         Q1 = (R1 - w) + beta * (P1 @ V)
#         pi_new = (Q1 > Q0).astype(int)
#         if np.all(pi_new == pi):
#             break
#         pi = pi_new
#     return V, Q0, Q1

# def whittle_index_discounted(state, solver, beta, P0, P1, R0, R1,
#                              w_min=-1000.0, w_max=1000.0, tol_w=1e-5, tol_mdp=1e-8):
#     solve_f = solve_vi_discounted if solver == 'vi' else solve_pi_discounted
#     def f(w):
#         _, Q0, Q1 = solve_f(P0, P1, R0, R1, w, beta, tol_mdp)
#         return Q0[state] - Q1[state]
#     f_min, f_max = f(w_min), f(w_max)
#     expand = 1.5
#     while f_min * f_max > 0:
#         if abs(f_min) < abs(f_max):
#             w_min -= expand * abs(w_min - w_max) if abs(w_min - w_max) > 0 else 10.0
#             f_min = f(w_min)
#         else:
#             w_max += expand * abs(w_max - w_min)
#             f_max = f(w_max)
#         if abs(w_min) > 1e6 or abs(w_max) > 1e6:
#             raise ValueError("Cannot bracket root")
#     for _ in range(160):
#         w_mid = (w_min + w_max) / 2.0
#         f_mid = f(w_mid)
#         if abs(f_mid) < tol_w:
#             return w_mid
#         if f_mid * f_min > 0:
#             w_min = w_mid
#             f_min = f_mid
#         else:
#             w_max = w_mid
#     return (w_min + w_max) / 2.0

# # Average-cost solvers
# def solve_rvi(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=10000):
#     n = len(R0)
#     V = np.zeros(n)
#     for _ in range(max_iter):
#         Q0 = R0 + P0 @ V
#         Q1 = R1 - w + P1 @ V
#         V_tilde = np.maximum(Q0, Q1)
#         delta = V_tilde[ref]
#         V_new = V_tilde - delta
#         if np.max(np.abs(V_new - V)) < tol:
#             V = V_new
#             break
#         V = V_new
#     return V, delta, Q0, Q1

# def policy_evaluation_avg(pi, R0, R1, P0, P1, w, ref=0):
#     n = len(R0)
#     P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(n)])
#     r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(n)])
#     A = np.zeros((n+1, n+1))
#     A[:n, :n] = np.eye(n) - P_pi
#     A[:n, -1] = 1.0
#     A[n, ref] = 1.0
#     b = np.zeros(n+1)
#     b[:n] = r_pi
#     sol = np.linalg.solve(A, b)
#     return sol[:n], sol[-1]

# def solve_pi_avg(R0, R1, P0, P1, w, ref=0, tol=1e-12, max_iter=100):
#     n = len(R0)
#     pi = np.zeros(n, dtype=int)
#     for _ in range(max_iter):
#         h, g = policy_evaluation_avg(pi, R0, R1, P0, P1, w, ref)
#         Q0 = R0 + P0 @ h
#         Q1 = R1 - w + P1 @ h
#         pi_new = (Q1 > Q0).astype(int)
#         if np.all(pi_new == pi):
#             break
#         pi = pi_new
#     return h, g, Q0, Q1

# def whittle_index_bisection_avg(state, solver, R0, R1, P0, P1,
#                                 w_min=-2000.0, w_max=2000.0, tol_w=1e-10,
#                                 tol_mdp=1e-12, ref=0):
#     solve_func = solve_rvi if solver == 'rvi' else solve_pi_avg
#     def f(w):
#         _, _, Q0, Q1 = solve_func(R0, R1, P0, P1, w, ref=ref, tol=tol_mdp)
#         return Q0[state] - Q1[state]
#     f_min, f_max = f(w_min), f(w_max)
#     if f_min * f_max > 0:
#         expand = 2.0
#         while f_min * f_max > 0:
#             if abs(f_min) < abs(f_max):
#                 w_min -= expand
#                 f_min = f(w_min)
#             else:
#                 w_max += expand
#                 f_max = f(w_max)
#             if abs(w_min) > 1e4 or abs(w_max) > 1e4:
#                 raise RuntimeError(f"Could not bracket index for state {state}")
#     for _ in range(60):
#         w_mid = (w_min + w_max) / 2.0
#         f_mid = f(w_mid)
#         if abs(f_mid) < tol_w:
#             return w_mid
#         if f_mid * f_min > 0:
#             w_min = w_mid
#             f_min = f_mid
#         else:
#             w_max = w_mid
#     return (w_min + w_max) / 2.0

# # =====================================================================
# # 3. Verification helpers (discounted / average)
# # =====================================================================
# def check_monotonicity(indices, state_labels, expected_order):
#     for i in range(1, len(expected_order)):
#         s_prev = expected_order[i-1]
#         s_curr = expected_order[i]
#         if indices[s_curr] <= indices[s_prev] + 1e-8:
#             msg = (f"Monotonicity broken: {state_labels[s_prev]} "
#                    f"({indices[s_prev]:.4f}) >= {state_labels[s_curr]} "
#                    f"({indices[s_curr]:.4f})")
#             return False, msg
#     return True, "Indices are strictly increasing in the expected state ordering."

# def verify_discounted_threshold(indices, P0, P1, R0, R1, beta, state_labels,
#                                 tol=1e-5, w_eps=1e-6):
#     S = len(indices)
#     unique = sorted(set(indices))
#     test_Ws = set()
#     for lam in unique:
#         test_Ws.update([lam - w_eps, lam, lam + w_eps])
#     if unique:
#         test_Ws.add(unique[0] - 10.0)
#     test_Ws = sorted(test_Ws)

#     failures = []
#     for W in test_Ws:
#         pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
#         V = policy_evaluation_discounted(pi, P0, P1, R0, R1, W, beta)
#         Q0 = R0 + beta * (P0 @ V)
#         Q1 = (R1 - W) + beta * (P1 @ V)
#         for s in range(S):
#             if pi[s] == 0 and Q1[s] > Q0[s] + tol:
#                 failures.append(f"W={W:.6f}, {state_labels[s]}: passive but Q1 > Q0")
#             if pi[s] == 1 and Q0[s] > Q1[s] + tol:
#                 failures.append(f"W={W:.6f}, {state_labels[s]}: active but Q0 > Q1")
#     if not failures:
#         return True, f"Discounted verification passed for {len(test_Ws)} test W values."
#     else:
#         return False, f"Discounted verification FAILED: first errors:\n" + "\n".join(failures[:10])

# def verify_average_threshold(indices, P0, P1, R0, R1, state_labels, tol=1e-5, w_eps=1e-6):
#     S = len(indices)
#     unique = sorted(set(indices))
#     test_Ws = set()
#     for lam in unique:
#         test_Ws.update([lam - w_eps, lam, lam + w_eps])
#     if unique:
#         test_Ws.add(unique[0] - 10.0)
#     test_Ws = sorted(test_Ws)

#     failures = []
#     for W in test_Ws:
#         pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
#         h, g = policy_evaluation_avg(pi, R0, R1, P0, P1, W, ref=0)
#         Q0 = R0 + P0 @ h
#         Q1 = R1 - W + P1 @ h
#         for s in range(S):
#             if pi[s] == 0 and Q1[s] > Q0[s] + tol:
#                 failures.append(f"W={W:.6f}, {state_labels[s]}: passive but Q1 > Q0")
#             if pi[s] == 1 and Q0[s] > Q1[s] + tol:
#                 failures.append(f"W={W:.6f}, {state_labels[s]}: active but Q0 > Q1")
#     if not failures:
#         return True, f"Average-cost verification passed for {len(test_Ws)} test W values."
#     else:
#         return False, f"Average-cost verification FAILED: first errors:\n" + "\n".join(failures[:10])

# # =====================================================================
# # 4. LaTeX formatting functions
# # =====================================================================
# def escape_latex(s):
#     """Escape special LaTeX characters."""
#     return (s.replace('\\', '\\textbackslash ')
#             .replace('&', '\\&')
#             .replace('%', '\\%')
#             .replace('$', '\\$')
#             .replace('#', '\\#')
#             .replace('_', '\\_')
#             .replace('{', '\\{')
#             .replace('}', '\\}')
#             .replace('~', '\\textasciitilde ')
#             .replace('^', '\\textasciicircum '))

# def matrix_to_latex_array(P, row_labels, col_labels):
#     """Build an array environment for a transition matrix."""
#     rows = []
#     header = " & " + " & ".join(col_labels) + " \\\\"
#     rows.append(header)
#     rows.append("\\hline")
#     for i, rl in enumerate(row_labels):
#         vals = " & ".join(f"{P[i,j]:.6f}" for j in range(len(col_labels)))
#         rows.append(f"{rl} & {vals} \\\\")
#     return "\\begin{array}{c|" + "c"*len(col_labels) + "}\n" + "\n".join(rows) + "\n\\end{array}"

# def reward_vector_to_latex_array(R, row_labels, action_name):
#     """Build a two-column array for a reward vector."""
#     rows = [f"\\text{{State}} & \\text{{{action_name} Reward}} \\\\", "\\hline"]
#     for i, rl in enumerate(row_labels):
#         rows.append(f"{rl} & {R[i]:.6f} \\\\")
#     return "\\begin{array}{c|c}\n" + "\n".join(rows) + "\n\\end{array}"

# def format_index(val, tol=1e-6):
#     """Return a string for an index, or '--' if NaN."""
#     if np.isnan(val):
#         return "--"
#     return f"{val:.6f}"

# def color_if_diff(v1, v2, threshold=1e-4):
#     """If two numbers differ by more than threshold, return red textcolor."""
#     if np.isnan(v1) or np.isnan(v2):
#         return ""
#     if abs(v1 - v2) > threshold:
#         return "\\textcolor{red}"
#     return ""

# def build_index_table(state_labels, methods_dict, reference_col=None, ref_name="Package", caption="", label=""):
#     """
#     methods_dict: {col_name: list_of_values}
#     reference_col: list of values to compare against for colouring (e.g., package).
#     Returns a LaTeX table body (without the outer tabular environment). We'll embed it inside a table float.
#     """
#     cols = list(methods_dict.keys())
#     ncol = len(cols)
#     header = " & ".join(["\\textbf{State}"] + [f"\\textbf{{{c}}}" for c in cols]) + " \\\\"
#     rows = [header, "\\midrule"]
#     for i, s in enumerate(state_labels):
#         vals = []
#         for c in cols:
#             v = methods_dict[c][i]
#             val_str = format_index(v)
#             # Coloring logic: compare to reference column if provided
#             if reference_col is not None and c != ref_name:
#                 ref_val = reference_col[i]
#                 color_cmd = color_if_diff(v, ref_val)
#                 if color_cmd:
#                     val_str = f"{color_cmd}{{{val_str}}}"
#             vals.append(val_str)
#         rows.append(f"{s} & " + " & ".join(vals) + " \\\\")
#     return "\n".join(rows)

# def write_parameter_table(f, C, K, C_switch, beta, N, state_labels):
#     f.write("\\begin{table}[h!]\n")
#     f.write("\\centering\n")
#     f.write("\\caption{Model Parameters for Instance \\theinstance}\n")
#     f.write("\\begin{tabular}{@{}llr@{}}\n")
#     f.write("\\toprule\n")
#     f.write("\\textbf{Parameter} & \\textbf{Description} & \\textbf{Value} \\\\\n")
#     f.write("\\midrule\n")
#     f.write(f"$N$ & Maximum degradation level & {N} \\\\\n")
#     f.write(f"$k$ & Number of states & {len(state_labels)} \\\\\n")
#     f.write(f"$C$ & Active repair cost ($C_m$) & {C} \\\\\n")
#     f.write(f"$c_{{\\text{{switch}}}}$ & Switching cost & {C_switch} \\\\\n")
#     f.write(f"$K$ & Failure cost ($K_m$) & {K} \\\\\n")
#     f.write(f"$\\beta$ & Discount factor & {beta} \\\\\n")
#     f.write("\\bottomrule\n")
#     f.write("\\end{tabular}\n")
#     f.write("\\end{table}\n\n")

# def write_survival_probs(f, p_full):
#     N = len(p_full) - 1
#     probs = ", ".join([f"p_{{{k}}} = {p_full[k]:.4f}" for k in range(N+1)])
#     f.write(f"\\[\n{probs}\n\\]\n\n")

# # =====================================================================
# # 5. Main experiment loop – LaTeX output
# # =====================================================================
# def run_experiments_latex():
#     # Parameter ranges
#     C_vals = [5.0, 10.0, 20.0, 400.0, 600.0, 1200.0]
#     K_vals = [500.0, 1000.0, 2000.0]
#     C_switch_vals = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 75.0, 100.0,
#                      200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0]
#     beta_vals = [0.9, 0.95, 0.9999]

#     N = 5
#     num_random_matrices = 3
#     np.random.seed(42)
#     trans_matrices = []
#     for _ in range(num_random_matrices):
#         p = np.sort(np.random.uniform(0.0, 1.0, N))[::-1]
#         p_full = np.append(p, 0.0)
#         trans_matrices.append(p_full)

#     q = make_geometric_repair_dist(N, r=0.5)

#     total_combos = (len(trans_matrices) * len(C_vals) * len(K_vals) *
#                     len(C_switch_vals) * len(beta_vals))
#     instance_counter = 0

#     now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
#     filename = f"Whittle_Experiment_LaTeX_{now_str}.tex"

#     with open(filename, "w") as f:
#         # LaTeX preamble
#         f.write("\\documentclass{article}\n")
#         f.write("\\usepackage[margin=1in]{geometry}\n")
#         f.write("\\usepackage{booktabs}\n")
#         f.write("\\usepackage{multirow}\n")
#         f.write("\\usepackage{xcolor}\n")
#         f.write("\\usepackage{amsmath}\n")
#         f.write("\\newcounter{instance}\n")
#         f.write("\\setcounter{instance}{0}\n")
#         f.write("\\newcommand{\\theinstance}{\\arabic{instance}}\n")
#         f.write("\\begin{document}\n\n")
#         f.write("\\title{Whittle Indices: Closed-Form vs VI/PI vs Package}\n")
#         f.write("\\author{Automatic Experiment}\n")
#         f.write("\\date{\\today}\n")
#         f.write("\\maketitle\n\n")

#         # Summary of parameter ranges
#         f.write("\\section*{Experiment Setup}\n")
#         f.write("\\begin{itemize}\n")
#         f.write(f"\\item $C$ (activation cost): {C_vals}\n")
#         f.write(f"\\item $K$ (penalty cost): {K_vals}\n")
#         f.write(f"\\item $c_{{\\text{{switch}}}}$ (switching cost): {C_switch_vals}\n")
#         f.write(f"\\item $\\beta$ (discount factor): {beta_vals}\n")
#         f.write(f"\\item Number of transition matrices: {len(trans_matrices)}\n")
#         f.write(f"\\item Total combinations: {total_combos}\n")
#         f.write("\\end{itemize}\n\n")

#         # For optional aggregated comparison: store indices for a specific combination
#         aggregated_data = {}   # key: beta, value: {state_label: {'closed':..., 'package':...}}

#         for mat_idx, p_full in enumerate(trans_matrices, start=1):
#             N_current = len(p_full) - 1
#             f.write(f"\\section{{Transition Matrix {mat_idx}}}\n")
#             f.write(f"Survival probabilities: $p = [{', '.join(f'{p_full[i]:.4f}' for i in range(N_current+1))}]$\n\n")

#             for C in C_vals:
#                 for K in K_vals:
#                     for C_switch in C_switch_vals:
#                         # We'll aggregate only for a specific combination (e.g., C=5.0, K=500.0, C_switch=500.0, mat=1)
#                         do_aggregate = (mat_idx == 1 and C == 5.0 and K == 500.0 and C_switch == 500.0)

#                         for beta in beta_vals:
#                             instance_counter += 1
#                             print(f"[{instance_counter}/{total_combos}] "
#                                   f"C={C}, K={K}, Csw={C_switch}, beta={beta}, mat={mat_idx}",
#                                   flush=True)

#                             # Build MDP
#                             P0, P1, R0, R1, state_labels = build_extended_mdp_imperfect(
#                                 N_current, p_full[:-1], q, C, K, C_switch)

#                             # Increment instance counter for LaTeX
#                             f.write("\\stepcounter{instance}\n")

#                             # ---- Problem parameters ----
#                             write_parameter_table(f, C, K, C_switch, beta, N_current, state_labels)

#                             # ---- Survival probabilities ----
#                             write_survival_probs(f, p_full)

#                             # ---- Passive action matrices ----
#                             f.write("\\subsection*{Transition and Reward for Passive Action ($a=0$)}\n")
#                             f.write("\\subsubsection*{Transition Matrix}\n")
#                             f.write("\\[\n")
#                             f.write(matrix_to_latex_array(P0, state_labels, state_labels))
#                             f.write("\n\\]\n")
#                             f.write("\\subsubsection*{Reward Vector}\n")
#                             f.write("\\[\n")
#                             f.write(reward_vector_to_latex_array(R0, state_labels, "Passive"))
#                             f.write("\n\\]\n\n")

#                             # ---- Active action matrices ----
#                             f.write("\\subsection*{Transition and Reward for Active Action ($a=1$)}\n")
#                             f.write("\\subsubsection*{Transition Matrix}\n")
#                             f.write("\\[\n")
#                             f.write(matrix_to_latex_array(P1, state_labels, state_labels))
#                             f.write("\n\\]\n")
#                             f.write("\\subsubsection*{Reward Vector}\n")
#                             f.write("\\[\n")
#                             f.write(reward_vector_to_latex_array(R1, state_labels, "Active"))
#                             f.write("\n\\]\n\n")

#                             # ---- Discounted indices ----
#                             disc_results = {}
#                             for method in ['vi', 'pi']:
#                                 indices = []
#                                 for s in range(len(R0)):
#                                     try:
#                                         lam = whittle_index_discounted(
#                                             s, method, beta, P0, P1, R0, R1,
#                                             w_min=-2000.0, w_max=2000.0, tol_w=1e-5, tol_mdp=1e-8)
#                                         indices.append(lam)
#                                     except:
#                                         indices.append(np.nan)
#                                 disc_results[method] = indices

#                             # Closed-form
#                             try:
#                                 w01_cf, w_pass_cf = closed_form_indices(p_full[:-1], beta, K, C, C_switch)
#                                 closed_disc = [w01_cf] + list(w_pass_cf)   # length = N+2
#                             except:
#                                 closed_disc = [np.nan] * len(state_labels)

#                             # Package
#                             if PKG_AVAILABLE:
#                                 try:
#                                     model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#                                     pkg_disc = model.whittle_indices(discount=beta)
#                                 except:
#                                     pkg_disc = [np.nan] * len(state_labels)
#                             else:
#                                 pkg_disc = [np.nan] * len(state_labels)

#                             # Build table
#                             methods_disc = {
#                                 'ClosedForm': closed_disc,
#                                 'VI': disc_results['vi'],
#                                 'PI': disc_results['pi'],
#                                 'Package': pkg_disc
#                             }
#                             disc_table_body = build_index_table(
#                                 state_labels, methods_disc, reference_col=pkg_disc, ref_name="Package",
#                                 caption=f"Discounted Whittle indices ($\\beta={beta}$)",
#                                 label=f"tab:disc_{instance_counter}"
#                             )

#                             f.write("\\subsection*{Discounted Whittle Indices}\n")
#                             f.write("\\begin{table}[h!]\n\\centering\n")
#                             f.write(f"\\caption{{Discounted Whittle indices for Instance \\theinstance}}\n")
#                             f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
#                             f.write(disc_table_body)
#                             f.write("\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

#                             # Monotonicity and verification (using closed-form)
#                             expected_order = [1, 0] + list(range(2, len(state_labels)))
#                             mono_ok, mono_msg = check_monotonicity(closed_disc, state_labels, expected_order)
#                             disc_verif_ok, disc_verif_msg = verify_discounted_threshold(
#                                 closed_disc, P0, P1, R0, R1, beta, state_labels)

#                             f.write("\\noindent\\textbf{Discounted monotonicity (closed-form):} "
#                                     f"{'PASS' if mono_ok else 'FAIL'}\n\n")
#                             f.write(f"{escape_latex(mono_msg)}\n\n")
#                             f.write("\\noindent\\textbf{Discounted indexability verification:} "
#                                     f"{'PASS' if disc_verif_ok else 'FAIL'}\n\n")
#                             f.write(f"{escape_latex(disc_verif_msg)}\n\n")

#                             # Store for aggregated comparison
#                             if do_aggregate:
#                                 aggregated_data.setdefault(beta, {})
#                                 for i, s in enumerate(state_labels):
#                                     aggregated_data[beta][s] = {
#                                         'closed': closed_disc[i],
#                                         'package': pkg_disc[i]
#                                     }

#                             # ---- Average-cost indices ----
#                             avg_results = {}
#                             for method in ['rvi', 'pi']:
#                                 indices = []
#                                 for s in range(len(R0)):
#                                     try:
#                                         lam = whittle_index_bisection_avg(
#                                             s, method, R0, R1, P0, P1,
#                                             w_min=-2000.0, w_max=2000.0, tol_w=1e-8, tol_mdp=1e-12)
#                                         indices.append(lam)
#                                     except:
#                                         indices.append(np.nan)
#                                 avg_results[method] = indices

#                             if PKG_AVAILABLE:
#                                 try:
#                                     model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
#                                     pkg_avg = model.whittle_indices()
#                                 except:
#                                     pkg_avg = [np.nan] * len(state_labels)
#                             else:
#                                 pkg_avg = [np.nan] * len(state_labels)

#                             methods_avg = {
#                                 'RVI': avg_results['rvi'],
#                                 'PI': avg_results['pi'],
#                                 'Package': pkg_avg
#                             }
#                             avg_table_body = build_index_table(state_labels, methods_avg,
#                                                                reference_col=pkg_avg, ref_name="Package")

#                             f.write("\\subsection*{Average-Cost Whittle Indices}\n")
#                             f.write("\\begin{table}[h!]\n\\centering\n")
#                             f.write(f"\\caption{{Average-cost Whittle indices for Instance \\theinstance}}\n")
#                             f.write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\n")
#                             f.write(avg_table_body)
#                             f.write("\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

#                             avg_verif_ok, avg_verif_msg = verify_average_threshold(
#                                 avg_results['rvi'], P0, P1, R0, R1, state_labels)

#                             # Average monotonicity using RVI
#                             mono_avg_ok, mono_avg_msg = check_monotonicity(avg_results['rvi'], state_labels, expected_order)
#                             f.write("\\noindent\\textbf{Average-cost monotonicity (RVI):} "
#                                     f"{'PASS' if mono_avg_ok else 'FAIL'}\n\n")
#                             f.write(f"{escape_latex(mono_avg_msg)}\n\n")
#                             f.write("\\noindent\\textbf{Average-cost indexability verification:} "
#                                     f"{'PASS' if avg_verif_ok else 'FAIL'}\n\n")
#                             f.write(f"{escape_latex(avg_verif_msg)}\n\n")

#                             f.write("\\newpage\n")

#         # ---- Aggregated comparison table (only if we collected data) ----
#         if aggregated_data:
#             f.write("\\section*{Whittle Indices Comparison (Switching Cost $C_{\\text{switch}}=500.0$)}\n")
#             f.write("\\begin{table}[h!]\n\\centering\n")
#             f.write("\\caption{Closed-Form vs Package Discounted Indices -- Colour indicates mismatch}\n")
#             f.write("\\begin{tabular}{@{}l")
#             for _ in beta_vals:
#                 f.write("cc")
#             f.write("{@{\\hskip 0pt}}\n")
#             f.write("\\toprule\n")
#             # Header with multirow for state and beta columns
#             f.write("\\multirow{2}{*}{\\textbf{State}} & ")
#             for i, beta in enumerate(beta_vals):
#                 f.write(f"\\multicolumn{{2}}{{c}}{{$\\beta = {beta}$}}")
#                 if i < len(beta_vals)-1:
#                     f.write(" & ")
#             f.write(" \\\\\n")
#             f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n")
#             f.write(" & ")
#             for i, beta in enumerate(beta_vals):
#                 f.write("\\textbf{Closed-Form} & \\textbf{Package}")
#                 if i < len(beta_vals)-1:
#                     f.write(" & ")
#             f.write(" \\\\\n\\midrule\n")

#             # Build rows for each state in a fixed order (e.g., (0,1), (0,0), (1,0), ...)
#             state_order = [f"({k},1)" for k in range(N_current+1)] + [f"({k},0)" for k in range(N_current+1)]
#             # Actually we want only the passive states? The sample shows (0,1), (0,0), (1,0), ..., (5,0).
#             # We'll include all states as they appear.
#             for s in state_order:
#                 row = f"{s} & "
#                 entries = []
#                 for beta in beta_vals:
#                     if beta in aggregated_data and s in aggregated_data[beta]:
#                         vals = aggregated_data[beta][s]
#                         cf = vals['closed']; pkg = vals['package']
#                         # Colour if mismatch
#                         cf_str = format_index(cf)
#                         pkg_str = format_index(pkg)
#                         if not np.isnan(cf) and not np.isnan(pkg) and abs(cf - pkg) > 1e-4:
#                             cf_str = f"\\textcolor{{blue}}{{{cf_str}}}"
#                             pkg_str = f"\\textcolor{{blue}}{{{pkg_str}}}"
#                         entries.append(f"{cf_str} & {pkg_str}")
#                     else:
#                         entries.append("-- & --")
#                 row += " & ".join(entries) + " \\\\"
#                 f.write(row + "\n")
#             f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

#         f.write("\\end{document}\n")

#     print(f"LaTeX file saved as '{filename}'.")

# if __name__ == "__main__":
#     run_experiments_latex()




import numpy as np
import pandas as pd
from datetime import datetime

# Attempt package import
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; package column will be NaN.")

# =====================================================================
# 1. Extended MDP builder (same as before)
# =====================================================================
def build_extended_mdp_imperfect(N, p, q, c, K, c_switch):
    num_active = N + 1
    num_states = 2 * (N + 1)
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    def act_idx(k): return k
    def pas_idx(k): return num_active + k

    state_labels = [f"({k},1)" for k in range(N + 1)] + [f"({k},0)" for k in range(N + 1)]

    for k in range(N + 1):
        qk = np.asarray(q[k], dtype=float)
        s_active = act_idx(k)
        for i in range(k + 1):
            P1[s_active, act_idx(i)] += qk[i]
        R1[s_active] = -c

        s_passive = pas_idx(k)
        for i in range(k + 1):
            P1[s_passive, act_idx(i)] += qk[i]
        R1[s_passive] = -(c + c_switch)

    for k in range(N + 1):
        for s in (act_idx(k), pas_idx(k)):
            if k < N:
                P0[s, pas_idx(k + 1)] = p[k]
                P0[s, pas_idx(0)] = 1 - p[k]
                R0[s] = -(K * (1 - p[k]))
            else:
                P0[s, pas_idx(0)] = 1.0
                R0[s] = -K
    return P0, P1, R0, R1, state_labels

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

# Placeholder closed-form function (replace with your own)
def closed_form_indices(p, beta, K, C, C_switch):
    # Replace with actual closed-form computation.
    # Returning NaN for illustration; your implementation should fill these.
    return np.nan, np.full(len(p)+1, np.nan)   # (w01, w_passive[0..N])

# =====================================================================
# 2. Discounted and average solvers (unchanged from your code)
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
    V = np.linalg.solve(np.eye(S) - beta * P_pi, r_pi)
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
    return V, Q0, Q1

def whittle_index_discounted(state, solver, beta, P0, P1, R0, R1,
                             w_min=-1000.0, w_max=1000.0, tol_w=1e-5, tol_mdp=1e-8):
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
            w_min = w_mid
            f_min = f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0

# Average-cost solvers
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
    return V, delta, Q0, Q1

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
    return sol[:n], sol[-1]

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
    return h, g, Q0, Q1

def whittle_index_bisection_avg(state, solver, R0, R1, P0, P1,
                                w_min=-2000.0, w_max=2000.0, tol_w=1e-10,
                                tol_mdp=1e-12, ref=0):
    solve_func = solve_rvi if solver == 'rvi' else solve_pi_avg
    def f(w):
        _, _, Q0, Q1 = solve_func(R0, R1, P0, P1, w, ref=ref, tol=tol_mdp)
        return Q0[state] - Q1[state]
    f_min, f_max = f(w_min), f(w_max)
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
# 3. Verification helpers (discounted / average)
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
    unique = sorted(set(indices))
    test_Ws = set()
    for lam in unique:
        test_Ws.update([lam - w_eps, lam, lam + w_eps])
    if unique:
        test_Ws.add(unique[0] - 10.0)
    test_Ws = sorted(test_Ws)

    failures = []
    for W in test_Ws:
        pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, W, beta)
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - W) + beta * (P1 @ V)
        for s in range(S):
            if pi[s] == 0 and Q1[s] > Q0[s] + tol:
                failures.append(f"W={W:.6f}, {state_labels[s]}: passive but Q1 > Q0")
            if pi[s] == 1 and Q0[s] > Q1[s] + tol:
                failures.append(f"W={W:.6f}, {state_labels[s]}: active but Q0 > Q1")
    if not failures:
        return True, f"Discounted verification passed for {len(test_Ws)} test W values."
    else:
        return False, f"Discounted verification FAILED: first errors:\n" + "\n".join(failures[:10])

def verify_average_threshold(indices, P0, P1, R0, R1, state_labels, tol=1e-5, w_eps=1e-6):
    S = len(indices)
    unique = sorted(set(indices))
    test_Ws = set()
    for lam in unique:
        test_Ws.update([lam - w_eps, lam, lam + w_eps])
    if unique:
        test_Ws.add(unique[0] - 10.0)
    test_Ws = sorted(test_Ws)

    failures = []
    for W in test_Ws:
        pi = np.array([1 if indices[s] >= W else 0 for s in range(S)])
        h, g = policy_evaluation_avg(pi, R0, R1, P0, P1, W, ref=0)
        Q0 = R0 + P0 @ h
        Q1 = R1 - W + P1 @ h
        for s in range(S):
            if pi[s] == 0 and Q1[s] > Q0[s] + tol:
                failures.append(f"W={W:.6f}, {state_labels[s]}: passive but Q1 > Q0")
            if pi[s] == 1 and Q0[s] > Q1[s] + tol:
                failures.append(f"W={W:.6f}, {state_labels[s]}: active but Q0 > Q1")
    if not failures:
        return True, f"Average-cost verification passed for {len(test_Ws)} test W values."
    else:
        return False, f"Average-cost verification FAILED: first errors:\n" + "\n".join(failures[:10])

# =====================================================================
# 4. LaTeX formatting functions (adjusted to match example)
# =====================================================================
def escape_latex(s):
    """Escape special LaTeX characters."""
    return (s.replace('\\', '\\textbackslash ')
            .replace('&', '\\&')
            .replace('%', '\\%')
            .replace('$', '\\$')
            .replace('#', '\\#')
            .replace('_', '\\_')
            .replace('{', '\\{')
            .replace('}', '\\}')
            .replace('~', '\\textasciitilde ')
            .replace('^', '\\textasciicircum '))

def format_index(val, tol=1e-6):
    """Return a string for an index, or '--' if NaN."""
    if np.isnan(val):
        return "--"
    return f"{val:.6f}"

def write_parameter_table(f, C, K, C_switch, beta, N, state_labels):
    """Modified: switching cost row is coloured blue."""
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write("\\caption{Model Parameters for Instance \\theinstance}\n")
    f.write("\\begin{tabular}{@{}llr@{}}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Parameter} & \\textbf{Description} & \\textbf{Value} \\\\\n")
    f.write("\\midrule\n")
    f.write(f"$N$ & Maximum degradation level & {N} \\\\\n")
    f.write(f"$k$ & Number of states & {len(state_labels)} \\\\\n")
    f.write(f"$C$ & Active repair cost ($C_m$) & {C} \\\\\n")
    # Blue switching cost
    f.write(f"\\textcolor{{blue}}{{$c_{{\\text{{switch}}}}$}} & "
            f"\\textcolor{{blue}}{{Switching cost}} & "
            f"\\textcolor{{blue}}{{{C_switch}}} \\\\\n")
    f.write(f"$K$ & Failure cost ($K_m$) & {K} \\\\\n")
    f.write(f"$\\beta$ & Discount factor & {beta} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n\n")

def write_survival_probs(f, p_full):
    """Modified: subsection header and quad separators."""
    f.write("\\subsection*{Survival Probabilities (Instance \\theinstance)}\n")
    N = len(p_full) - 1
    probs = ",\\quad ".join([f"p_{{{k}}} = {p_full[k]:.4f}" for k in range(N+1)])
    f.write(f"\\[\n{probs}\n\\]\n\n")

def write_transition_matrix(f, P, row_labels, col_labels):
    """Transition matrix with resizebox and reduced column separation."""
    header = " & " + " & ".join(col_labels) + " \\\\"
    rows = [header, "\\hline"]
    for i, rl in enumerate(row_labels):
        vals = " & ".join(f"{P[i,j]:.6f}" for j in range(len(col_labels)))
        rows.append(f"{rl} & {vals} \\\\")
    # Build the array environment
    array_code = "\\begin{array}{c|" + "c"*len(col_labels) + "}\n" + "\n".join(rows) + "\n\\end{array}"
    f.write("\\[\n")
    f.write("\\resizebox{\\linewidth}{!}{%\n")
    f.write("\\setlength{\\tabcolsep}{1.5pt}%\n")
    f.write(f"${array_code}$%\n")
    f.write("}\n")
    f.write("\\]\n")

def write_reward_vectors_side_by_side(f, R0, R1, state_labels):
    """Active and Passive reward vectors in two minipages."""
    f.write("\\subsection*{Reward Vectors}\n\n")
    f.write("\\begin{center}\n")
    # Active Reward (R1)
    f.write("\\begin{minipage}[t]{0.45\\linewidth}\n")
    f.write("\\centering\n")
    f.write("\\textbf{Active Reward}\n")
    f.write("\\[\n")
    f.write("\\begin{array}{c|c}\n")
    f.write("\\text{State} & \\text{Reward} \\\\\\hline\n")
    for i, rl in enumerate(state_labels):
        f.write(f"{rl} & {R1[i]:.6f} \\\\\n")
    f.write("\\end{array}\n")
    f.write("\\]\n")
    f.write("\\end{minipage}\n")
    f.write("\\hspace{1cm}\n")
    # Passive Reward (R0)
    f.write("\\begin{minipage}[t]{0.45\\linewidth}\n")
    f.write("\\centering\n")
    f.write("\\textbf{Passive Reward}\n")
    f.write("\\[\n")
    f.write("\\begin{array}{c|c}\n")
    f.write("\\text{State} & \\text{Reward} \\\\\\hline\n")
    for i, rl in enumerate(state_labels):
        f.write(f"{rl} & {R0[i]:.6f} \\\\\n")
    f.write("\\end{array}\n")
    f.write("\\]\n")
    f.write("\\end{minipage}\n")
    f.write("\\end{center}\n\n")

def write_split_index_table(f, state_labels, methods_dict, caption, label):
    """
    Split table: left half = passive states (.,0), right half = active states (.,1).
    methods_dict: keys are column names (e.g., 'VI','PI','Package'), values are lists of indices.
    """
    N = (len(state_labels)//2) - 1  # number of degradation levels
    # Passive states: indices for (k,0), k=0..N
    passive_idx = [N+1 + k for k in range(N+1)]   # after active states
    active_idx = [k for k in range(N+1)]          # (k,1)

    cols = list(methods_dict.keys())
    ncol = len(cols)

    # Build header
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write(f"\\caption{{{caption}}}\n")
    f.write("\\begin{tabular}{@{}l" + "c"*ncol + "@{\\hspace{2em}}l" + "c"*ncol + "@{}}\n")
    f.write("\\toprule\n")
    f.write("\\multicolumn{" + str(ncol+1) + "}{c}{\\textbf{States (.,0)}} & "
            "\\multicolumn{" + str(ncol+1) + "}{c}{\\textbf{States (.,1)}} \\\\\n")
    # Column subheadings
    subhead = " & ".join(["\\textbf{State}"] + [f"\\textbf{{{c}}}" for c in cols])
    f.write("\\cmidrule(lr){1-" + str(ncol+1) + "} \\cmidrule(lr){" + str(ncol+2) + "-" + str(2*ncol+2) + "}\n")
    f.write(subhead + " & " + subhead + " \\\\\n")
    f.write("\\midrule\n")

    for k in range(N+1):
        row_left = state_labels[passive_idx[k]]
        row_right = state_labels[active_idx[k]]
        left_vals = [row_left] + [format_index(methods_dict[c][passive_idx[k]]) for c in cols]
        right_vals = [row_right] + [format_index(methods_dict[c][active_idx[k]]) for c in cols]
        f.write(" & ".join(left_vals) + " & " + " & ".join(right_vals) + " \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n\n")

# =====================================================================
# 5. Main experiment loop – LaTeX output
# =====================================================================
def run_experiments_latex():
    # Parameter ranges
    C_vals = [5.0, 10.0, 20.0, 400.0, 600.0, 1200.0]
    K_vals = [500.0, 1000.0, 2000.0]
    C_switch_vals = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 75.0, 100.0,
                     200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0]
    beta_vals = [0.9, 0.95, 0.9999]

    N = 5
    num_random_matrices = 3
    np.random.seed(42)
    trans_matrices = []
    for _ in range(num_random_matrices):
        p = np.sort(np.random.uniform(0.0, 1.0, N))[::-1]
        p_full = np.append(p, 0.0)
        trans_matrices.append(p_full)

    q = make_geometric_repair_dist(N, r=0.5)

    total_combos = (len(trans_matrices) * len(C_vals) * len(K_vals) *
                    len(C_switch_vals) * len(beta_vals))
    instance_counter = 0

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"Whittle_Experiment_LaTeX_{now_str}.tex"

    with open(filename, "w") as f:
        # LaTeX preamble
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{multirow}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage{amsmath}\n")
        f.write("\\newcounter{instance}\n")
        f.write("\\setcounter{instance}{0}\n")
        f.write("\\newcommand{\\theinstance}{\\arabic{instance}}\n")
        f.write("\\begin{document}\n\n")
        f.write("\\title{Whittle Indices: Closed-Form vs VI/PI vs Package}\n")
        f.write("\\author{Automatic Experiment}\n")
        f.write("\\date{\\today}\n")
        f.write("\\maketitle\n\n")

        # Summary of parameter ranges
        f.write("\\section*{Experiment Setup}\n")
        f.write("\\begin{itemize}\n")
        f.write(f"\\item $C$ (activation cost): {C_vals}\n")
        f.write(f"\\item $K$ (penalty cost): {K_vals}\n")
        f.write(f"\\item $c_{{\\text{{switch}}}}$ (switching cost): {C_switch_vals}\n")
        f.write(f"\\item $\\beta$ (discount factor): {beta_vals}\n")
        f.write(f"\\item Number of transition matrices: {len(trans_matrices)}\n")
        f.write(f"\\item Total combinations: {total_combos}\n")
        f.write("\\end{itemize}\n\n")

        # For aggregated comparison: store VI vs Package for C_switch=1.0
        aggregated_data = {}   # key: beta, value: {state_label: {'vi':..., 'package':...}}

        for mat_idx, p_full in enumerate(trans_matrices, start=1):
            N_current = len(p_full) - 1
            f.write(f"\\section{{Transition Matrix {mat_idx}}}\n")
            f.write(f"Survival probabilities: $p = [{', '.join(f'{p_full[i]:.4f}' for i in range(N_current+1))}]$\n\n")

            for C in C_vals:
                for K in K_vals:
                    for C_switch in C_switch_vals:
                        # Aggregate only for the specific combination matching the example
                        do_aggregate = (mat_idx == 1 and C == 5.0 and K == 500.0 and C_switch == 1.0)

                        for beta in beta_vals:
                            instance_counter += 1
                            print(f"[{instance_counter}/{total_combos}] "
                                  f"C={C}, K={K}, Csw={C_switch}, beta={beta}, mat={mat_idx}",
                                  flush=True)

                            # Build MDP
                            P0, P1, R0, R1, state_labels = build_extended_mdp_imperfect(
                                N_current, p_full[:-1], q, C, K, C_switch)

                            # Increment instance counter for LaTeX
                            f.write("\\stepcounter{instance}\n")

                            # ---- Problem parameters ----
                            write_parameter_table(f, C, K, C_switch, beta, N_current, state_labels)

                            # ---- Survival probabilities ----
                            write_survival_probs(f, p_full)

                            # ---- Passive action matrices ----
                            f.write("\\subsection*{Transition and Reward for Passive Action ($a=0$)}\n")
                            f.write("\\subsubsection*{Transition Matrix}\n")
                            write_transition_matrix(f, P0, state_labels, state_labels)

                            # ---- Active action matrices ----
                            f.write("\\subsection*{Transition and Reward for Active Action ($a=1$)}\n")
                            f.write("\\subsubsection*{Transition Matrix}\n")
                            write_transition_matrix(f, P1, state_labels, state_labels)

                            # ---- Reward vectors (side-by-side) ----
                            write_reward_vectors_side_by_side(f, R0, R1, state_labels)

                            # ---- Discounted indices ----
                            disc_results = {}
                            for method in ['vi', 'pi']:
                                indices = []
                                for s in range(len(R0)):
                                    try:
                                        lam = whittle_index_discounted(
                                            s, method, beta, P0, P1, R0, R1,
                                            w_min=-2000.0, w_max=2000.0, tol_w=1e-5, tol_mdp=1e-8)
                                        indices.append(lam)
                                    except:
                                        indices.append(np.nan)
                                disc_results[method] = indices

                            # Closed-form
                            try:
                                w01_cf, w_pass_cf = closed_form_indices(p_full[:-1], beta, K, C, C_switch)
                                closed_disc = [w01_cf] + list(w_pass_cf)   # length = N+2
                            except:
                                closed_disc = [np.nan] * len(state_labels)

                            # Package
                            if PKG_AVAILABLE:
                                try:
                                    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                                    pkg_disc = model.whittle_indices(discount=beta)
                                except:
                                    pkg_disc = [np.nan] * len(state_labels)
                            else:
                                pkg_disc = [np.nan] * len(state_labels)

                            # Build split table for discounted indices (VI, PI, Package)
                            methods_disc = {
                                'VI': disc_results['vi'],
                                'PI': disc_results['pi'],
                                'Package': pkg_disc
                            }
                            write_split_index_table(
                                f, state_labels, methods_disc,
                                caption=f"Discounted Whittle indices ($\\beta={beta}$)",
                                label=f"tab:disc_{instance_counter}"
                            )

                            # Monotonicity and verification (using closed-form)
                            expected_order = [1, 0] + list(range(2, len(state_labels)))
                            mono_ok, mono_msg = check_monotonicity(closed_disc, state_labels, expected_order)
                            disc_verif_ok, disc_verif_msg = verify_discounted_threshold(
                                closed_disc, P0, P1, R0, R1, beta, state_labels)

                            f.write("\\noindent\\textbf{Discounted monotonicity (closed-form):} "
                                    f"{'PASS' if mono_ok else 'FAIL'}\n\n")
                            f.write(f"{escape_latex(mono_msg)}\n\n")
                            f.write("\\noindent\\textbf{Discounted indexability verification:} "
                                    f"{'PASS' if disc_verif_ok else 'FAIL'}\n\n")
                            f.write(f"{escape_latex(disc_verif_msg)}\n\n")

                            # Store VI & Package for aggregated comparison
                            if do_aggregate:
                                aggregated_data.setdefault(beta, {})
                                for i, s in enumerate(state_labels):
                                    aggregated_data[beta][s] = {
                                        'vi': disc_results['vi'][i],
                                        'package': pkg_disc[i]
                                    }

                            # ---- Average-cost indices ----
                            avg_results = {}
                            for method in ['rvi', 'pi']:
                                indices = []
                                for s in range(len(R0)):
                                    try:
                                        lam = whittle_index_bisection_avg(
                                            s, method, R0, R1, P0, P1,
                                            w_min=-2000.0, w_max=2000.0, tol_w=1e-8, tol_mdp=1e-12)
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

                            methods_avg = {
                                'RVI': avg_results['rvi'],
                                'PI': avg_results['pi'],
                                'Package': pkg_avg
                            }
                            write_split_index_table(
                                f, state_labels, methods_avg,
                                caption="Average-cost Whittle indices",
                                label=f"tab:avg_{instance_counter}"
                            )

                            avg_verif_ok, avg_verif_msg = verify_average_threshold(
                                avg_results['rvi'], P0, P1, R0, R1, state_labels)

                            mono_avg_ok, mono_avg_msg = check_monotonicity(
                                avg_results['rvi'], state_labels, expected_order)
                            f.write("\\noindent\\textbf{Average-cost monotonicity (RVI):} "
                                    f"{'PASS' if mono_avg_ok else 'FAIL'}\n\n")
                            f.write(f"{escape_latex(mono_avg_msg)}\n\n")
                            f.write("\\noindent\\textbf{Average-cost indexability verification:} "
                                    f"{'PASS' if avg_verif_ok else 'FAIL'}\n\n")
                            f.write(f"{escape_latex(avg_verif_msg)}\n\n")

                            f.write("\\newpage\n")

        # ---- Aggregated comparison table (VI vs Package for C_switch=1.0) ----
        if aggregated_data:
            f.write("\\section*{Whittle Indices Comparison (Switching Cost $C_{\\text{switch}}=1.0$)}\n")
            f.write("\\begin{table}[h!]\n\\centering\n")
            f.write("\\caption{Discounted Whittle Indices: VI vs Package ($C_{\\text{switch}}=1.0$)}\n")
            f.write("\\begin{tabular}{@{}l")
            for _ in beta_vals:
                f.write("cc")
            f.write("{@{\\hskip 0pt}}\n")
            f.write("\\toprule\n")
            f.write("\\multirow{2}{*}{\\textbf{State}} & ")
            for i, beta in enumerate(beta_vals):
                f.write(f"\\multicolumn{{2}}{{c}}{{$\\beta = {beta}$}}")
                if i < len(beta_vals)-1:
                    f.write(" & ")
            f.write(" \\\\\n")
            f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n")
            f.write(" & ")
            for i, beta in enumerate(beta_vals):
                f.write("\\textbf{VI} & \\textbf{Package}")
                if i < len(beta_vals)-1:
                    f.write(" & ")
            f.write(" \\\\\n\\midrule\n")

            # State order: passive (k,0) then active (k,1) for k=0..N
            N_current = len(trans_matrices[0]) - 1  # same N for all matrices
            state_order = [f"({k},0)" for k in range(N_current+1)] + [f"({k},1)" for k in range(N_current+1)]
            for s in state_order:
                # Label as "Passive (k,0)" or "Active (k,1)"
                if s.endswith(",0)"):
                    label = "Passive " + s
                else:
                    label = "Active " + s
                row = f"{label} & "
                entries = []
                for beta in beta_vals:
                    if beta in aggregated_data and s in aggregated_data[beta]:
                        vals = aggregated_data[beta][s]
                        vi = vals['vi']; pkg = vals['package']
                        vi_str = format_index(vi)
                        pkg_str = format_index(pkg)
                        if not np.isnan(vi) and not np.isnan(pkg) and abs(vi - pkg) > 1e-4:
                            vi_str = f"\\textcolor{{blue}}{{{vi_str}}}"
                            pkg_str = f"\\textcolor{{blue}}{{{pkg_str}}}"
                        entries.append(f"{vi_str} & {pkg_str}")
                    else:
                        entries.append("-- & --")
                row += " & ".join(entries) + " \\\\"
                f.write(row + "\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        f.write("\\end{document}\n")

    print(f"LaTeX file saved as '{filename}'.")

if __name__ == "__main__":
    run_experiments_latex()
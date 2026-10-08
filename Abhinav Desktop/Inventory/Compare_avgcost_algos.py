# compare_avgcost_algos.py
# Full implementation: DP-RVI (ground truth), RVI-Q (sync & async), SSP-Q (sync & async)
# Numerical comparison plots and summary table.

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math

# -------------------------------
# MDP specification (user's)
# -------------------------------
states = [0,1]
actions = [0,1]
P = {
    (0,0): np.array([0.5, 0.5]),
    (0,1): np.array([0.2, 0.8]),
    (1,0): np.array([0.7, 0.3]),
    (1,1): np.array([0.4, 0.6])
}
c = {
    (0,0): 1.0,
    (0,1): 1.5,
    (1,0): 0.5,
    (1,1): 0.8
}
ref_state = 0  # anchor state for relative values / f(Q)

# -------------------------------
# DP-RVI (exact model-based) for ground truth rho and V
# -------------------------------
def dp_rvi(P, c, ref_state=0, tol=1e-12, max_iter=10000):
    nS = len(states); nA = len(actions)
    V = np.zeros(nS)
    rho = 0.0
    for it in range(max_iter):
        V_old = V.copy()
        Q = np.zeros((nS, nA))
        for s in states:
            for a in actions:
                Q[s,a] = c[(s,a)] + np.dot(P[(s,a)], V_old)
        V_new = np.min(Q, axis=1)
        shift = V_new[ref_state]
        V = V_new - shift
        rho = shift
        if np.max(np.abs(V - V_old)) < tol:
            break
    policy = {s: int(np.argmin([c[(s,a)] + np.dot(P[(s,a)], V) for a in actions])) for s in states}
    return rho, V, policy, it+1

# -------------------------------
# Helper sampling function
# -------------------------------
def sample_next(pdist):
    return np.searchsorted(np.cumsum(pdist), random.random())

# -------------------------------
# RVI-Q-learning (synchronous)
# f(Q) = min_b Q(ref_state,b)
# -------------------------------
def rvi_q_learning_synchronous(P, c, states, actions, ref_state=0, alpha=1.0, max_iter=5000, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    nS = len(states); nA = len(actions)
    Q = np.zeros((nS, nA))
    f_trace = []
    for n in range(max_iter):
        gamma_n = alpha / (n+1.0)
        fQ = float(np.min(Q[ref_state,:]))
        # synchronous sampled updates
        delta = np.zeros_like(Q)
        for s in states:
            for a in actions:
                s_next = sample_next(P[(s,a)])
                g_sample = c[(s,a)]
                minQnext = float(np.min(Q[s_next,:]))
                target = g_sample + minQnext - fQ - Q[s,a]
                delta[s,a] = gamma_n * target
        Q += delta
        f_trace.append(fQ)
    final_policy = {s:int(np.argmin(Q[s,:])) for s in states}
    # estimate relative V from Q (V(s)=min_a Q(s,a)) and normalize
    V_est = np.min(Q, axis=1)
    V_est = V_est - V_est[ref_state]
    rho_est = float(np.min(Q[ref_state,:]))
    return Q, f_trace, final_policy, rho_est, V_est

# -------------------------------
# RVI-Q-learning (asynchronous)
# update one (s,a) per iter; stepsize per-visit
# -------------------------------
def rvi_q_learning_asynchronous(P, c, states, actions, ref_state=0, alpha=1.0, max_iter=20000, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    Q = np.zeros((len(states), len(actions)))
    visit_count = np.zeros_like(Q)
    f_trace = []
    for n in range(max_iter):
        s = random.choice(states)
        a = random.choice(actions)
        visit_count[s,a] += 1
        gamma = alpha / visit_count[s,a]
        s_next = sample_next(P[(s,a)])
        g_sample = c[(s,a)]
        minQnext = float(np.min(Q[s_next,:]))
        fQ = float(np.min(Q[ref_state,:]))
        target = g_sample + minQnext - fQ - Q[s,a]
        Q[s,a] += gamma * target
        f_trace.append(fQ)
    final_policy = {s:int(np.argmin(Q[s,:])) for s in states}
    V_est = np.min(Q, axis=1)
    V_est = V_est - V_est[ref_state]
    rho_est = float(np.min(Q[ref_state,:]))
    return Q, f_trace, final_policy, rho_est, V_est

# -------------------------------
# SSP-Q-learning (synchronous)
# uses lambda update b(n)=1/(n+1)^p with p in (0.5,1); choose p=0.75
# project lambda to [-K,K]
# -------------------------------
def ssp_q_learning_synchronous(P, c, states, actions, ref_state=0, lambda0=0.0, alpha=1.0, max_iter=5000, seed=None, K=10.0):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    Q = np.zeros((len(states), len(actions)))
    lamb = lambda0
    p = 0.75
    lambda_trace = []
    for n in range(max_iter):
        gamma_n = alpha / (n+1.0)
        b_n = 1.0 / ((n+1.0)**p)
        delta = np.zeros_like(Q)
        for s in states:
            for a in actions:
                s_next = sample_next(P[(s,a)])
                g_sample = c[(s,a)]
                minQnext = float(np.min(Q[s_next,:]))
                target = g_sample + minQnext - lamb - Q[s,a]
                delta[s,a] = gamma_n * target
        Q += delta
        lamb = lamb + b_n * float(np.min(Q[ref_state,:]))
        lamb = max(-K, min(K, lamb))
        lambda_trace.append(lamb)
    final_policy = {s:int(np.argmin(Q[s,:])) for s in states}
    V_est = np.min(Q, axis=1)
    V_est = V_est - V_est[ref_state]
    return Q, lambda_trace, final_policy, float(lambda_trace[-1]), V_est

# -------------------------------
# SSP-Q-learning (asynchronous)
# one (s,a) per iter, lambda updated slowly
# -------------------------------
def ssp_q_learning_asynchronous(P, c, states, actions, ref_state=0, lambda0=0.0, alpha=1.0, max_iter=20000, seed=None, K=10.0):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    Q = np.zeros((len(states), len(actions)))
    lamb = lambda0
    visit_count = np.zeros_like(Q)
    p = 0.75
    lambda_trace = []
    for n in range(max_iter):
        s = random.choice(states)
        a = random.choice(actions)
        visit_count[s,a] += 1
        gamma = alpha / visit_count[s,a]
        s_next = sample_next(P[(s,a)])
        g_sample = c[(s,a)]
        minQnext = float(np.min(Q[s_next,:]))
        target = g_sample + minQnext - lamb - Q[s,a]
        Q[s,a] += gamma * target
        b_n = 1.0 / ((n+1.0)**p)
        lamb = lamb + b_n * float(np.min(Q[ref_state,:]))
        lamb = max(-K, min(K, lamb))
        lambda_trace.append(lamb)
    final_policy = {s:int(np.argmin(Q[s,:])) for s in states}
    V_est = np.min(Q, axis=1)
    V_est = V_est - V_est[ref_state]
    return Q, lambda_trace, final_policy, float(lambda_trace[-1]), V_est

# -------------------------------
# Run multiple trials and aggregate statistics
# -------------------------------
def run_trials(n_runs=6, max_iter_sync=5000, max_iter_async=20000, seed_base=123):
    random.seed(seed_base); np.random.seed(seed_base)
    rho_true, V_true, policy_true, dp_iters = dp_rvi(P, c, ref_state)
    print(f"DP-RVI ground truth: rho={rho_true:.6f}, V={V_true}, policy={policy_true}, iters={dp_iters}")

    rvi_sync_traces = []
    rvi_async_traces = []
    ssp_sync_traces = []
    ssp_async_traces = []
    summary_rows = []

    for run in range(n_runs):
        seed = seed_base + run
        # RVI-Q synchronous
        Qr_sync, ftrace_r_sync, polr_sync, rho_r_sync, V_r_sync = rvi_q_learning_synchronous(P, c, states, actions, ref_state=ref_state, alpha=1.0, max_iter=max_iter_sync, seed=seed)
        # RVI-Q asynchronous
        Qr_async, ftrace_r_async, polr_async, rho_r_async, V_r_async = rvi_q_learning_asynchronous(P, c, states, actions, ref_state=ref_state, alpha=1.0, max_iter=max_iter_async, seed=seed)
        # SSP synchronous
        Qs_sync, ltrace_s_sync, pols_sync, lambda_s_sync, V_s_sync = ssp_q_learning_synchronous(P, c, states, actions, ref_state=ref_state, lambda0=0.0, alpha=1.0, max_iter=max_iter_sync, seed=seed, K=10.0)
        # SSP asynchronous
        Qs_async, ltrace_s_async, pols_async, lambda_s_async, V_s_async = ssp_q_learning_asynchronous(P, c, states, actions, ref_state=ref_state, lambda0=0.0, alpha=1.0, max_iter=max_iter_async, seed=seed, K=10.0)

        rvi_sync_traces.append(ftrace_r_sync)
        rvi_async_traces.append(ftrace_r_async)
        ssp_sync_traces.append(ltrace_s_sync)
        ssp_async_traces.append(ltrace_s_async)

        summary_rows.append({
            'run': run,
            'DP_rho': rho_true,
            'DP_V0': V_true[0],
            'DP_V1': V_true[1],
            'RVI_sync_rho_last': rho_r_sync,
            'RVI_sync_V0_last': V_r_sync[0],
            'RVI_sync_V1_last': V_r_sync[1],
            'RVI_sync_policy': str(polr_sync),
            'RVI_async_rho_last': rho_r_async,
            'RVI_async_V0_last': V_r_async[0],
            'RVI_async_V1_last': V_r_async[1],
            'RVI_async_policy': str(polr_async),
            'SSP_sync_lambda_last': lambda_s_sync,
            'SSP_sync_V0_last': V_s_sync[0],
            'SSP_sync_V1_last': V_s_sync[1],
            'SSP_sync_policy': str(pols_sync),
            'SSP_async_lambda_last': lambda_s_async,
            'SSP_async_V0_last': V_s_async[0],
            'SSP_async_V1_last': V_s_async[1],
            'SSP_async_policy': str(pols_async)
        })

    # convert traces to arrays for plotting (pad/truncate for equal length where needed)
    rvi_sync_arr = np.array(rvi_sync_traces)  # shape (n_runs, max_iter_sync)
    rvi_async_arr = np.array([np.array(t[:max_iter_sync]) for t in rvi_async_traces])  # truncate to max_iter_sync
    ssp_sync_arr = np.array(ssp_sync_traces)
    ssp_async_arr = np.array([np.array(t[:max_iter_sync]) for t in ssp_async_traces])  # truncate

    return summary_rows, rvi_sync_arr, rvi_async_arr, ssp_sync_arr, ssp_async_arr, rho_true, V_true, policy_true

# Parameters
n_runs = 6
max_iter_sync = 5000
max_iter_async = 20000

# Run experiments
summary_rows, rvi_sync_arr, rvi_async_arr, ssp_sync_arr, ssp_async_arr, rho_true, V_true, policy_true = run_trials(
    n_runs=n_runs, max_iter_sync=max_iter_sync, max_iter_async=max_iter_async, seed_base=1234)

# -------------------------------
# Compute means and stds for plotting
# -------------------------------
rvi_sync_mean = np.mean(rvi_sync_arr, axis=0)
rvi_sync_std = np.std(rvi_sync_arr, axis=0)
rvi_async_mean = np.mean(rvi_async_arr, axis=0)
rvi_async_std = np.std(rvi_async_arr, axis=0)
ssp_sync_mean = np.mean(ssp_sync_arr, axis=0)
ssp_sync_std = np.std(ssp_sync_arr, axis=0)
ssp_async_mean = np.mean(ssp_async_arr, axis=0)
ssp_async_std = np.std(ssp_async_arr, axis=0)

# -------------------------------
# Plots
# -------------------------------
iters = np.arange(1, max_iter_sync+1)
plt.figure(figsize=(9,4))
plt.plot(iters, rvi_sync_mean, label='RVI-Q sync (mean f(Q))')
plt.fill_between(iters, rvi_sync_mean - rvi_sync_std, rvi_sync_mean + rvi_sync_std, alpha=0.2)
plt.plot(iters, rvi_async_mean, label='RVI-Q async (mean f(Q, truncated))')
plt.fill_between(iters, rvi_async_mean - rvi_async_std, rvi_async_mean + rvi_async_std, alpha=0.15)
plt.hlines(rho_true, 1, max_iter_sync, linestyles='dashed', label='DP rho (true)')
plt.title("RVI-Q (sync & async): f(Q) trace (mean ± std) vs DP rho")
plt.xlabel("iteration")
plt.ylabel("f(Q) = min_b Q(ref,b)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,4))
plt.plot(iters, ssp_sync_mean, label='SSP-Q sync (mean lambda)')
plt.fill_between(iters, ssp_sync_mean - ssp_sync_std, ssp_sync_mean + ssp_sync_std, alpha=0.2)
plt.plot(iters, ssp_async_mean, label='SSP-Q async (mean lambda, truncated)')
plt.fill_between(iters, ssp_async_mean - ssp_async_std, ssp_async_mean + ssp_async_std, alpha=0.15)
plt.hlines(rho_true, 1, max_iter_sync, linestyles='dashed', label='DP rho (true)')
plt.title("SSP-Q (sync & async): lambda trace (mean ± std) vs DP rho")
plt.xlabel("iteration")
plt.ylabel("lambda")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Summary table showing convergence of rho/V (last values from each run)
# -------------------------------
df_summary = pd.DataFrame(summary_rows)
display_cols = ['run','DP_rho','DP_V0','DP_V1',
                'RVI_sync_rho_last','RVI_sync_V0_last','RVI_sync_V1_last','RVI_sync_policy',
                'RVI_async_rho_last','RVI_async_V0_last','RVI_async_V1_last','RVI_async_policy',
                'SSP_sync_lambda_last','SSP_sync_V0_last','SSP_sync_V1_last','SSP_sync_policy',
                'SSP_async_lambda_last','SSP_async_V0_last','SSP_async_V1_last','SSP_async_policy']
print("\nSummary table (last-value estimates per run):")
print(df_summary[display_cols].to_string(index=False))

print("\nDP ground truth:")
print({'rho_true': rho_true, 'V_true': V_true, 'policy_true': policy_true})

# Save CSV if desired
df_summary.to_csv("avgcost_comparison_summary.csv", index=False)
print("\nSaved summary CSV: avgcost_comparison_summary.csv")

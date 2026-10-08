# compare_ssp_rvi_q.py
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

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
i0 = 0  # reference state for f(Q)

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
# Synchronous RVI-Q-learning
# f(Q) = min_b Q(i0,b)
# -------------------------------
def rvi_q_learning_synchronous(P, c, states, actions, i0=0, alpha=1.0, max_iter=5000, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    nS = len(states); nA = len(actions)
    Q = np.zeros((nS, nA))
    f_trace = []
    for n in range(max_iter):
        gamma_n = alpha / (n+1.0)
        fQ = float(np.min(Q[i0,:]))
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
    return Q, f_trace, final_policy

# -------------------------------
# Synchronous SSP-Q-learning (with lambda update)
# b(n) = 1/(n+1)^p with p in (0.5,1), choose p=0.75
# Project lambda to [-K,K]
# -------------------------------
def ssp_q_learning_synchronous(P, c, states, actions, lambda0=0.0, alpha=1.0, max_iter=5000, seed=None, K=10.0):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    nS = len(states); nA = len(actions)
    Q = np.zeros((nS, nA))
    lamb = lambda0
    lambda_trace = []
    pwr = 0.75
    for n in range(max_iter):
        gamma_n = alpha / (n+1.0)
        b_n = 1.0 / ((n+1.0)**pwr)
        delta = np.zeros_like(Q)
        for s in states:
            for a in actions:
                s_next = sample_next(P[(s,a)])
                g_sample = c[(s,a)]
                minQnext = float(np.min(Q[s_next,:]))
                # Target: g + min Q(next,.) - lambda - Q(s,a)
                target = g_sample + minQnext - lamb - Q[s,a]
                delta[s,a] = gamma_n * target
        Q += delta
        # lambda slow update with projection
        lamb = lamb + b_n * float(np.min(Q[i0,:]))
        lamb = max(-K, min(K, lamb))
        lambda_trace.append(lamb)
    final_policy = {s:int(np.argmin(Q[s,:])) for s in states}
    return Q, lambda_trace, final_policy

# -------------------------------
# Asynchronous SSP-Q-learning
# Update one (s,a) per iter, update lambda each iter with b(n)
# -------------------------------
def ssp_q_learning_asynchronous(P, c, states, actions, lambda0=0.0, alpha=1.0, max_iter=40000, seed=None, K=10.0):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    nS = len(states); nA = len(actions)
    Q = np.zeros((nS, nA))
    lamb = lambda0
    visit_count = np.zeros((nS, nA))
    lambda_trace = []
    pwr = 0.75  # exponent for b(n)
    for n in range(max_iter):
        s = random.choice(states); a = random.choice(actions)
        visit_count[s,a] += 1
        gamma = alpha / visit_count[s,a]
        b_n = 1.0 / ((n+1.0)**pwr)
        s_next = sample_next(P[(s,a)])
        g_sample = c[(s,a)]
        minQnext = float(np.min(Q[s_next,:]))
        target = g_sample + minQnext - lamb - Q[s,a]
        Q[s,a] += gamma * target
        # slow lambda update with projection
        lamb = lamb + b_n * float(np.min(Q[i0,:]))
        lamb = max(-K, min(K, lamb))
        lambda_trace.append(lamb)
    final_policy = {s:int(np.argmin(Q[s,:])) for s in states}
    return Q, lambda_trace, final_policy

# -------------------------------
# Run experiments: multiple trials for each algorithm
# -------------------------------
def run_experiments(n_runs=8, max_iter_sync=8000, max_iter_async=40000):
    rho_true, V_true, policy_true, dp_iters = dp_rvi(P, c, ref_state=i0)
    print(f"DP-RVI ground truth: rho={rho_true:.6f}, V={V_true}, policy={policy_true}, iters={dp_iters}")

    rvi_f_traces = []
    rvi_pols = []
    ssp_sync_lambda_traces = []
    ssp_sync_pols = []
    ssp_async_lambda_traces = []
    ssp_async_pols = []

    for run in range(n_runs):
        seed = 1000 + run
        Qr, ftrace, polr = rvi_q_learning_synchronous(P, c, states, actions, i0=i0, alpha=1.0, max_iter=max_iter_sync, seed=seed)
        rvi_f_traces.append(ftrace); rvi_pols.append(polr)
        Qs, ltrace_s, pols = ssp_q_learning_synchronous(P, c, states, actions, lambda0=0.0, alpha=1.0, max_iter=max_iter_sync, seed=seed, K=10.0)
        ssp_sync_lambda_traces.append(ltrace_s); ssp_sync_pols.append(pols)
        Qa, ltrace_a, pola = ssp_q_learning_asynchronous(P, c, states, actions, lambda0=0.0, alpha=1.0, max_iter=max_iter_async, seed=seed, K=10.0)
        ssp_async_lambda_traces.append(ltrace_a); ssp_async_pols.append(pola)

    # aggregates
    rvi_mean = np.mean(np.array(rvi_f_traces), axis=0)
    rvi_std = np.std(np.array(rvi_f_traces), axis=0)
    ssp_sync_mean = np.mean(np.array(ssp_sync_lambda_traces), axis=0)
    ssp_sync_std = np.std(np.array(ssp_sync_lambda_traces), axis=0)
    ssp_async_mean = np.mean(np.array(ssp_async_lambda_traces), axis=0)
    ssp_async_std = np.std(np.array(ssp_async_lambda_traces), axis=0)

    # plots
    plt.figure(figsize=(8,4))
    iters = np.arange(1, max_iter_sync+1)
    plt.plot(iters, rvi_mean, label='RVI-Q (mean f(Q))')
    plt.fill_between(iters, rvi_mean-rvi_std, rvi_mean+rvi_std, alpha=0.2)
    plt.hlines(rho_true, 1, max_iter_sync, linestyles='dashed', label='DP rho')
    plt.title("Synchronous RVI-Q-learning: f(Q) (mean ± std) vs DP rho")
    plt.xlabel("iteration")
    plt.ylabel("f(Q) = min_b Q(i0,b)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(iters, ssp_sync_mean, label='SSP-Q sync (mean lambda)')
    plt.fill_between(iters, ssp_sync_mean-ssp_sync_std, ssp_sync_mean+ssp_sync_std, alpha=0.2)
    plt.hlines(rho_true, 1, max_iter_sync, linestyles='dashed', label='DP rho')
    plt.title("Synchronous SSP-Q-learning: lambda (mean ± std) vs DP rho")
    plt.xlabel("iteration")
    plt.ylabel("lambda")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,4))
    iters_a = np.arange(1, max_iter_async+1)
    plt.plot(iters_a, ssp_async_mean, label='SSP-Q async (mean lambda)')
    plt.fill_between(iters_a, ssp_async_mean-ssp_async_std, ssp_async_mean+ssp_async_std, alpha=0.2)
    plt.hlines(rho_true, 1, max_iter_async, linestyles='dashed', label='DP rho')
    plt.title("Asynchronous SSP-Q-learning: lambda (mean ± std) vs DP rho")
    plt.xlabel("iteration")
    plt.ylabel("lambda")
    plt.legend()
    plt.grid(True)
    plt.show()

    # summary table
    rows = []
    for run in range(n_runs):
        rows.append({
            'run': run,
            'RVI_final_policy': str(rvi_pols[run]),
            'RVI_f_last': float(rvi_f_traces[run][-1]),
            'SSP_sync_final_policy': str(ssp_sync_pols[run]),
            'SSP_sync_lambda_last': float(ssp_sync_lambda_traces[run][-1]),
            'SSP_async_final_policy': str(ssp_async_pols[run]),
            'SSP_async_lambda_last': float(ssp_async_lambda_traces[run][-1])
        })
    df = pd.DataFrame(rows)
    print("\nSummary of runs (first rows):")
    print(df.head())
    print("\nDP ground truth:")
    print({'rho':rho_true, 'V':V_true, 'policy':policy_true})

# Main
if __name__ == "__main__":
    run_experiments(n_runs=6, max_iter_sync=5000, max_iter_async=20000)

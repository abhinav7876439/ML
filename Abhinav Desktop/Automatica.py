# borkar_rvi_whittle.py
import numpy as np
import random
import math
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

# ------------------ Problem specification (Fu et al. example) ------------------
nS = 4
nA = 2
P0 = np.array([
    [0.5, 0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5, 0.5]
])
P1 = P0.T
# rewards (action-independent here, but keep r(i,u) notation in code)
r_state = np.array([-1.0, 0.0, 0.0, 1.0])
true_indices = np.array([-0.5, 0.5, 1.0, -1.0])

# Population
N = 100
M = 20

# Scheduling / stepsizes (practical version of (17))
T = 50           # fast steps per slow 'outer' iteration (only for plotting convenience)
n_slow = 400     # -> total fast steps = n_slow * T
alpha_C = 1.0    # for a(n) = C / ceil(n/500)
Cprime = 0.05    # for b(n)
eps = 0.10       # exploration prob
min_eps = 0.01

# --- Helpers
def f_of_Q(Q_one_hat):
    """RVI normalization f(Q) = (1/(2d)) sum_i (Q(i,0)+Q(i,1))"""
    d = Q_one_hat.shape[0]
    return 0.5 / d * np.sum(Q_one_hat)

def a_of_nu(localcount):
    """a(nu) = alpha_C / ceil(nu/500), nu=1,2,...  (large for first few visits)"""
    if localcount <= 0:
        return 1.0
    return alpha_C / math.ceil(localcount / 500)

def b_of_global(global_step):
    """b(n) = C' / (1 + ceil(n log n / 500)) applied sparsely; ensures b = o(a)."""
    if global_step <= 0:
        return 0.0
    denom = 1 + math.ceil((global_step * math.log(max(2, global_step))) / 500.0)
    return Cprime / denom

def step_one(state, action):
    probs = P0[state] if action == 0 else P1[state]
    next_state = np.random.choice(nS, p=probs)
    # action-independent reward here:
    return next_state, r_state[state]

def select_active(states, lambda_per_state, M, eps):
    if random.random() < eps:
        return np.random.choice(len(states), size=M, replace=False)
    idx = lambda_per_state[states]
    return np.argsort(-idx)[:M]

def oracle_active(states):
    idx = true_indices[states]
    return np.argsort(-idx)[:M]

# --- Variables
# Separate Q per hat-state: Q_k[hat, s, a]
Q_k = np.zeros((nS, nS, nA))
for hat in range(nS):
    for s in range(nS):
        for a in range(nA):
            Q_k[hat, s, a] = r_state[s]  # initialization Q(i,u) = r(i,u)

lambda_k = np.zeros(nS)              # Whittle index estimates per hat-state
nu = np.zeros((nS, nA), dtype=int)   # local clocks ν(i,u) shared (homogeneous arms)

arm_states = np.random.choice(nS, size=N)

# Diagnostics
total_steps = n_slow * T
lambda_history = np.zeros((total_steps, nS))
reward_history = np.zeros(total_steps)
reward_oracle = np.zeros(total_steps)

step_idx = 0
for slow_iter in range(n_slow):
    # For selection we use lambda_per_state[s] = lambda_k[s]
    lambda_per_state = lambda_k.copy()

    for t in range(T):
        active_idx = select_active(arm_states, lambda_per_state, M, eps)
        oracle_idx = oracle_active(arm_states)

        transitions = []
        total_rew = 0.0

        # --- Simulate one step for all arms (shared memory)
        active_set = set(active_idx.tolist())
        pre_states = arm_states.copy()  # keep pre-step for oracle accounting
        for i in range(N):
            s = pre_states[i]
            a = 1 if i in active_set else 0
            s_next, rew = step_one(s, a)
            transitions.append((s, a, s_next, rew))
            arm_states[i] = s_next
            total_rew += rew

        # instantaneous rewards (per-arm averages)
        reward_history[step_idx] = total_rew / N
        reward_oracle[step_idx] = np.mean(r_state[pre_states])  # oracle's immediate reward baseline

        # log lambdas
        lambda_history[step_idx, :] = lambda_k.copy()

        # --- FAST timescale: RVI-Q updates per hat-state
        # precompute f(Q_k[hat]) for each hat
        f_vals = np.array([f_of_Q(Q_k[hat]) for hat in range(nS)])

        for (s, a, s_next, rew) in transitions:
            nu[s, a] += 1
            aval = a_of_nu(nu[s, a])

            # Indicator I{X_n=s, U_n=a} is implemented by ONLY updating Q[s,a] just visited
            for hat in range(nS):
                # (1-u)(r(i,0)+lambda_hat) + u*r(i,1)  (here r(i,0)=r(i,1)=r_state[i])
                h_term = (1 - a) * (r_state[s] + lambda_k[hat]) + a * (r_state[s])
                max_next = max(Q_k[hat, s_next, 0], Q_k[hat, s_next, 1])
                # RVI increment:
                delta = h_term + max_next - f_vals[hat] - Q_k[hat, s, a]
                Q_k[hat, s, a] += aval * delta

        step_idx += 1
        eps = max(min_eps, eps * (1 - step_idx / (total_steps + 1)))

        # --- SLOWER timescale: λ updates (applied sparsely so b=o(a))
        if step_idx % N == 0:
            b = b_of_global(step_idx)
            for hat in range(nS):
                # λ_{n+1} = λ_n + b(n) * (Q(hat,1;hat) - Q(hat,0;hat))
                lambda_k[hat] += b * (Q_k[hat, hat, 1] - Q_k[hat, hat, 0])
            lambda_per_state = lambda_k.copy()

# --- Plots
steps = np.arange(total_steps)
plt.figure(figsize=(10,4))
for s in range(nS):
    plt.plot(steps, lambda_history[:, s], label=f'state {s+1}')
    plt.hlines(true_indices[s], 0, total_steps-1, linestyles='dashed', linewidth=1)
plt.title('Estimated λ_hat vs time (dashed = true Whittle indices)')
plt.xlabel('fast step')
plt.ylabel('λ estimate')
plt.legend(); plt.grid(True); plt.show()

window = 50
avg_agent = np.convolve(reward_history, np.ones(window)/window, mode='valid')
avg_oracle = np.convolve(reward_oracle, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,3))
plt.plot(np.arange(len(avg_agent)), avg_agent, label='learned-index agent')
plt.plot(np.arange(len(avg_oracle)), avg_oracle, label='oracle (true Whittle)')
plt.title(f'Running avg reward per arm (window={window})')
plt.xlabel('fast-step (windowed)'); plt.ylabel('avg reward'); plt.legend(); plt.grid(True); plt.show()

print("Final λ_hat (states 1..4):", lambda_k)
print("True indices             :", true_indices)

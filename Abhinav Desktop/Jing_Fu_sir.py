# average_cost_two_timescale_whittle.py
import numpy as np
import random
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
r = np.array([-1.0, 0.0, 0.0, 1.0])          # immediate reward depends only on state
true_indices = np.array([-0.5, 0.5, 1.0, -1.0])

# population
N = 100
M = 20

# algorithm hyperparameters
T = 50             # slow step interval (fast steps per slow update)
n_slow = 400       # number of slow iterations -> total fast steps = n_slow * T
alpha0 = 1.0        # Q-learning numerator (alpha = alpha0 / N_sa)
beta0 = 0.05        # lambda slow-step numerator
eps = 0.10          # exploration prob
eta_g = 0.001       # step-size for average reward g (very small -> slow)
min_epsilon = 0.01

# Initialize Q(s,a) as immediate reward r[s], per your instruction
Q = np.zeros((nS, nA))
for s in range(nS):
    for a in range(nA):
        Q[s, a] = r[s]

lambda_est = np.zeros(nS)            # lambda estimates initialized to zero
N_sa = np.zeros((nS, nA), dtype=int) # visit counts aggregated across arms

# initial states for all arms (can also initialize all to state 0)
arm_states = np.random.choice(nS, size=N)

# diagnostics
total_steps = n_slow * T
reward_history = np.zeros(total_steps)
reward_oracle = np.zeros(total_steps)
lambda_history = np.zeros((total_steps, nS))
g_history = np.zeros(total_steps)    # average-reward estimate trace

# helper functions
def select_arms_to_activate(states, lambda_est, M, eps):
    Nloc = len(states)
    if random.random() < eps:
        return np.random.choice(Nloc, size=M, replace=False)
    arm_indices = lambda_est[states]
    # pick top-M arms by index
    return np.argsort(-arm_indices)[:M]

def step_one(state, action):
    if action == 0:
        probs = P0[state]
    else:
        probs = P1[state]
    next_state = np.random.choice(nS, p=probs)
    reward = r[state]
    return next_state, reward

def oracle_select(states):
    arm_indices = true_indices[states]
    return np.argsort(-arm_indices)[:M]

# initialize global average reward estimate g (per-arm)
g = 0.0

# main loop
step_idx = 0
for slow_iter in range(n_slow):
    for t in range(T):
        # pick active arms (learning agent) and oracle active set
        active_arm_idxs = select_arms_to_activate(arm_states, lambda_est, M, eps)
        oracle_active = oracle_select(arm_states)

        total_reward = 0.0
        total_reward_oracle = 0.0
        transitions = []

        # collect transitions for all N arms (shared-memory architecture)
        # we use simple synchronous update: sample next state and reward for each arm
        for i in range(N):
            a = 1 if i in active_arm_idxs else 0
            s = arm_states[i]
            s_next, rew = step_one(s, a)
            transitions.append((s, a, s_next, rew))
            arm_states[i] = s_next
            total_reward += rew if (a == 1 or a == 0) else rew  # same either way since reward depends only on state

        # compute oracle reward (oracle uses true indices -- purely greedy no exploration)
        for i in range(N):
            a_or = 1 if i in oracle_active else 0
            s_or = arm_states[i]  # note: after step above we updated arm_states; this is consistent baseline per-step
            total_reward_oracle += r[s_or]

        # store diagnostics (per-arm average reward at this step)
        reward_history[step_idx] = total_reward / N
        reward_oracle[step_idx] = total_reward_oracle / N
        lambda_history[step_idx, :] = lambda_est.copy()
        g_history[step_idx] = g

        # FAST timescale: update Q using average-cost TD target:
        # target = reward - g + min_v Q[next_state, v]
        for (s, a, s_next, rew) in transitions:
            N_sa[s, a] += 1
            alpha = alpha0 / (N_sa[s, a])
            target = rew - g + np.min(Q[s_next, :])   # cost-minimization style (min); here reward so we keep as reward - g + min
            # note: if you prefer reward-maximization, use max and adapt signs consistently
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * target

        # update global average reward estimate g on a very slow timescale
        # we take observed total_reward/N as the per-arm instantaneous reward
        r_instant = total_reward / N
        g = g + eta_g * (r_instant - g)

        step_idx += 1

        # epsilon decay (optional)
        eps = max(min_epsilon, eps * (1 - step_idx / (total_steps + 1)))

    # SLOW timescale: update lambda estimates every T fast steps toward Q[s,1] - Q[s,0]
    beta = beta0 / (1 + 0.001 * slow_iter)    # slowly decaying beta
    for s in range(nS):
        idx_est = Q[s, 1] - Q[s, 0]
        lambda_est[s] = (1 - beta) * lambda_est[s] + beta * idx_est

# Post-processing & plotting
steps = np.arange(total_steps)

plt.figure(figsize=(10, 4))
for s in range(nS):
    plt.plot(steps, lambda_history[:, s], label=f'state {s+1}')
    plt.hlines(true_indices[s], 0, total_steps-1, linestyles='dashed', linewidth=1)
plt.title('Estimated lambda vs time (dashed lines = true indices)')
plt.xlabel('fast-step')
plt.ylabel('lambda estimate')
plt.legend()
plt.grid(True)
plt.show()

window = 50
avg_agent = np.convolve(reward_history, np.ones(window) / window, mode='valid')
avg_oracle = np.convolve(reward_oracle, np.ones(window) / window, mode='valid')
plt.figure(figsize=(8,3))
plt.plot(np.arange(len(avg_agent)), avg_agent, label='learned-index agent')
plt.plot(np.arange(len(avg_oracle)), avg_oracle, label='oracle (true Whittle)')
plt.title(f'Running average reward (window={window})')
plt.xlabel('fast-step (windowed)')
plt.ylabel('avg reward per arm')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,3))
plt.plot(g_history)
plt.title('Estimated global average reward g over time')
plt.xlabel('fast-step')
plt.ylabel('g')
plt.grid(True)
plt.show()

print('Final lambda estimates (per state 1..4):', lambda_est)
print('True indices                :', true_indices)
print('Final average reward g      :', g)

# borkar_whittle_rvi_qlearning.py
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# reproducibility
np.random.seed(0)
random.seed(0)

# ---------------- Problem: Fu et al. circulant example ----------------
nS = 4                  # states 0..3 (corresponds to 1..4 in paper)
nA = 2                  # actions 0 (passive) / 1 (active)
P0 = np.array([
    [0.5, 0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5, 0.5]
])
P1 = P0.T
r_state = np.array([-1.0, 0.0, 0.0, 1.0])   # immediate reward depends only on state
true_indices = np.array([-0.5, 0.5, 1.0, -1.0])

# population & control
N = 100         # total arms
M = 20          # number to activate at each time

# algorithm hyperparameters (practical choices following paper)
T = 50                  # fast steps per slow update
n_slow = 400            # number of slow iterations -> total steps = n_slow * T
alpha_C = 1.0           # constant C in a(n) = C / ceil(n/500)
Cprime = 0.05           # constant for b(n)
eps = 0.10              # exploration prob (uniform randomization)
min_eps = 0.01

# RVI parameters: we will use RVI-style f(Q) = (1/(2d)) sum_{i} (Q(i,0)+Q(i,1))
def f_of_Q(Q):
    d = Q.shape[0]
    return 0.5 / d * np.sum(Q)   # sum over both actions and states

# Initialize Q_k and lambda_k for each hat_k in S
# For each hat_k we maintain Q_k (shape nS x nA) and lambda_k (scalar)
Q_k = np.zeros((nS, nS, nA))   # Q_k[hat_k, s, a]
lambda_k = np.zeros(nS)        # lambda_k[hat_k]

# Initialize each Q_k(i,u) = r(i) as per the excerpt
for hat in range(nS):
    for s in range(nS):
        for a in range(nA):
            Q_k[hat, s, a] = r_state[s]

# local clocks ν(i,u) (counts of visits/updates for each state-action pair) (shared)
nu = np.zeros((nS, nA), dtype=int)

# Initialize arm states (random or all zeros)
arm_states = np.random.choice(nS, size=N)

# diagnostics
total_steps = n_slow * T
lambda_history = np.zeros((total_steps, nS))
reward_history = np.zeros(total_steps)
reward_oracle = np.zeros(total_steps)
step_idx = 0

# helper: step dynamics
def sample_next(state, action):
    probs = P0[state] if action == 0 else P1[state]
    next_state = np.random.choice(nS, p=probs)
    reward = r_state[state]
    return next_state, reward

# helper: selection using current lambda estimates per state (lambda per state is lambda_k[state])
def select_active(arm_states, lambda_k_per_state, M, eps):
    if random.random() < eps:
        return np.random.choice(len(arm_states), size=M, replace=False)
    # compute index per arm by mapping its current state to that state's lambda estimate
    arm_indices = lambda_k_per_state[arm_states]
    return np.argsort(-arm_indices)[:M]

# oracle (uses true indices to select top-M)
def oracle_active(arm_states, M):
    arm_indices = true_indices[arm_states]
    return np.argsort(-arm_indices)[:M]

# step-size "a" dependent on localclock ν as in paper (practical discrete form)
def a_of_nu(localcount):
    # a(n) = C / ceil(n / 500)
    if localcount <= 0:
        return 1.0  # initial large step if never visited; practical choice
    return alpha_C / math.ceil(localcount / 500)

# b(n) schedule per paper: b(n) = C' / (1 + ceil(n log n / 500)) * I{n (mod N) == 0}
def b_of_global(global_step):
    if global_step <= 0:
        return 0.0
    denom = 1 + math.ceil((global_step * math.log(max(2, global_step))) / 500.0)
    return Cprime / denom

# main loop
for slow in range(n_slow):
    # lambda_k per state for selection is lambda_k[state] (hat_k corresponding to that state)
    lambda_per_state = lambda_k.copy()

    for t in range(T):
        # choose active arms by ranking lambda_per_state at each arm's state (with prob 1-eps)
        active_idx = select_active(arm_states, lambda_per_state, M, eps)
        oracle_idx = oracle_active(arm_states, M)

        # collect transitions
        transitions = []
        total_rew = 0.0

        # simulate one step for every arm (this is synchronous sampling)
        # note: classification whether arm active depends on membership in active_idx
        active_set = set(active_idx.tolist())
        for i in range(N):
            s = arm_states[i]
            a = 1 if i in active_set else 0
            s_next, rew = sample_next(s, a)
            transitions.append((s, a, s_next, rew))
            arm_states[i] = s_next
            total_rew += rew

        # oracle reward: using oracle_idx but we compute reward after transition (consistent baseline)
        # for fairness we use same post-step states (arm_states updated), but oracle picks using pre-step states.
        # compute per-arm oracle reward by summing r_state at arm's pre-step state if it was selected by oracle
        # To be faithful to description, compute oracle instantaneous reward using pre-transition states:
        oracle_reward = 0.0
        # reconstruct pre-step states by stepping transitions backwards isn't trivial; instead, compute as:
        # use the fact we recorded transitions list in same order -> pre-step state is transitions[i][0]
        for i in range(N):
            s_pre = transitions[i][0]
            a_or = 1 if i in oracle_idx else 0
            # oracle_reward uses reward of pre-step state (since reward depends only on state)
            oracle_reward += r_state[s_pre]

        reward_history[step_idx] = total_rew / N
        reward_oracle[step_idx] = oracle_reward / N
        lambda_history[step_idx, :] = lambda_k.copy()

        # FAST timescale updates: for each hat_k, update Q_k according to (14)/(20) with RVI f(Q_k)
        # compute f(Q_k) for each hat_k
        f_vals = np.zeros(nS)
        for hat in range(nS):
            f_vals[hat] = f_of_Q(Q_k[hat])

        # Apply updates per observed (s,a) across arms:
        for (s, a, s_next, rew) in transitions:
            # increment local clock for this (s,a)
            nu[s, a] += 1
            aval = a_of_nu(nu[s, a])

            # For each hat_k, compute the increment
            for hat in range(nS):
                # compute h part: (1-u)(r(i,0)+lambda_hat) + u r(i,1)
                h_term = (1 - a) * (r_state[s] + lambda_k[hat]) + a * (r_state[s])
                # plus expected next state contribution is replaced by sample max_v Qn(X_{n+1}, v; hat)
                max_next = max(Q_k[hat, s_next, 0], Q_k[hat, s_next, 1])
                # RVI-TD target minus current Q:
                delta = h_term + max_next - f_vals[hat] - Q_k[hat, s, a]
                # stochastic update with stepsize a(nu)
                Q_k[hat, s, a] += aval * delta

        # SLOW global counter step increment
        step_idx += 1

        # optional: slowly decay exploration
        eps = max(min_eps, eps * (1 - step_idx / (total_steps + 1)))

        # Update lambda_k on a slower timescale b(n): do it only when step_idx % N == 0 as in their schedule
        # Use b(n) only at special global steps (makes b = o(a) naturally)
        if step_idx % N == 0:
            b = b_of_global(step_idx)
            for hat in range(nS):
                # update lambda for hat_k using Q_k(hat_k,1) - Q_k(hat_k,0)
                delta_lambda = Q_k[hat, hat, 1] - Q_k[hat, hat, 0]
                lambda_k[hat] += b * delta_lambda
            # recompute lambda_per_state after an update if needed
            lambda_per_state = lambda_k.copy()

    # end of T fast steps (one slow iteration)
    # (optionally you could also do an extra small slow-step here - but we already did updates at step_idx % N times)

# Postprocessing: plots and summaries
steps = np.arange(total_steps)

plt.figure(figsize=(10,4))
for s in range(nS):
    plt.plot(steps, lambda_history[:, s], label=f'state {s+1}')
    plt.hlines(true_indices[s], 0, total_steps-1, linestyles='dashed', linewidth=1)
plt.title('Estimated lambda_k (hat_k index estimator) vs time\n(dashed lines = true Whittle indices)')
plt.xlabel('fast-step')
plt.ylabel('lambda estimate')
plt.legend()
plt.grid(True)
plt.show()

window = 50
avg_agent = np.convolve(reward_history, np.ones(window)/window, mode='valid')
avg_oracle = np.convolve(reward_oracle, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,3))
plt.plot(np.arange(len(avg_agent)), avg_agent, label='learned-index agent')
plt.plot(np.arange(len(avg_oracle)), avg_oracle, label='oracle (true Whittle)')
plt.title(f'Running average reward per arm (window={window})')
plt.xlabel('fast-step (windowed)')
plt.ylabel('avg reward per arm')
plt.legend()
plt.grid(True)
plt.show()

print("Final lambda_k (for hat_k = 1..4):", lambda_k)
print("True indices                :", true_indices)
print("Final Q_k for each hat_k (showing Q_k[hat, s, :]):")
for hat in range(nS):
    print(f"hat={hat+1}:")
    for s in range(nS):
        print(f"  state {s+1}: Q = {Q_k[hat, s, 0]:.4f}, {Q_k[hat, s, 1]:.4f}")


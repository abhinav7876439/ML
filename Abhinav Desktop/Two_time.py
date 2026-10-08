"""
Two-time-scale Q-learning (Python script)

This script implements:
 1) Two-time-scale asynchronous Q-learning (slow policy update every T fast steps)
 2) Synchronous Q-learning (simulated per-(s,a) updates)
 3) Deterministic Q-learning (deterministic transition function)

Run as a script (or import functions in your notebook).
"""

import numpy as np
import random
import math
import argparse
import matplotlib.pyplot as plt

# ------------------------ Environment classes ------------------------
class SimpleMDP:
    """Small illustrative MDP for cost-minimization.
    States: 0..nS-1, Actions: 0..nA-1
    Transition probabilities P[s,a] (nS vector) and cost[s,a] are provided.
    """
    def __init__(self, nS=5, nA=2):
        self.nS = nS
        self.nA = nA
        self.P = np.zeros((nS, nA, nS))
        # simple dynamics: action 0 tends left, action 1 tends right
        for s in range(nS):
            for a in range(nA):
                left = max(0, s-1)
                stay = s
                right = min(nS-1, s+1)
                if a == 0:
                    self.P[s,a,left] += 0.6      ## (Toy Example Action 0: Move Left with 60% probability)
                    self.P[s,a,stay] += 0.3
                    self.P[s,a,right] += 0.1
                else:
                    self.P[s,a,right] += 0.6
                    self.P[s,a,stay] += 0.3
                    self.P[s,a,left] += 0.1
        # costs: higher near terminal state (nS-1)
        self.cost = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                dist = abs((nS-1) - s)   ## distance
                self.cost[s,a] = 1.0 + 0.25 * dist + (-0.1 if (a==1 and 0< s < nS-1) else 0.0)
        # Make terminal absorbing with larger cost
        for a in range(nA):
            self.P[nS-1,a,:] = 0.0
            self.P[nS-1,a,nS-1] = 1.0
            self.cost[nS-1,a] = 5.0

    def step(self, s, a):
        probs = self.P[s,a]
        next_s = np.random.choice(self.nS, p=probs)
        c = self.cost[s,a]
        return next_s, c

    def sample_next(self, s, a):
        """Return a random next-state and cost (same as step) - helper for synchronous updates"""
        return self.step(s,a)


class DeterministicMDP:
    """Deterministic MDP with f(s,a) deterministic transition and costs."""
    def __init__(self, nS=5, nA=2):
        self.nS = nS
        self.nA = nA
        self.f = {}    ## f is a dictionary that stores the state transition function for the deterministic MDP
        self.cost = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                if a == 0:
                    next_s = max(0, s-1)
                else:
                    next_s = min(nS-1, s+1)
                self.f[(s,a)] = next_s
                self.cost[s,a] = 1.0 + 0.5*abs((nS-1)/2 - s) + (0 if a==1 else 0.1)

    def step(self, s, a):
        next_s = self.f[(s,a)]
        c = self.cost[s,a]
        return next_s, c


# ------------------------ Algorithms ------------------------
def two_timescale_q_learning(env, n_slow=100, T=50, alpha0=1.0, epsilon0=0.2, min_epsilon=0.01, start_state=0):
    """Two-time-scale asynchronous Q-learning.
    - Slow timescale: policy update every T steps -> greedy wrt current Q (cost minimization: argmin).
    - Fast timescale: T steps using current policy (with epsilon-greedy exploration) and update Q per observed transition.
    - Step-size per (s,a): alpha = alpha0 / N_sa (Robbins-Monro type).
    Returns Q, final_policy, cost_history, visit_counts
    """
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    N_sa = np.zeros((nS, nA), dtype=int)  ## counts how many times each state-action pair (s,a) has been updated
    policy = np.argmin(Q, axis=1)
    cost_history = []    ## A list that stores the instantaneous costs encountered during the Q-learning updates (to track convergence)
    s = start_state
    epsilon = epsilon0     ## epsilon0: Initial exploration probability for ε-greedy
    total_steps = n_slow * T
    step_idx = 0
    for n in range(n_slow):
        # slow update: greedy policy    ### Slow timescale: policy update every T steps
        policy = np.argmin(Q, axis=1)

        # fast updates: T steps of epsilon-greedy using current policy
        for k in range(T): ## Fast timescale: Q-value updates (at every step using current policy)
            if random.random() < epsilon:
                a = random.randrange(nA)     # explore
            else:
                a = int(policy[s])           # exploit
            next_s, c = env.step(s, a)
            N_sa[s,a] += 1
            alpha = alpha0 / (N_sa[s,a])     ### alpha0: Base step-size constant for Q-learning
            target = c + np.min(Q[next_s])
            Q[s,a] = (1 - alpha) * Q[s,a] + alpha * target
            cost_history.append(c)
            s = next_s
            step_idx += 1
            if s == env.nS - 1:   ## If s is terminal i.e.(env.nS - 1), reset to start_state
                s = start_state
            epsilon = max(min_epsilon, epsilon0 * (1 - step_idx / (total_steps + 1)))   ## Decay epsilon linearly
    final_policy = np.argmin(Q, axis=1)
    return Q, final_policy, cost_history, N_sa


def synchronous_q_learning(env, iterations=500, alpha0=0.1):
    """Synchronous Q-learning: update every (s,a) per iteration with one sampled next-state."""
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    N_sa = np.zeros((nS, nA), dtype=int)    ## counts how many times each state-action pair (s,a) has been updated.
    ## Each time the algorithm samples a transition (s,a) → (next_s,c), it appends c (the cost) to cost_trace
    cost_trace = []       ## A list that stores the instantaneous costs encountered during the Q-learning updates
    for k in range(iterations):
        for s in range(nS):
            for a in range(nA):
                next_s, c = env.sample_next(s,a)
                N_sa[s,a] += 1
                # simple decaying alpha schedule (can be replaced with alpha0 / N_sa)
                alpha = alpha0 / (1 + 0.01 * k)  ## Here, we can have alpha(s,a) = 1 / N_sa[s,a] as well
                target = c + np.min(Q[next_s])
                Q[s,a] = (1 - alpha) * Q[s,a] + alpha * target
                cost_trace.append(c)
    policy = np.argmin(Q, axis=1)
    return Q, policy, cost_trace, N_sa


def deterministic_q_learning(env, iterations=200):
    """Deterministic Q-learning update along a trajectory.
       Q_{k+1}(x_k,u_k) = c(x_k,u_k) + min_v Q(f(x_k,u_k), v)
    """
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    s = 0
    for k in range(iterations):
        a = int(np.argmin(Q[s]))
        next_s, c = env.step(s, a)
        Q[s,a] = c + np.min(Q[next_s])
        s = next_s
    policy = np.argmin(Q, axis=1)
    return Q, policy


# ------------------------ Utility / main ------------------------
def print_table(Q):
    print(np.round(Q, 3))


def main():
    parser = argparse.ArgumentParser(description='Two-time-scale Q-learning examples')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', choices=['two','sync','det'], default='two', help='which algorithm to run')
    parser.add_argument('--nslow', type=int, default=150)
    parser.add_argument('--T', type=int, default=60)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.run == 'two':
        env = SimpleMDP(nS=5, nA=2)
        Q, policy, costs, N = two_timescale_q_learning(env, n_slow=args.nslow, T=args.T, alpha0=1.0, epsilon0=0.25)
        print('Two-time-scale Q-learning result:')
        print('Q:')
        print_table(Q)
        print('policy:', policy)
        print('visit counts:')
        print(N)
        if len(costs) > 0:
            window = min(50, len(costs))
            running_avg = np.convolve(costs, np.ones(window)/window, mode='valid')
            plt.plot(running_avg)
            plt.title('Running average immediate cost')
            plt.xlabel('fast-step (windowed)')
            plt.ylabel('avg cost')
            plt.show()

    elif args.run == 'sync':
        env = SimpleMDP(nS=5, nA=2)
        Q, policy, costs, N = synchronous_q_learning(env, iterations=300, alpha0=0.2)
        print('Synchronous Q-learning result:')
        print('Q:')
        print_table(Q)
        print('policy:', policy)

    else:
        env = DeterministicMDP(nS=5, nA=2)
        Q, policy = deterministic_q_learning(env, iterations=200)
        print('Deterministic Q-learning result:')
        print('Q:')
        print_table(Q)
        print('policy:', policy)


if __name__ == '__main__':
    main()



##################################### Below is the implementation of the two-time-scale Whittle-index learning algorithm



"""
Two-time-scale Whittle-index learning for (Jing Fu et al.) (2019) example

This script implements a two-timescale algorithm to estimate Whittle indices for a homogeneous
population of restless bandits (N identical arms). The example is taken from Fu et al. (2019):
- 4 states (1..4). Transition matrices P0 (passive) and P1 (active) are circulant; P1 = P0^T.
- Rewards r(1)=-1, r(2)=0, r(3)=0, r(4)=1 (action-independent).

Algorithm (practical implementation / heuristic):
- Maintain Q(s,a) estimates for s in {1..4}, a in {0,1} (discounted reward setting).
- Fast timescale: simulate N arms; at each time choose M arms to activate by ranking current
  index estimates; with probability eps do random exploration.
  Update Q(s,a) using TD(0)-style updates with step-size alpha = alpha0 / N_sa(s,a).
- Slow timescale: every T steps update lambda estimates for each state via a slow Robbins-Monro
  step towards the difference Q(s,1)-Q(s,0), which approximates the subsidy that equalizes
  action values (i.e., the Whittle index heuristic we use here).

Notes:
- This is an implementable, practical algorithm (heuristic) intended to reproduce the experiment
  behaviour (convergence of estimated indices and average reward) reported in the paper.
- We use discounted returns (gamma < 1) for numerical stability and because it is simpler to
  implement in a short script. The restless-bandit theory in Fu et al. is average-cost based;
  the qualitative behaviour (ordinal ranking of indices) is what matters for resource allocation.

Run the script to see: (1) convergence of lambda estimates towards known true indices
[-0.5, 0.5, 1.0, -1.0], and (2) running-average reward compared to the oracle policy that uses
exact Whittle indices from the beginning.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

# ------------------ Problem specification (Fu et al. example) ------------------
# States are 0..3 corresponding to 1..4 in the text
nS = 4
nA = 2
# Passive transition matrix P0 (rows: from-state, cols: to-state)  ## Circulant Dynamics
P0 = np.array([
    [0.5, 0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5, 0.5]
])
# Active transition P1 = P0^T
P1 = P0.T
# rewards (action-independent) for states 0..3 (corresponds to 1..4)
r = np.array([-1.0, 0.0, 0.0, 1.0])
# True Whittle indices reported in paper for states 1..4
true_indices = np.array([-0.5, 0.5, 1.0, -1.0])

# Restless bandit population
N = 100  # number of identical arms
M = 20   # number active per time-step

# two-timescale algorithm hyperparameters
T = 50               # slow update interval (every T fast steps we update lambda)
n_slow = 400         # number of slow iterations -> total fast steps = n_slow * T
alpha0 = 1.0         # numerator for Q-learning step-size alpha = alpha0 / N_sa
beta0 = 0.05         # slow-timescale step-size (can be decreased over time)
eps = 0.10           # exploration probability
gamma = 0.99         # discount factor

# Initialize Q(s,a) to immediate reward r(s) (per instructions)
# We create Q as shape (nS, nA); Q[:, :] = r[:, None]
Q = np.zeros((nS, nA))
for s in range(nS):
    for a in range(nA):
        Q[s,a] = r[s]

# initialize lambda estimates (Whittle indices) to zero
lambda_est = np.zeros(nS)

# visit counts for (s,a) aggregated across all arms (shared-memory architecture)
N_sa = np.zeros((nS, nA), dtype=int)

# initialize states for all N arms uniformly at random among states (or all in state 0)
arm_states = np.random.choice(nS, size=N)

# diagnostics
total_steps = n_slow * T
reward_history = np.zeros(total_steps)
lambda_history = np.zeros((total_steps, nS))
# running average reward for oracle (exact Whittle indices) baseline
reward_oracle = np.zeros(total_steps)

# helper: pick M arms to activate using indices; if exploration, pick random M
def select_arms_to_activate(states, lambda_est, M, eps):
    N = len(states)
    if random.random() < eps:
        return np.random.choice(N, size=M, replace=False)
    # compute index per arm by mapping lambda_est[state]
    arm_indices = lambda_est[states]
    # pick top M arms by index (ties broken arbitrarily)
    return np.argsort(-arm_indices)[:M]

# Function to apply action to a state and sample next state and reward
def step_one(state, action):
    if action == 0:
        probs = P0[state]
    else:
        probs = P1[state]
    next_state = np.random.choice(nS, p=probs)
    reward = r[state]
    return next_state, reward

# oracle policy reward: always use true indices to pick top-M arms (no exploration)
def oracle_select(states):
    arm_indices = true_indices[states]
    return np.argsort(-arm_indices)[:M]

# main loop (fast steps)
step_idx = 0
for slow_iter in range(n_slow):
    # during this slow iteration we keep using current lambda_est for ranking
    for t in range(T):
        # choose which arms to activate this step (shared lambda ranking + epsilon exploration)
        active_arm_idxs = select_arms_to_activate(arm_states, lambda_est, M, eps)
        # baseline oracle active set for comparison (no exploration)
        oracle_active = oracle_select(arm_states)

        total_reward = 0.0
        total_reward_oracle = 0.0
        # We'll gather transitions for all arms and update Q (shared memory updates)
        transitions = []  # list of tuples (s, a, s_next, reward)
        for i in range(N):
            a = 1 if i in active_arm_idxs else 0
            s = arm_states[i]
            s_next, rew = step_one(s, a)
            transitions.append((s, a, s_next, rew))
            arm_states[i] = s_next
            total_reward += rew if a==1 or a==0 else rew
        # oracle reward
        for i in range(N):
            a_or = 1 if i in oracle_active else 0
            total_reward_oracle += r[arm_states[i]] if a_or==1 or a_or==0 else r[arm_states[i]]

        reward_history[step_idx] = total_reward / N
        reward_oracle[step_idx] = total_reward_oracle / N
        lambda_history[step_idx,:] = lambda_est.copy()

        # Fast-timescale Q updates: apply TD(0)-style update for each observed (s,a)
        for (s,a,s_next,rew) in transitions:
            N_sa[s,a] += 1
            alpha = alpha0 / (N_sa[s,a])
            target = rew + gamma * np.max(Q[s_next,:])
            Q[s,a] = (1 - alpha) * Q[s,a] + alpha * target

        step_idx += 1

    # End of T fast steps: slow-timescale lambda update based on current Q
    # We update each state's lambda towards Q[s,1] - Q[s,0] (the subsidy that would make active and passive equal)
    # Use small step-size beta that may decay with slow_iter
    beta = beta0 / (1 + 0.001 * slow_iter)
    for s in range(nS):
        # estimate of index as Q(s,1) - Q(s,0)
        idx_est = Q[s,1] - Q[s,0]
        lambda_est[s] = (1 - beta) * lambda_est[s] + beta * idx_est

# Post processing: plot lambda estimates and running average reward
steps = np.arange(total_steps)
plt.figure(figsize=(10,4))
for s in range(nS):
    plt.plot(steps, lambda_history[:,s], label=f'state {s+1}')
for s in range(nS):
    plt.hlines(true_indices[s], 0, total_steps-1, linestyles='dashed', linewidth=1)
plt.title('Estimated lambda vs time (dashed lines = true indices)')
plt.xlabel('fast-step')
plt.ylabel('lambda estimate')
plt.legend()
plt.grid(True)
plt.show()

# Running average reward comparison (windowed)
window = 50
avg_agent = np.convolve(reward_history, np.ones(window)/window, mode='valid')
avg_oracle = np.convolve(reward_oracle, np.ones(window)/window, mode='valid')
plt.figure(figsize=(8,3))
plt.plot(np.arange(len(avg_agent)), avg_agent, label='learned-index agent')
plt.plot(np.arange(len(avg_oracle)), avg_oracle, label='oracle (true Whittle)')
plt.title('Running average reward (window=%d)' % window)
plt.xlabel('fast-step (windowed)')
plt.ylabel('avg reward per arm')
plt.legend()
plt.grid(True)
plt.show()

print('Final lambda estimates (per state 1..4):', lambda_est)
print('True indices                :', true_indices)

# End of script

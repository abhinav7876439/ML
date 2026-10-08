"""
qlearning.py  —  Hands-on Q-Learning on GridWorld  |  LSO 2026, IIT Delhi
========================================================================
HOW TO USE
----------
1.  Make sure gridworld.py is in the same folder.
2.  Run:  python qlearning.py
3.  Experiment by changing the parameters in the HYPERPARAMETERS section
    below.  Each parameter has a short note explaining what it controls
    and what you should expect when you change it.

EXPERIMENTS TO TRY
------------------
(A) Learning rate (alpha)
    • Try alpha = 0.01  vs  0.5  vs  0.9
    • What happens to the learning curve?  Is it smooth or noisy?

(B) Discount factor (gamma)
    • Try gamma = 0.5  vs  0.9  vs  0.99
    • Does the agent find shorter paths with higher gamma?

(C) Number of episodes
    • Try n_episodes = 100  vs  500  vs  2000
    • At what point does the policy stop improving?

(D) Max steps per episode
    • Try max_steps = 20  vs  100  vs  200
    • What happens with very short episodes in a large grid?

(E) Exploration vs Exploitation  (epsilon-greedy)
    • Try epsilon_start = 1.0, epsilon_end = 0.01 (aggressive decay)
      vs epsilon_start = 0.3, epsilon_end = 0.3   (no decay, fixed)
    • Does the agent explore enough?  Does it get stuck?

(F) Stochastic environment
    • In gridworld.py, set slip_prob = 0.2 when creating the env
    • Does Q-learning still converge?  Does it take more episodes?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gridworld import GridWorld, ACTIONS, N_ACTIONS

# =====================================================================
#  HYPERPARAMETERS  ← change these and observe the effect
# =====================================================================

ALPHA          = 0.3      # Learning rate  ∈ (0, 1]
                           # How much each new experience overwrites old estimates.
                           # Large → fast but noisy.  Small → slow but stable.

GAMMA          = 0.95     # Discount factor  ∈ [0, 1)
                           # How much the agent values future rewards.
                           # 0 = fully myopic.  Close to 1 = far-sighted.

N_EPISODES     = 600      # Total training episodes.
                           # One episode = one run from S until G or max_steps.

MAX_STEPS      = 100      # Maximum steps allowed per episode.
                           # Prevents infinite loops early in training.

EPSILON_START  = 1.0      # Initial exploration rate  ∈ [0, 1].
                           # 1.0 = fully random at the start.

EPSILON_END    = 0.05     # Final (minimum) exploration rate.
                           # The agent will always explore with this probability.

EPSILON_DECAY  = "exp"    # Decay schedule: "exp" (exponential) or "linear".
                           # Controls how quickly exploration is reduced.

EVAL_EVERY     = 50       # Evaluate the greedy policy every this many episodes.
                           # (Does not affect training — purely for plotting.)

SEED           = 42       # Random seed for reproducibility.
                           # Change to get a different run.

# =====================================================================
#  ENVIRONMENT  ← you can also tweak the environment here
# =====================================================================
#
#  Try: slip_prob=0.2 for a stochastic (slippery) grid.
#  Try: n_rows=7, n_cols=7, goal=(6,6) for a larger grid.

env = GridWorld(
    n_rows       = 5,
    n_cols       = 5,
    start        = (0, 0),
    goal         = (4, 4),
    goal_reward  = 10.0,
    step_penalty = -0.1,
    slip_prob    = 0.0,    # ← set > 0 for experiment (F)
)

# =====================================================================
#  Q-LEARNING  (do not modify below unless you want to experiment)
# =====================================================================

np.random.seed(SEED)

# ── Q-table: shape (n_states, n_actions), initialised to zero ────────
Q = np.zeros((env.n_states, env.n_actions))


def epsilon_schedule(episode):
    """Return epsilon for the given episode number."""
    if EPSILON_DECAY == "exp":
        decay = np.exp(np.log(EPSILON_END / EPSILON_START) / N_EPISODES)
        return max(EPSILON_END, EPSILON_START * (decay ** episode))
    else:  # linear
        frac = episode / N_EPISODES
        return max(EPSILON_END, EPSILON_START + frac * (EPSILON_END - EPSILON_START))


def greedy_policy(Q):
    """Return the greedy (argmax) action for every state."""
    return np.argmax(Q, axis=1)


def evaluate_policy(Q, n_eval=20):
    """
    Run n_eval greedy episodes.  Return mean total reward and mean steps
    to goal (-1 if goal not reached).
    """
    total_rewards = []
    steps_to_goal = []
    for _ in range(n_eval):
        s    = env.reset()
        done = False
        ep_r = 0.0
        for step in range(MAX_STEPS):
            a              = np.argmax(Q[s])
            s, r, done     = env.step(a)
            ep_r          += r
            if done:
                steps_to_goal.append(step + 1)
                break
        else:
            steps_to_goal.append(-1)   # did not reach goal
        total_rewards.append(ep_r)
    return np.mean(total_rewards), np.mean([x for x in steps_to_goal if x > 0] or [-1])


# ── Training loop ─────────────────────────────────────────────────────
episode_rewards  = []   # total reward per training episode
epsilons         = []   # epsilon per episode
eval_episodes    = []   # episode numbers at which we evaluated
eval_rewards     = []   # mean greedy reward at each evaluation point
eval_steps       = []   # mean steps to goal at each evaluation point

print("=" * 60)
print("  Q-Learning on GridWorld  |  LSO 2026 Hands-on Session")
print("=" * 60)
print(f"  alpha={ALPHA}  gamma={GAMMA}  episodes={N_EPISODES}")
print(f"  epsilon: {EPSILON_START} → {EPSILON_END} ({EPSILON_DECAY})")
print(f"  max_steps={MAX_STEPS}  slip_prob={env.slip_prob}")
print("=" * 60)

for ep in range(N_EPISODES):
    s       = env.reset()
    epsilon = epsilon_schedule(ep)
    ep_r    = 0.0

    for _ in range(MAX_STEPS):
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            a = np.random.randint(N_ACTIONS)        # explore
        else:
            a = np.argmax(Q[s])                     # exploit

        s_next, r, done = env.step(a)
        ep_r           += r

        # ── Q-learning update ──────────────────────────────────────
        #
        #   Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
        #
        td_target  = r + GAMMA * np.max(Q[s_next]) * (1 - done)
        td_error   = td_target - Q[s, a]
        Q[s, a]   += ALPHA * td_error
        # ──────────────────────────────────────────────────────────

        s = s_next
        if done:
            break

    episode_rewards.append(ep_r)
    epsilons.append(epsilon)

    # periodic evaluation
    if (ep + 1) % EVAL_EVERY == 0:
        mean_r, mean_steps = evaluate_policy(Q)
        eval_episodes.append(ep + 1)
        eval_rewards.append(mean_r)
        eval_steps.append(mean_steps)
        print(f"  Episode {ep+1:4d}  |  ε={epsilon:.3f}"
              f"  |  train_r={ep_r:+6.2f}"
              f"  |  greedy_r={mean_r:+6.2f}"
              f"  |  steps={mean_steps:.1f}")

print("=" * 60)
print("Training complete.")

# =====================================================================
#  PLOTS
# =====================================================================

fig = plt.figure(figsize=(14, 9))
fig.suptitle(
    f"Q-Learning on GridWorld  —  "
    f"α={ALPHA}, γ={GAMMA}, ε {EPSILON_START}→{EPSILON_END} ({EPSILON_DECAY}), "
    f"episodes={N_EPISODES}",
    fontsize=12, fontweight="bold", color="#002147"
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

GOLD  = "#C8902A"
BLUE  = "#002147"
RED   = "#C0392B"
LGRAY = "#AAAAAA"

# ── 1. Episode reward (smoothed) ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
window = max(1, N_EPISODES // 40)
smoothed = np.convolve(episode_rewards,
                       np.ones(window) / window, mode="valid")
ax1.plot(episode_rewards, color=LGRAY, lw=0.6, alpha=0.5, label="raw")
ax1.plot(range(window - 1, len(episode_rewards)),
         smoothed, color=BLUE, lw=1.8, label=f"smoothed (w={window})")
ax1.set_xlabel("Episode", color=BLUE)
ax1.set_ylabel("Total reward", color=BLUE)
ax1.set_title("Training reward per episode", color=BLUE, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── 2. Epsilon decay ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(range(N_EPISODES), epsilons, color=RED, lw=1.8)
ax2.set_xlabel("Episode", color=BLUE)
ax2.set_ylabel("ε (exploration rate)", color=BLUE)
ax2.set_title("Exploration schedule", color=BLUE, fontweight="bold")
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

# ── 3. Greedy policy reward ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(eval_episodes, eval_rewards, color=GOLD,
         marker="o", ms=5, lw=1.8)
ax3.set_xlabel("Episode", color=BLUE)
ax3.set_ylabel("Mean greedy reward", color=BLUE)
ax3.set_title("Greedy policy performance", color=BLUE, fontweight="bold")
ax3.grid(True, alpha=0.3)

# ── 4. Steps to goal ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
valid_steps = [s for s in eval_steps if s > 0]
valid_eps   = [eval_episodes[i] for i, s in enumerate(eval_steps) if s > 0]
ax4.plot(valid_eps, valid_steps, color=BLUE,
         marker="s", ms=5, lw=1.8)
ax4.set_xlabel("Episode", color=BLUE)
ax4.set_ylabel("Mean steps to goal", color=BLUE)
ax4.set_title("Path length (greedy policy)", color=BLUE, fontweight="bold")
ax4.grid(True, alpha=0.3)

# ── 5. Q-value heatmap (max_a Q(s,a)) ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
V = np.max(Q, axis=1).reshape(env.n_rows, env.n_cols)
# mask walls
V_masked = np.ma.array(V, mask=False)
for (r, c) in env.walls:
    V_masked.mask[r, c] = True
im = ax5.imshow(V_masked, cmap="YlOrRd")
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
ax5.set_title("State-value  max$_a$ Q(s,a)", color=BLUE, fontweight="bold")
ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}
policy = greedy_policy(Q)
for r in range(env.n_rows):
    for c in range(env.n_cols):
        s = r * env.n_cols + c
        if (r, c) in env.walls:
            ax5.text(c, r, "■", ha="center", va="center",
                     fontsize=11, color="#374151")
        elif (r, c) == env.goal:
            ax5.text(c, r, "G", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=BLUE)
        else:
            ax5.text(c, r, ARROW[policy[s]], ha="center", va="center",
                     fontsize=13, color=BLUE)
ax5.set_xticks(range(env.n_cols))
ax5.set_yticks(range(env.n_rows))
ax5.grid(False)

plt.savefig("qlearning_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to qlearning_results.png")

# ── Final policy on the grid ──────────────────────────────────────────
env.render(policy=greedy_policy(Q),
           title=f"Learned Greedy Policy  (α={ALPHA}, γ={GAMMA})")

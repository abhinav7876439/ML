"""
gridworld.py  —  A simple Gridworld environment for the LSO 2026 hands-on session.
========================================================================
Grid layout (default 5x5):

    S . . . .
    . # # . .
    . # . . .
    . . . # .
    . . . # G

  S = start (0,0)   G = goal (4,4)   # = wall   . = free cell

The agent moves UP / DOWN / LEFT / RIGHT.
  - Stepping into a wall or out of bounds: agent stays, receives step penalty.
  - Reaching the goal: +10 reward, episode ends.
  - Every other step: -0.1 step penalty (encourages shorter paths).

Stochastic variant (slip_prob > 0):
  With probability slip_prob the intended action is replaced by a random one,
  modelling a slippery floor.  Useful for discussing stochastic transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────────────────────────────
#  Default grid  (True = wall)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_WALLS = {(1, 1), (1, 2), (2, 1), (3, 3), (4, 3)}

ACTIONS      = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
N_ACTIONS    = 4


class GridWorld:
    """
    Parameters
    ----------
    n_rows, n_cols : grid dimensions
    start          : (row, col) of the start cell
    goal           : (row, col) of the goal cell
    walls          : set of (row, col) wall positions
    goal_reward    : reward for reaching the goal
    step_penalty   : (negative) reward per non-goal step
    slip_prob      : probability of taking a uniformly random action instead
                     of the intended one  (0 = fully deterministic)
    """

    def __init__(
        self,
        n_rows      = 5,
        n_cols      = 5,
        start       = (0, 0),
        goal        = (4, 4),
        walls       = None,
        goal_reward = 10.0,
        step_penalty= -0.1,
        slip_prob   = 0.0,
    ):
        self.n_rows       = n_rows
        self.n_cols       = n_cols
        self.start        = start
        self.goal         = goal
        self.walls        = walls if walls is not None else DEFAULT_WALLS
        self.goal_reward  = goal_reward
        self.step_penalty = step_penalty
        self.slip_prob    = slip_prob

        self.n_states  = n_rows * n_cols
        self.n_actions = N_ACTIONS
        self.state     = start          # current (row, col)

    # ── helpers ──────────────────────────────────────────────────────

    def _rc_to_s(self, r, c):
        """(row, col)  →  flat state index."""
        return r * self.n_cols + c

    def _s_to_rc(self, s):
        """Flat state index  →  (row, col)."""
        return divmod(s, self.n_cols)

    def state_index(self):
        """Return flat index of current state."""
        return self._rc_to_s(*self.state)

    def _is_valid(self, r, c):
        """True if (r,c) is inside the grid and not a wall."""
        return (0 <= r < self.n_rows and
                0 <= c < self.n_cols and
                (r, c) not in self.walls)

    # ── Gym-style interface ───────────────────────────────────────────

    def reset(self):
        """Reset to start. Returns flat state index."""
        self.state = self.start
        return self.state_index()

    def step(self, action):
        """
        Take action (0-3).  Returns (next_state, reward, done).

        next_state : flat state index
        reward     : float
        done       : bool — True if goal reached
        """
        # stochastic slip
        if self.slip_prob > 0 and np.random.rand() < self.slip_prob:
            action = np.random.randint(N_ACTIONS)

        dr, dc = ACTION_DELTA[action]
        r, c   = self.state
        nr, nc = r + dr, c + dc

        if self._is_valid(nr, nc):
            self.state = (nr, nc)

        done   = (self.state == self.goal)
        reward = self.goal_reward if done else self.step_penalty
        return self.state_index(), reward, done

    def render(self, policy=None, title="GridWorld"):
        """
        Render the grid.  If a flat policy array is supplied (shape: n_states,)
        arrows are drawn showing the greedy action in each free cell.
        """
        ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}

        grid = np.zeros((self.n_rows, self.n_cols))
        for (r, c) in self.walls:
            grid[r, c] = 1
        sr, sc = self.start
        gr, gc = self.goal
        grid[sr, sc] = 2
        grid[gr, gc] = 3

        cmap = ListedColormap(["#EEF2FF", "#374151", "#C8902A", "#16A34A"])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)

        # grid lines
        for x in range(self.n_cols + 1):
            ax.axvline(x - 0.5, color="white", lw=1.5)
        for y in range(self.n_rows + 1):
            ax.axhline(y - 0.5, color="white", lw=1.5)

        # labels / arrows
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) == self.start:
                    ax.text(c, r, "S", ha="center", va="center",
                            fontsize=14, fontweight="bold", color="white")
                elif (r, c) == self.goal:
                    ax.text(c, r, "G", ha="center", va="center",
                            fontsize=14, fontweight="bold", color="white")
                elif (r, c) in self.walls:
                    pass
                elif policy is not None:
                    s = self._rc_to_s(r, c)
                    ax.text(c, r, ARROW[policy[s]], ha="center", va="center",
                            fontsize=16, color="#002147")

        ax.set_xticks(range(self.n_cols))
        ax.set_yticks(range(self.n_rows))
        ax.set_xticklabels(range(self.n_cols))
        ax.set_yticklabels(range(self.n_rows))
        ax.set_title(title, fontsize=13, fontweight="bold", color="#002147")

        patches = [
            mpatches.Patch(color="#C8902A", label="Start (S)"),
            mpatches.Patch(color="#16A34A", label="Goal (G)"),
            mpatches.Patch(color="#374151", label="Wall"),
            mpatches.Patch(color="#EEF2FF", label="Free cell"),
        ]
        ax.legend(handles=patches, loc="upper right",
                  bbox_to_anchor=(1.45, 1.02), fontsize=9)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = GridWorld()
    print("GridWorld created.")
    print(f"  Grid size  : {env.n_rows} x {env.n_cols}")
    print(f"  States     : {env.n_states}")
    print(f"  Actions    : {N_ACTIONS}  ({ACTIONS})")
    print(f"  Start      : {env.start}  →  state index {env.reset()}")
    print(f"  Goal       : {env.goal}   →  state index {env._rc_to_s(*env.goal)}")
    print()

    # one random episode
    s = env.reset()
    print("Random episode (max 20 steps):")
    for step in range(20):
        a        = np.random.randint(N_ACTIONS)
        s_next, r, done = env.step(a)
        print(f"  step {step+1:2d}: action={ACTIONS[a]:5s}  "
              f"state={env._s_to_rc(s_next)}  reward={r:+.1f}  done={done}")
        s = s_next
        if done:
            break

    env.render(title="GridWorld — no policy")

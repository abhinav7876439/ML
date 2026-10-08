import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import bisect
import networkx as nx
from scipy.optimize import brentq
# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")

class MDPSolver:
    """
    MDP solver with Value Iteration and Policy Iteration.
    Assumes 2 actions (RED=0, BLUE=1) and maximisation of reward.
    """
    def __init__(self, transitions, state_labels, gamma=0.9, epsilon=1e-6):
        """
        Parameters
        ----------
        transitions : dict
            Keys: (state, action) -> list of (prob, reward, next_state)
            next_state = -1 indicates a terminal state (zero value).
        state_labels : list of str
            Names of the states.
        gamma : float
            Discount factor.
        epsilon : float
            Convergence threshold for Value Iteration.
        """
        self.transitions = transitions
        self.state_labels = state_labels
        self.num_states = len(state_labels)
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = None
        self.policy = None

        # Build probability and reward arrays for each action
        self.P = [np.zeros((self.num_states, self.num_states)) for _ in range(2)]
        self.R = [np.zeros(self.num_states) for _ in range(2)]
        for (s, a), lst in transitions.items():
            for p, r, ns in lst:
                if ns != -1:  # ignore terminal (value = 0)
                    self.P[a][s, ns] += p
            # Expected immediate reward for (s,a)
            self.R[a][s] = sum(p * r for p, r, ns in lst if ns != -1)

    # ------------------- Value Iteration -------------------
    def _bellman_update(self, V):
        """One Bellman update (maximisation)."""
        V_new = np.zeros(self.num_states)
        for s in range(self.num_states):
            q = np.zeros(2)
            for a in range(2):
                q[a] = self.R[a][s] + self.gamma * np.dot(self.P[a][s, :], V)
            V_new[s] = np.max(q)
        return V_new

    def value_iteration(self, max_iterations=10000, verbose=False):
        """Run Value Iteration until convergence."""
        self.V = np.zeros(self.num_states)
        stop_thresh = self.epsilon * (1 - self.gamma) / (2 * self.gamma)
        for it in range(max_iterations):
            V_old = self.V.copy()
            self.V = self._bellman_update(V_old)
            diff = np.max(np.abs(self.V - V_old))
            if diff < stop_thresh:
                if verbose:
                    print(f"Value Iteration converged in {it+1} iterations, max change = {diff:.2e}")
                break
        self.policy = self._extract_policy(self.V)
        return {'V': self.V, 'policy': self.policy, 'iterations': it+1, 'error': diff}

    # ------------------- Policy Iteration -------------------
    def _policy_evaluation(self, policy):
        """Solve linear system for V^π: (I - γ P_π) V = R_π."""
        P_pi = np.zeros((self.num_states, self.num_states))
        R_pi = np.zeros(self.num_states)
        for s in range(self.num_states):
            a = policy[s]
            P_pi[s, :] = self.P[a][s, :]
            R_pi[s] = self.R[a][s]
        A = np.eye(self.num_states) - self.gamma * P_pi
        V = np.linalg.solve(A, R_pi)
        return V

    def policy_iteration(self, max_iterations=100, verbose=False):
        """Run Policy Iteration (exact evaluation + greedy improvement)."""
        # Start with an arbitrary policy (e.g., all RED)
        policy = np.zeros(self.num_states, dtype=int)
        for it in range(max_iterations):
            V = self._policy_evaluation(policy)
            new_policy = self._extract_policy(V)
            if np.array_equal(policy, new_policy):
                if verbose:
                    print(f"Policy Iteration converged in {it+1} iterations")
                break
            policy = new_policy
        self.V = V
        self.policy = policy
        return {'V': V, 'policy': policy, 'iterations': it+1}

    # ------------------- Helper methods -------------------
    def _extract_policy(self, V):
        """Greedy policy: action with highest Q-value."""
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            q = np.zeros(2)
            for a in range(2):
                q[a] = self.R[a][s] + self.gamma * np.dot(self.P[a][s, :], V)
            policy[s] = np.argmax(q)
        return policy

    def get_q_values(self, V):
        """Compute Q(s,a) from V."""
        Q = np.zeros((self.num_states, 2))
        for s in range(self.num_states):
            for a in range(2):
                Q[s, a] = self.R[a][s] + self.gamma * np.dot(self.P[a][s, :], V)
        return Q

    def print_results(self, method='value'):
        """Display V*, policy, and Q-values."""
        print("=" * 70)
        print(f"OPTIMAL VALUE FUNCTION AND POLICY ({method} iteration)")
        print("=" * 70)
        df = pd.DataFrame({
            'State': self.state_labels,
            'V*(s)': self.V,
            'Policy': ['RED' if p == 0 else 'BLUE' for p in self.policy]
        })
        print(df.to_string(index=False))

        Q = self.get_q_values(self.V)
        q_df = pd.DataFrame(Q, index=self.state_labels,
                            columns=['Q(s,RED)', 'Q(s,BLUE)'])
        q_df['Advantage'] = q_df['Q(s,RED)'] - q_df['Q(s,BLUE)']
        print("\nQ-VALUES")
        print(q_df)



def build_extended_mdp(N, p, C, K, C_switch):
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    # State labels
    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N+1)]

    # Active transitions
    for s in range(num_states):
        if s == 0:
            P1[s, 0] = 1.0
            R1[s] = -C
        else:
            P1[s, 0] = 1.0
            R1[s] = -(C + C_switch)

    # Passive transitions
    for s in range(num_states):
        if s == 0:
            P0[s, 2] = p[0]
            P0[s, 1] = 1-p[0]
            R0[s] = -(K*(1-p[0]))
        elif s == 1:
            P0[s, 2] = p[0]
            P0[s, 1] = 1-p[0]
            R0[s] = -(K*(1-p[0]))
        else:
            idx = s-1
            if idx < N:
                P0[s, s+1] = p[idx]
                P0[s, 1] = 1-p[idx]
                R0[s] = -(K*(1-p[idx]))
            else:
                P0[s, 1] = 1.0
                R0[s] = -K

    return P0, P1, R0, R1, state_labels

def print_transition_matrix(P, state_labels, action_name):
    print("="*70)
    print(f"Transition matrix for action {action_name}")
    print("="*70)
    df = pd.DataFrame(P, index=state_labels, columns=state_labels)
    print(df)


def print_reward_vector(R, state_labels, action_name):
    print("="*70)
    print(f"Reward vector for action {action_name}")
    print("="*70)
    df = pd.DataFrame(R, index=state_labels, columns=[action_name])
    print(df)


# ============================================================
# Define the MDP (transitions and rewards) – as per your diagram
# ============================================================
# States: 0=s1, 1=s2, 2=s3
# Actions: 0=RED, 1=BLUE


transitions = {
    # RED action (0)
    (0, 0): [(0.5, 0, 0), (0.5, -1, 1)],      # s1 -> s1 (r=0), s1->s2 (r=-1)
    (1, 0): [(0.25, -1, 0), (0.75, -2, 2)],   # s2 -> s1 (r=-1), s2->s3 (r=-2)
    (2, 0): [(0.5, 3, 1), (0.5, 3, 2)],       # s3 -> s2 (r=3), s3->s3 (r=3)
    # BLUE action (1) – always to s1
    (0, 1): [(1.0, 1, 0)],                     # s1 -> s1 (r=1)
    (1, 1): [(1.0, 2, 0)],                     # s2 -> s1 (r=2)
    (2, 1): [(1.0, 1, 0)]                      # s3 -> s1 (r=1)
}

# Check that probabilities sum to 1 for all (state, action)
for (s, a), lst in transitions.items():
    total = sum(p for p, r, ns in lst)
    if total != 1.0:
        raise ValueError(f"Sum of probs for state {s}, action {a} is {total}, not 1")

state_labels = ['s1', 's2', 's3']
gamma = 0.9
epsilon = 1e-6

# Create solver instance
solver = MDPSolver(transitions, state_labels, gamma=gamma, epsilon=epsilon)

# ------------------- Run Value Iteration -------------------
print("\n" + "="*70)
print("VALUE ITERATION")
print("="*70)
vi_result = solver.value_iteration(verbose=True)
solver.print_results(method='value')

# ------------------- Run Policy Iteration -------------------
# Reset V and policy to avoid confusion
solver.V = None
solver.policy = None
print("\n\n" + "="*70)
print("POLICY ITERATION")
print("="*70)
pi_result = solver.policy_iteration(verbose=True)
solver.print_results(method='policy')



# ----------------------------------------------------------------------
# Main comparison
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Problem parameters
    N = 5
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)                       # terminal zero for the extended MDP
    C, K = 5.0, 500.0
    C_switch = 1.0
    beta = 0.9999                           # discount factor

    print("=" * 70)
    print(f"Survival probabilities p[0..{N}]: {np.round(p_full, 3)}")
    print("=" * 70)

    # Build MDP
    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)

    # Print matrices
    print_transition_matrix(P0_ext, state_labels_ext, "Passive (a=0)")
    print_reward_vector(R0_ext, state_labels_ext, "Passive (a=0)")
    print_transition_matrix(P1_ext, state_labels_ext, "Active (a=1)")
    print_reward_vector(R1_ext, state_labels_ext, "Active (a=1)")

    # ----- Discounted Indices -----
    print("\n" + "="*70)
    print("DISCOUNTED WHITTLE INDICES (beta = {})".format(beta))
    print("="*70)

    

    # ----- Average Cost Indices -----
    print("\n" + "="*70)
    print("AVERAGE COST WHITTLE INDICES")
    print("="*70)


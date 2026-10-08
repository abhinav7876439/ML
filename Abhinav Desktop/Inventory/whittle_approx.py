"""
Approximates Whittle indexes with the approach described in F Robeldo Paper
"""

import markovianbandit as mb
import numpy as np

def tab_QWI_homogeneous(MDPs : list[mb.RestlessBandit], discount, M, epsilon, t_end, alpha, beta, verbose=False, skip=1):
    """
    - MDPs : list of identical MDPs, each treated as a black-box, i.e we do not access transition probabilities
    - discount : for the MDP. Used only in Q and lambda equations
    - M : number of arms to activate at each time step
    - epsilon : probability of choosing M arms at random at each time step
    - t_end : number of steps for the approximation
    - alpha : function that takes an int n and returns the nth step size for fast timescale
    - beta : function that takes an int n and returns the nth step size for slow timescale
    - skip : returns one index every skip time steps
    """
    N = len(MDPs) # number of arms
    S = MDPs[0].get_state_space_size() # number of states of each MDP
    old_states = [MDP.start_simulation(0) for MDP in MDPs] # initial states
    rewards = [0 for _ in range(N)] # rewards associated with old states and chosen action
    new_states = [0 for _ in range(N)]

    # initialize Q and Lambda values
    Q_values = np.zeros((S, S, 2), dtype=float)
    Lambda_values = np.zeros(S, dtype=float)
    Lambda_history = np.zeros((S, t_end + 1), dtype=float)
    print("=== QWI STOCHASTIC APPROXIMATION ===")

    for t in range(1, t_end + 1):
        if 100 * t % t_end == 0:
            print(f"Progress {100 * t/t_end}%")

        actions = np.zeros(N, dtype=int)
        # if np.random.random() < epsilon(t): # activate M arms at random
            # arms_to_activate = np.random.choice(N, M, replace=False)
            #else:
                #arms_to_activate = np.argpartition(a, -M)[-M:] # NOT TESTED
        if np.random.random() < epsilon(t): # activate M arms at random
            arms_to_activate = np.random.choice(N, M, replace=False)
        else:
            arms_to_activate = np.random.choice(N, M, replace=False)  # Replace with your selection logic if needed

        for i in arms_to_activate:
            actions[i] = 1

        for i, a in enumerate(actions):
            new_states[i], rewards[i] = MDPs[i].next(a) # apply action a to MDP i and get associated reward and new state

        alpha_step, beta_step = alpha(t), beta(t)
        if verbose:
            print(Q_values)

        for x in range(S):
            # save old Q-values for row x
            old_Q_x = Q_values[x, :, :].copy()
            # update Q-values
            for r_n, s_n, new_s_n, a_n in zip(rewards, old_states, new_states, actions):
                next_Q = discount * max(old_Q_x[new_s_n, 0], old_Q_x[new_s_n, 1])
                Q_values[x, s_n, a_n] *= (1 - alpha_step)
                if a_n == 0:
                    Q_values[x, s_n, a_n] += alpha_step * (r_n + Lambda_values[x] + next_Q)
                else: # a_n==1
                    Q_values[x, s_n, a_n] += alpha_step * (r_n + next_Q)

            # update Lambda-values
            Lambda_values[x] += beta_step * (old_Q_x[x, 1] - old_Q_x[x, 0])
            Lambda_history[x, t] = Lambda_values[x] # save new lambda values in history

        old_states = new_states.copy() # update old_states values

    return Lambda_history[:,::skip]




import numpy as np

# Example: Inventory Arm as a RestlessBandit
class InventoryArm:
    def __init__(self, max_inventory, h, p, gamma, l, lam, mu, switch_cost):
        self.max_inventory = max_inventory
        self.h = h
        self.p = p
        self.gamma = gamma
        self.l = l
        self.lam = lam
        self.mu = mu
        self.switch_cost = switch_cost
        self.state_space = range(max_inventory + 1)
        self.state = 0

    def get_state_space_size(self):
        return len(self.state_space)

    def start_simulation(self, seed):
        np.random.seed(seed)
        self.state = 0
        return self.state

    def next(self, action):
        # action: 0 = idle, 1 = produce
        produced = int(action)
        demand = np.random.poisson(self.lam)
        perish = np.random.binomial(self.state, self.gamma)
        next_state = self.state + produced - demand - perish
        next_state = max(0, min(next_state, self.max_inventory))
        # Cost function (can be customized)
        cost = self.h * next_state + self.p * self.gamma * next_state + self.l * self.lam * (next_state == 0)
        self.state = next_state
        reward = -cost  # negative cost as reward
        return next_state, reward

# Parameters
K = 3
max_inventory = 5
h, p, gamma, l, lam, mu, switch_cost = 1, 2, 0.1, 5, 0.6, 0.8, 3

# Create bandit objects for each arm
MDPs = [InventoryArm(max_inventory, h, p, gamma, l, lam, mu, switch_cost) for _ in range(K)]

# Define step sizes
alpha = lambda t: 1.0 / (t + 1)
beta = lambda t: 1.0 / (10 * (t + 1))
discount = 0.95
M = 1
epsilon = lambda t: 0.1
t_end = 1000

# Run Whittle index approximation
from whittle_approx import tab_QWI_homogeneous
Lambda_history = tab_QWI_homogeneous(MDPs, discount, M, epsilon, t_end, alpha, beta, verbose=True, skip=10)

print("Learned Lambda history (Whittle indices):")
print(Lambda_history)

def tab_QGI_homogeneous(MDPs : list[mb.RestlessBandit], discount, M, epsilon, t_end, alpha, beta, verbose=False, skip=1):
    """
    - MDPs : list of identical MDPs, each treated as a black-box, i.e we do not access transition probabilities
    - discount : for the MDP. Used only in Q and lambda equations
    - M : number of arms to activate at each time step (MUST BE ONE)
    - epsilon(t) : probability of choosing M arms at random at time step t
    - t_end : number of steps for the approximation
    - alpha : function that takes an int n and returns the nth step size for fast timescale
    - beta : function that takes an int n and returns the nth step size for slow timescale
    - skip : returns one index every skip time steps
    """
    N = len(MDPs) # number of arms
    S = MDPs[0].get_P0P1R0R1()[2].size # number of states of each MDP
    old_states = [MDP.start_simulation(0) for MDP in MDPs] # initial states
    rewards = [0 for _ in range(N)] # rewards associated with old states and chosen action
    new_states = [0 for _ in range(N)]

    assert M==1 and discount < 1 # Gittins indices

    # initialize Q and M values
    Q_values = np.zeros((S, S), dtype=float)
    M_values = np.zeros(S, dtype=float)
    GI_history = np.zeros((S, t_end + 1), dtype=float)
    print("=== QGI STOCHASTIC APPROXIMATION ===")

    for t in range(1, t_end + 1):
        if 100 * t % t_end == 0:
            print(f"Progress {100 * t/t_end}%")

        actions = np.zeros(N, dtype=int)
        if np.random.random() < epsilon(t): # activate M arms at random
            arm_to_activate = np.random.randint(0, N)
        else:
            arm_to_activate = np.argpartition(a, -M)[-M:] # NOT TESTED
        actions[arm_to_activate] = 1

        for i, a in enumerate(actions):
            new_states[i], rewards[i] = MDPs[i].next(a) # apply action a to MDP i and get associated reward and new state

        alpha_step, beta_step = alpha(t), beta(t)
        
        for x in range(S):
            # save old Q-values for row x
            old_Q_x = Q_values[x, :].copy()
            # update Q-values
            for r_n, s_n, new_s_n, a_n in zip(rewards, old_states, new_states, actions):
                next_Q = discount * max(old_Q_x[new_s_n], M_values[x])
                if a_n == 1:
                    Q_values[x, s_n] *= (1 - alpha_step)
                    Q_values[x, s_n] += alpha_step * (r_n + next_Q)

            # update Lambda-values
            M_values[x] += beta_step * (old_Q_x[x] - M_values[x])
            GI_history[x, t] = (1 - discount) * M_values[x] # save new lambda values in history

        old_states = new_states.copy() # update old_states values

    return GI_history[:,::skip]
    
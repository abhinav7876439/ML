import numpy as np

class InventoryArm:
    def __init__(self, max_inventory, h, p, l, gamma, lam, mu, switch_cost):
        self.max_inventory = max_inventory
        self.h = h
        self.p = p
        self.l = l
        self.gamma = gamma      # perish rate
        self.lam = lam          # demand rate
        self.mu = mu            # production rate
        self.switch_cost = switch_cost

        self.state_space = range(max_inventory + 1)
        self.state = 0
        self.prev_action = 0   # track previous action for switching cost

        # For parameter estimation
        self.demand_sum = 0
        self.demand_count = 0
        self.perish_sum = 0
        self.perish_count = 0
        self.prod_sum = 0
        self.prod_count = 0

    def get_state_space_size(self):
        return len(self.state_space)

    def start_simulation(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = 0
        self.prev_action = 0
        return self.state

    def next(self, action):
        """
        Simulate one step of inventory evolution.
        action: 0 = idle, 1 = produce
        """
        # Production: Poisson(μ) items produced when producing
        produced = np.random.poisson(self.mu) if action == 1 else 0
        self.prod_sum += produced
        self.prod_count += 1 if action == 1 else 0

        # Demand arrivals: Poisson(λ)
        demand = np.random.poisson(self.lam)
        self.demand_sum += demand
        self.demand_count += 1

        # Perish: Binomial(state, γ)
        perish = np.random.binomial(self.state, self.gamma)
        self.perish_sum += perish
        self.perish_count += 1

        # Inventory update
        next_state = self.state + produced - demand - perish
        next_state = max(0, min(next_state, self.max_inventory))

        # Cost components
        holding_cost = self.h * next_state
        perish_cost = self.p * perish
        lost_sales_cost = self.l * max(demand - (self.state + produced - perish), 0)
        total_cost = holding_cost + perish_cost + lost_sales_cost

        # Add switch cost if action changed
        if action != self.prev_action:
            total_cost += self.switch_cost

        reward = -total_cost
        self.state = next_state
        self.prev_action = action

        return next_state, reward

    def get_estimates(self):
        # Avoid divide-by-zero
        est_gamma = self.perish_sum / max(self.perish_count, 1)
        est_lambda = self.demand_sum / max(self.demand_count, 1)
        est_mu = self.prod_sum / max(self.prod_count, 1)
        return est_gamma, est_lambda, est_mu


# Parameters
max_inventory = 5
h, p, l, gamma, lam, mu, switch_cost = 1, 2, 5, 0.1, 0.6, 0.8, 3

arm = InventoryArm(max_inventory, h, p, l, gamma, lam, mu, switch_cost)

# Simulation
t_end = 10000
arm.start_simulation(seed=42)

for t in range(t_end):
    action = np.random.choice([0, 1])
    arm.next(action)

# Estimation
est_gamma, est_lambda, est_mu = arm.get_estimates()

print(f"Estimated gamma (perish rate): {est_gamma:.3f}")
print(f"Estimated lambda (demand rate): {est_lambda:.3f}")
print(f"Estimated mu (production rate): {est_mu:.3f}")


############################################
import numpy as np

# ------------------------------
# Inventory Arm with online parameter learning
# ------------------------------
class InventoryArm:
    def __init__(self, max_inventory, h, p, l, switch_cost):
        self.max_inventory = max_inventory
        self.h = h
        self.p = p
        self.l = l
        self.switch_cost = switch_cost

        self.state_space = range(max_inventory + 1)
        self.state = 0
        self.prev_action = 0

        # Unknown parameters (to be learned online)
        self.gamma = 0.1
        self.lam = 0.5
        self.mu = 0.5

    def get_state_space_size(self):
        return len(self.state_space)

    def start_simulation(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = 0
        self.prev_action = 0
        return self.state

    def next(self, action):
        # Simulate one step using current parameter estimates
        produced = np.random.poisson(self.mu) if action == 1 else 0   ### production -- Poisson(μ) (How to use those sort of ideas in the form of interarrival times?)
        demand = np.random.poisson(self.lam)
        perish = np.random.binomial(self.state, self.gamma)

        next_state = self.state + produced - demand - perish
        next_state = max(0, min(next_state, self.max_inventory))

        # Cost components
        holding_cost = self.h * next_state
        perish_cost = self.p * perish
        lost_sales_cost = self.l * max(demand - (self.state + produced - perish), 0)  ## we need to capture lost sales for each item or does we have a fixed cost for lost sales
        total_cost = holding_cost + perish_cost + lost_sales_cost

        # Switching cost
        if action != self.prev_action:
            total_cost += self.switch_cost

        reward = -total_cost

        # ------------------------------
        # Online parameter learning
        # ------------------------------
        eta = 0.01  # learning rate
        self.lam = max(0.01, self.lam + eta * (demand - self.lam))  ## error between observed demand and current estimate
        self.gamma = np.clip(self.gamma + eta * ((perish / max(self.state, 1)) - self.gamma), 0, 1)
        if action == 1:
            self.mu = max(0.01, self.mu + eta * (produced - self.mu))

        self.state = next_state
        self.prev_action = action

        return next_state, reward, self.lam, self.gamma, self.mu

# ------------------------------
# QWI approximation with parameter tracking
# ------------------------------
def tab_QWI_homogeneous(MDPs, discount, M, epsilon, t_end, alpha, beta, verbose=False, skip=1):
    N = len(MDPs)
    S = MDPs[0].get_state_space_size()
    old_states = [MDP.start_simulation(0) for MDP in MDPs]
    rewards = [0 for _ in range(N)]
    new_states = [0 for _ in range(N)]

    Q_values = np.zeros((S, S, 2), dtype=float)
    Lambda_values = np.zeros(S, dtype=float)
    Lambda_history = np.zeros((S, t_end + 1), dtype=float)

    # Store parameter histories
    lam_history = np.zeros((N, t_end + 1))
    gamma_history = np.zeros((N, t_end + 1))
    mu_history = np.zeros((N, t_end + 1))

    print("=== QWI STOCHASTIC APPROXIMATION ===")

    for t in range(1, t_end + 1):
        # Progress display
        if t % (t_end // 100) == 0:
            print(f"Progress {t * 100 / t_end:.1f}%")

        actions = np.zeros(N, dtype=int)
        if np.random.random() < epsilon(t):
            arms_to_activate = np.random.choice(N, M, replace=False)
        else:
            arms_to_activate = np.random.choice(N, M, replace=False)  # placeholder

        for i in arms_to_activate:
            actions[i] = 1

        # Step each MDP and track parameters
        for i, a in enumerate(actions):
            new_states[i], rewards[i], lam, gamma, mu = MDPs[i].next(a)
            lam_history[i, t] = lam
            gamma_history[i, t] = gamma
            mu_history[i, t] = mu

        alpha_step = alpha(t)
        beta_step = beta(t)

        for x in range(S):
            old_Q_x = Q_values[x, :, :].copy()
            for r_n, s_n, new_s_n, a_n in zip(rewards, old_states, new_states, actions):
                next_Q = discount * max(old_Q_x[new_s_n, 0], old_Q_x[new_s_n, 1])
                Q_values[x, s_n, a_n] *= (1 - alpha_step)
                if a_n == 0:
                    Q_values[x, s_n, a_n] += alpha_step * (r_n + Lambda_values[x] + next_Q)
                else:
                    Q_values[x, s_n, a_n] += alpha_step * (r_n + next_Q)
            Lambda_values[x] += beta_step * (old_Q_x[x, 1] - old_Q_x[x, 0])
            Lambda_history[x, t] = Lambda_values[x]

        old_states = new_states.copy()

    return Lambda_history[:, ::skip], lam_history[:, ::skip], gamma_history[:, ::skip], mu_history[:, ::skip]

# ------------------------------
# Simulation parameters
# ------------------------------
N = 3
max_inventory = 5
arms = [InventoryArm(max_inventory=max_inventory, h=1, p=2, l=5, switch_cost=3) for _ in range(N)]

alpha = lambda t: 0.1 / (1 + 0.001 * t)
beta = lambda t: 0.01 / (1 + 0.001 * t)
epsilon = lambda t: 0.05

# Run QWI approximation with parameter learning (Discounted case)
Lambda_hist, lam_hist, gamma_hist, mu_hist = tab_QWI_homogeneous(
    arms, discount=0.95, M=1, epsilon=epsilon, t_end=5000,
    alpha=alpha, beta=beta, verbose=False
)

# ------------------------------
# Print final learned parameters
# ------------------------------
for i in range(N):
    print(f"\nArm {i+1} final estimates:")
    print(f"  lambda: {lam_hist[i, -1]:.3f}")
    print(f"  gamma:  {gamma_hist[i, -1]:.3f}")
    print(f"  mu:     {mu_hist[i, -1]:.3f}")

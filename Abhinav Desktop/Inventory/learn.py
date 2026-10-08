import numpy as np

class InventoryArm:   ## (No switching depend on action)
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

        reward = -total_cost
        self.state = next_state

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



######################################################
print("\n##############################")

import numpy as np

class InventoryArm:
    def __init__(self, max_inventory, h, p, l, switch_cost):
        self.max_inventory = max_inventory
        self.h = h
        self.p = p
        self.l = l
        self.switch_cost = switch_cost
        self.state_space = range(max_inventory + 1)
        self.state = 0

        # Initialize unknown parameters (initial guesses)
        self.gamma = 0.05   # perishability rate
        self.lam = 0.5      # demand rate
        self.mu = 0.5       # production rate

        # For online estimation
        self.alpha = 0.01   # learning rate for exponential moving average
        self.demand_sum = 0
        self.perish_sum = 0
        self.prod_sum = 0

    def get_state_space_size(self):
        return len(self.state_space)

    def start_simulation(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = 0
        return self.state

    def next(self, action):
        """Simulate one step using current parameter estimates."""
        # Production
        produced = np.random.poisson(max(self.mu, 1e-3)) if action == 1 else 0
        self.prod_sum += produced

        # Demand and perish
        demand = np.random.poisson(max(self.lam, 1e-3))
        perish = np.random.binomial(self.state, min(max(self.gamma, 0), 1))

        self.demand_sum += demand
        self.perish_sum += perish

        # Online parameter updates (EMA smoothing)
        self.lam = (1 - self.alpha) * self.lam + self.alpha * demand
        self.gamma = (1 - self.alpha) * self.gamma + self.alpha * (perish / max(self.state, 1))
        if action == 1:
            self.mu = (1 - self.alpha) * self.mu + self.alpha * produced

        # Inventory update
        next_state = self.state + produced - demand - perish
        next_state = max(0, min(next_state, self.max_inventory))

        # Cost and reward
        cost = self.h * next_state + self.p * perish + self.l * max(demand - (self.state + produced - perish), 0)
        reward = -cost

        self.state = next_state
        return next_state, reward

    def get_estimates(self):
        return self.gamma, self.lam, self.mu


# Parameters
max_inventory = 5
h, p, l, switch_cost = 1, 2, 5, 3
arm = InventoryArm(max_inventory, h, p, l, switch_cost)

# Simulation loop
t_end = 10000
arm.start_simulation(seed=42)

for t in range(1, t_end + 1):
    action = np.random.choice([0, 1])
    arm.next(action)

    if t % 1000 == 0:
        est_gamma, est_lambda, est_mu = arm.get_estimates()
        print(f"Step {t}: γ={est_gamma:.3f}, λ={est_lambda:.3f}, μ={est_mu:.3f}")

# Final estimates
est_gamma, est_lambda, est_mu = arm.get_estimates()
print(f"\nFinal estimated gamma (perish rate): {est_gamma:.3f}")
print(f"Final estimated lambda (demand rate): {est_lambda:.3f}")
print(f"Final estimated mu (production rate): {est_mu:.3f}")

######################################################




######################################################
import numpy as np
import matplotlib.pyplot as plt

class InventoryAgent:
    def __init__(self, max_inventory, h, p, l, switch_cost):
        self.max_inventory = max_inventory
        self.h = h
        self.p = p
        self.l = l
        self.switch_cost = switch_cost
        self.state = 0

        # Initialize unknown parameters
        self.gamma = 0.05  # perishability rate
        self.lam = 0.5     # demand rate
        self.mu = 0.5      # production rate

        # EMA smoothing factor
        self.alpha = 0.01

    def simulate_step(self, action):
        produced = np.random.poisson(max(self.mu, 1e-3)) if action == 1 else 0
        demand = np.random.poisson(max(self.lam, 1e-3))
        perish = np.random.binomial(self.state, min(max(self.gamma, 0), 1))

        # EMA updates
        self.lam = (1 - self.alpha) * self.lam + self.alpha * demand
        self.gamma = (1 - self.alpha) * self.gamma + self.alpha * (perish / max(self.state, 1))
        if action == 1:
            self.mu = (1 - self.alpha) * self.mu + self.alpha * produced

        next_state = self.state + produced - demand - perish
        next_state = max(0, min(next_state, self.max_inventory))

        cost = self.h * next_state + self.p * perish + self.l * max(demand - (self.state + produced - perish), 0)
        self.state = next_state
        return next_state, cost

    def expected_cost(self, state, action):
        expected_produced = max(self.mu, 1e-3) if action == 1 else 0
        expected_demand = max(self.lam, 1e-3)
        expected_perish = state * min(max(self.gamma, 0), 1)
        next_state = state + expected_produced - expected_demand - expected_perish
        next_state = max(0, min(int(round(next_state)), self.max_inventory))
        cost = self.h * next_state + self.p * expected_perish + self.l * max(expected_demand - (state + expected_produced - expected_perish), 0)
        return cost

    def select_action(self):
        costs = [self.expected_cost(self.state, a) for a in [0, 1]]
        return int(np.argmin(costs))

    def get_estimates(self):
        return self.gamma, self.lam, self.mu

# Parameters
max_inventory = 5
h, p, l, switch_cost = 1, 2, 5, 3
agent = InventoryAgent(max_inventory, h, p, l, switch_cost)

# Simulation
t_end = 10000
np.random.seed(42)

gamma_history = []
lambda_history = []
mu_history = []

for t in range(1, t_end + 1):
    action = agent.select_action()
    agent.simulate_step(action)

    g, lam, mu = agent.get_estimates()
    gamma_history.append(g)
    lambda_history.append(lam)
    mu_history.append(mu)

    if t % 1000 == 0:
        print(f"Step {t}: γ={g:.3f}, λ={lam:.3f}, μ={mu:.3f}, Inventory={agent.state}")

# Final estimates
est_gamma, est_lambda, est_mu = agent.get_estimates()
print(f"\nFinal estimated gamma (perish rate): {est_gamma:.3f}")
print(f"Final estimated lambda (demand rate): {est_lambda:.3f}")
print(f"Final estimated mu (production rate): {est_mu:.3f}")

# Plot convergence
plt.figure(figsize=(10,6))
plt.plot(gamma_history, label='γ (perish rate)')
plt.plot(lambda_history, label='λ (demand rate)')
plt.plot(mu_history, label='μ (production rate)')
plt.xlabel('Simulation step')
plt.ylabel('Parameter estimate')
plt.title('Convergence of Inventory Parameter Estimates')
plt.legend()
plt.grid(True)
plt.show()
######################################################




import numpy as np

class InventoryAgent:
    def __init__(self, max_inventory, h, p, l, switch_cost):
        self.max_inventory = max_inventory
        self.h = h
        self.p = p
        self.l = l
        self.switch_cost = switch_cost
        self.state = 0

        # Initialize unknown parameters
        self.gamma = 0.05  # perishability rate
        self.lam = 0.5     # demand rate
        self.mu = 0.5      # production rate

        # Online estimation sums
        self.demand_sum = 0
        self.demand_count = 0
        self.perish_sum = 0
        self.perish_count = 0
        self.prod_sum = 0
        self.prod_count = 0

    def simulate_step(self, action):
        # Produce if action==1
        if action == 1:
            produce = np.random.exponential(1 / max(self.mu, 1e-3))
            produced = int(produce)
            self.prod_sum += produced
            self.prod_count += 1
        else:
            produced = 0

        demand = np.random.poisson(max(self.lam, 1e-3))
        perish = np.random.binomial(self.state, min(max(self.gamma, 0), 1))

        # Update sums for online estimation
        self.demand_sum += demand
        self.demand_count += 1
        self.perish_sum += perish
        self.perish_count += self.state if self.state > 0 else 1

        # Online parameter updates (running averages)
        self.lam = self.demand_sum / max(self.demand_count, 1)
        self.gamma = self.perish_sum / max(self.perish_count, 1)
        self.mu = self.prod_sum / max(self.prod_count, 1)

        next_state = self.state + produced - demand - perish
        next_state = max(0, min(next_state, self.max_inventory))
        cost = self.h * next_state + self.p * self.gamma * next_state + self.l * self.lam * (next_state == 0)
        self.state = next_state
        return next_state, cost

    def expected_cost(self, state, action):
        expected_produced = max(self.mu, 1e-3) if action == 1 else 0
        expected_demand = max(self.lam, 1e-3)
        expected_perish = state * min(max(self.gamma, 0), 1)
        next_state = state + expected_produced - expected_demand - expected_perish
        next_state = max(0, min(int(round(next_state)), self.max_inventory))
        cost = self.h * next_state + self.p * self.gamma * next_state + self.l * self.lam * (next_state == 0)
        return cost

    def select_action(self):
        costs = [self.expected_cost(self.state, a) for a in [0, 1]]
        return int(np.argmin(costs))

    def get_estimates(self):
        # Return in order: lambda, gamma, mu
        return self.lam, self.gamma, self.mu

    def set_estimates(self, lam, gamma, mu):
        self.lam = lam
        self.gamma = gamma
        self.mu = mu

# Parameters
max_inventory = 10
h, p, l, switch_cost = 1, 2, 5, 3

# --- Facility A (source agent) ---
agent_A = InventoryAgent(max_inventory, h, p, l, switch_cost)
t_end_A = 5000
for t in range(1, t_end_A + 1):
    action = agent_A.select_action()
    agent_A.simulate_step(action)
    if t % 1000 == 0:
        est_lambda, est_gamma, est_mu = agent_A.get_estimates()
        print(f"Facility A - Step {t}: λ={est_lambda:.3f}, γ={est_gamma:.3f}, μ={est_mu:.3f}, Inventory={agent_A.state}")

# Transfer learned parameters
prior_lambda, prior_gamma, prior_mu = agent_A.get_estimates()
print("\n--- Transfer learning: using Facility A's estimates for Facility B ---")
print(f"Transferred λ={prior_lambda:.3f}, γ={prior_gamma:.3f}, μ={prior_mu:.3f}\n")

# --- Facility B (target agent) ---
agent_B = InventoryAgent(max_inventory, h, p, l, switch_cost)
agent_B.set_estimates(prior_lambda, prior_gamma, prior_mu)

t_end_B = 5000
for t in range(1, t_end_B + 1):
    action = agent_B.select_action()
    agent_B.simulate_step(action)
    if t % 1000 == 0:
        est_lambda, est_gamma, est_mu = agent_B.get_estimates()
        print(f"Facility B - Step {t}: λ={est_lambda:.3f}, γ={est_gamma:.3f}, μ={est_mu:.3f}, Inventory={agent_B.state}")

# Final estimates for Facility B
est_lambda, est_gamma, est_mu = agent_B.get_estimates()
print(f"\nFacility B - Final estimated λ: {est_lambda:.3f}")
print(f"Facility B - Final estimated γ: {est_gamma:.3f}")
print(f"Facility B - Final estimated μ: {est_mu:.3f}")


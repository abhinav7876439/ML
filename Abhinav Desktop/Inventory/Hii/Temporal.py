import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools

class TDLambdaWhittleLearner:
    def __init__(self, classes, gamma=0.95, lambda_param=0.8, alpha=0.01):
        self.classes = classes
        self.K = len(classes)
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.alpha = alpha
        
        # Initialize Whittle index approximations for each class and state
        self.whittle_indices = []
        for k in range(self.K):
            # Estimate reasonable max inventory based on demand and production rates
            max_inv = int(5 * max(classes[k]['lambda'], classes[k]['mu'])) + 5
            self.whittle_indices.append(np.zeros(max_inv + 1))
        
        # Eligibility traces
        self.e_traces = []
        for k in range(self.K):
            max_inv = len(self.whittle_indices[k])
            self.e_traces.append(np.zeros(max_inv))
    
    def reset_traces(self):
        """Reset eligibility traces"""
        for k in range(self.K):
            self.e_traces[k].fill(0)
    
    def get_whittle_index(self, k, state):
        """Get Whittle index for class k at inventory level"""
        if state < len(self.whittle_indices[k]):
            return self.whittle_indices[k][state]
        else:
            # Extrapolate for states beyond what we've seen
            return self.whittle_indices[k][-1]
    
    def choose_action(self, state):
        """Choose action based on current Whittle index estimates"""
        indices = [self.get_whittle_index(k, state[k]) for k in range(self.K)]
        return np.argmax(indices)
    
    def compute_td_error(self, k, state, next_state, reward, subsidy=0):
        """Compute TD error for a class"""
        current_value = self.get_whittle_index(k, state)
        next_value = self.get_whittle_index(k, next_state) if next_state < len(self.whittle_indices[k]) else 0
        td_error = reward - subsidy + self.gamma * next_value - current_value
        return td_error
    
    def update_whittle_indices(self, k, state, td_error):
        """Update Whittle indices using TD error"""
        if state < len(self.whittle_indices[k]):
            # Update with eligibility traces
            self.whittle_indices[k][state] += self.alpha * td_error * self.e_traces[k][state]
            
            # Decay eligibility traces
            self.e_traces[k] *= self.gamma * self.lambda_param
    
    def update_traces(self, k, state):
        """Update eligibility traces for a state"""
        if state < len(self.e_traces[k]):
            self.e_traces[k][state] += 1
    
    def learn_from_episode(self, T=1000, explore_prob=0.1):
        """Learn from a single episode of experience"""
        # Initialize state
        state = [random.randint(0, 5) for _ in range(self.K)]
        total_cost = 0
        self.reset_traces()
        
        for t in range(T):
            # Choose action (epsilon-greedy)
            if random.random() < explore_prob:
                action = random.randint(0, self.K-1)
            else:
                action = self.choose_action(state)
            
            # Simulate the effect of the action
            next_state, cost = self.simulate_step(state, action)
            total_cost += cost
            
            # For each class, update based on what happened
            for k in range(self.K):
                # Calculate "reward" (negative cost) for this class
                class_reward = -self.calculate_class_cost(k, state[k], action == k)
                
                # Calculate TD error
                td_error = self.compute_td_error(k, state[k], next_state[k], class_reward)
                
                # Update eligibility traces
                self.update_traces(k, state[k])
                
                # Update Whittle indices
                self.update_whittle_indices(k, state[k], td_error)
            
            # Move to next state
            state = next_state
        
        return total_cost / T
    
    def simulate_step(self, state, action):
        """Simulate one time step"""
        next_state = state.copy()
        total_cost = 0
        
        # Process each class
        for k in range(self.K):
            # Demand arrives
            if random.random() < self.classes[k]['lambda']:
                if next_state[k] > 0:
                    next_state[k] -= 1
                else:
                    # Lost sale
                    total_cost += self.classes[k]['D']
            
            # Production if this class is being served
            if k == action and random.random() < self.classes[k]['mu']:
                next_state[k] += 1
            
            # Holding cost
            total_cost += self.classes[k]['h'] * next_state[k]
        
        return next_state, total_cost
    
    def calculate_class_cost(self, k, inventory, served):
        """Calculate cost for a specific class"""
        cost = self.classes[k]['h'] * inventory
        
        # Expected stockout cost
        if inventory == 0:
            cost += self.classes[k]['D'] * self.classes[k]['lambda']
        
        return cost
    
    def learn(self, n_episodes=1000, initial_explore=0.3, final_explore=0.01):
        """Learn over multiple episodes"""
        costs = []
        explore_rate = initial_explore
        
        for episode in range(n_episodes):
            # Decay exploration rate
            explore_rate = final_explore + (initial_explore - final_explore) * np.exp(-episode / (n_episodes / 3))
            
            # Learn from episode
            avg_cost = self.learn_from_episode(explore_prob=explore_rate)
            costs.append(avg_cost)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Avg Cost: {avg_cost:.4f}, Explore: {explore_rate:.4f}")
        
        return costs
    
    def get_policy(self):
        """Get the learned policy"""
        policy = {}
        for k in range(self.K):
            # Find threshold where Whittle index becomes positive
            indices = self.whittle_indices[k]
            threshold = None
            for i in range(len(indices)):
                if indices[i] > 0:
                    threshold = i
                    break
            
            policy[k] = threshold if threshold is not None else len(indices)
        
        return policy

# Q-learning approach for Whittle index learning
class QLearningWhittle:
    def __init__(self, classes, state_bins=10, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.classes = classes
        self.K = len(classes)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Discretize state space for each class
        self.state_bins = state_bins
        self.max_inventory = 20
        
        # Q-tables for each class (state, action) where action is 0 (passive) or 1 (active)
        self.Q_tables = [np.zeros((state_bins, 2)) for _ in range(self.K)]
        
        # Map continuous inventory to discrete state
        self.bin_size = self.max_inventory / state_bins
    
    def get_state_index(self, k, inventory):
        """Convert continuous inventory to discrete state index"""
        discrete_state = min(int(inventory / self.bin_size), self.state_bins - 1)
        return discrete_state
    
    def choose_action(self, k, inventory):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        state_idx = self.get_state_index(k, inventory)
        return 1 if self.Q_tables[k][state_idx, 1] > self.Q_tables[k][state_idx, 0] else 0
    
    def update(self, k, inventory, action, reward, next_inventory):
        """Update Q-table using Q-learning"""
        state_idx = self.get_state_index(k, inventory)
        next_state_idx = self.get_state_index(k, next_inventory)
        
        # Q-learning update
        best_next_action = np.argmax(self.Q_tables[k][next_state_idx])
        td_target = reward + self.gamma * self.Q_tables[k][next_state_idx, best_next_action]
        td_error = td_target - self.Q_tables[k][state_idx, action]
        
        self.Q_tables[k][state_idx, action] += self.alpha * td_error
    
    def learn_episode(self, T=1000):
        """Learn from one episode"""
        # Initialize state
        state = [random.randint(0, 10) for _ in range(self.K)]
        total_cost = 0
        
        for t in range(T):
            # For each class, choose action and update
            for k in range(self.K):
                action = self.choose_action(k, state[k])
                
                # Simulate one step for this class
                next_inventory, reward = self.simulate_class_step(k, state[k], action)
                
                # Update Q-table
                self.update(k, state[k], action, -reward, next_inventory)
                
                # Update state and accumulate cost
                state[k] = next_inventory
                total_cost += reward
            
        return total_cost / T
    
    def simulate_class_step(self, k, inventory, action):
        """Simulate one step for a single class"""
        cost = self.classes[k]['h'] * inventory
        
        # Demand arrives
        if random.random() < self.classes[k]['lambda']:
            if inventory > 0:
                inventory -= 1
            else:
                # Lost sale
                cost += self.classes[k]['D']
        
        # Production if active
        if action == 1 and random.random() < self.classes[k]['mu']:
            inventory += 1
        
        return inventory, cost
    
    def learn(self, n_episodes=1000):
        """Learn over multiple episodes"""
        costs = []
        
        for episode in range(n_episodes):
            avg_cost = self.learn_episode()
            costs.append(avg_cost)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Avg Cost: {avg_cost:.4f}")
        
        return costs
    
    def get_whittle_indices(self):
        """Estimate Whittle indices from Q-values"""
        whittle_indices = []
        
        for k in range(self.K):
            indices = []
            for state_idx in range(self.state_bins):
                # Whittle index is the difference between passive and active Q-values
                # This represents the subsidy that makes both actions equally attractive
                index = self.Q_tables[k][state_idx, 1] - self.Q_tables[k][state_idx, 0]
                indices.append(index)
            
            whittle_indices.append(indices)
        
        return whittle_indices

# Example usage and comparison
if __name__ == "__main__":
    # Define the classes
    classes = [
        {'lambda': 0.8, 'mu': 1.2, 'h': 0.5, 'D': 10.0},
        {'lambda': 0.6, 'mu': 1.0, 'h': 0.4, 'D': 8.0},
        {'lambda': 1.0, 'mu': 1.5, 'h': 0.7, 'D': 12.0}
    ]
    
    print("Learning Whittle indices with TD(λ)...")
    td_learner = TDLambdaWhittleLearner(classes, alpha=0.05)
    td_costs = td_learner.learn(n_episodes=500)
    
    print("\nLearning with Q-learning...")
    q_learner = QLearningWhittle(classes, state_bins=20, alpha=0.1)
    q_costs = q_learner.learn(n_episodes=500)
    
    # Get learned policies
    td_policy = td_learner.get_policy()
    q_whittle_indices = q_learner.get_whittle_indices()
    
    print("\nLearned TD(λ) thresholds:")
    for k, threshold in td_policy.items():
        print(f"Class {k+1}: Serve when inventory < {threshold}")
    
    print("\nLearned Q-learning Whittle indices (first few states):")
    for k, indices in enumerate(q_whittle_indices):
        print(f"Class {k+1}: {indices[:5]}")
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(td_costs, label='TD(λ) Learning')
    plt.plot(q_costs, label='Q-learning')
    plt.xlabel('Episode')
    plt.ylabel('Average Cost')
    plt.title('Learning Curves for Whittle Index Methods')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot learned Whittle indices for one class
    plt.figure(figsize=(10, 6))
    
    # TD learner indices
    td_indices = td_learner.whittle_indices[0][:20]
    plt.plot(td_indices, 'o-', label='TD(λ) Learned Indices')
    
    # Q-learning indices (scaled to similar range)
    q_indices = q_whittle_indices[0]
    scaling_factor = np.max(td_indices) / np.max(q_indices) if np.max(q_indices) > 0 else 1
    plt.plot(np.arange(len(q_indices)) * (20/len(q_indices)), 
             q_indices * scaling_factor, 's-', label='Q-learning Learned Indices (scaled)')
    
    plt.xlabel('Inventory Level')
    plt.ylabel('Whittle Index')
    plt.title('Learned Whittle Indices for Class 1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
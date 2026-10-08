import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import itertools

class NStepTDWhittleLearner:
    def __init__(self, classes, n_steps=3, alpha=0.05, gamma=0.95):
        self.classes = classes
        self.K = len(classes)
        self.n_steps = n_steps
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize Whittle index approximations for each class
        self.whittle_indices = []
        for k in range(self.K):
            max_inv = int(8 * max(classes[k]['lambda'], classes[k]['mu'])) + 10
            self.whittle_indices.append(np.zeros(max_inv + 1))
        
        # Buffers for n-step learning
        self.state_buffers = [deque(maxlen=n_steps+1) for _ in range(self.K)]
        self.reward_buffers = [deque(maxlen=n_steps+1) for _ in range(self.K)]
    
    def get_whittle_index(self, k, state):
        """Get Whittle index for class k at inventory level"""
        if state < len(self.whittle_indices[k]):
            return self.whittle_indices[k][state]
        return self.whittle_indices[k][-1]  # Extrapolate
    
    def choose_action(self, state, explore_prob=0.1):
        """Choose action based on current Whittle index estimates"""
        if random.random() < explore_prob:
            return random.randint(0, self.K-1)
        
        indices = [self.get_whittle_index(k, state[k]) for k in range(self.K)]
        return np.argmax(indices)
    
    def simulate_step(self, state, action):
        """Simulate one time step and return next state and cost"""
        next_state = state.copy()
        total_cost = 0
        
        # Process each class
        for k in range(self.K):
            # Demand arrives with probability λ
            if random.random() < self.classes[k]['lambda']:
                if next_state[k] > 0:
                    next_state[k] -= 1
                else:
                    total_cost += self.classes[k]['D']  # Lost sale
            
            # Production if this class is being served
            if k == action and random.random() < self.classes[k]['mu']:
                next_state[k] += 1
            
            # Holding cost
            total_cost += self.classes[k]['h'] * next_state[k]
        
        return next_state, total_cost
    
    def calculate_class_reward(self, k, inventory, served, demand_event):
        """Calculate reward for a specific class"""
        reward = -self.classes[k]['h'] * inventory  # Negative holding cost
        
        # Stockout penalty if demand arrives and no inventory
        if demand_event and inventory == 0:
            reward -= self.classes[k]['D']
        
        # Small bonus for being served (encourages activity)
        if served:
            reward += 0.1
        
        return reward
    
    def n_step_update(self, k, tau, T):
        """Perform n-step TD update for class k"""
        if tau >= 0 and tau + self.n_steps < T:
            # Calculate n-step return
            G = 0
            for i in range(tau + 1, min(tau + self.n_steps, T) + 1):
                G += (self.gamma ** (i - tau - 1)) * self.reward_buffers[k][i]
            
            # Add bootstrapped value if not at terminal state
            if tau + self.n_steps < T:
                next_state = self.state_buffers[k][tau + self.n_steps]
                G += (self.gamma ** self.n_steps) * self.get_whittle_index(k, next_state)
            
            # Get current state and current value estimate
            current_state = self.state_buffers[k][tau]
            current_value = self.get_whittle_index(k, current_state)
            
            # Update Whittle index
            td_error = G - current_value
            if current_state < len(self.whittle_indices[k]):
                self.whittle_indices[k][current_state] += self.alpha * td_error
    
    def learn_episode(self, T=1000, explore_prob=0.1):
        """Learn from a single episode using n-step TD"""
        # Initialize state and buffers
        state = [random.randint(0, 8) for _ in range(self.K)]
        total_cost = 0
        
        # Initialize buffers
        for k in range(self.K):
            self.state_buffers[k].clear()
            self.reward_buffers[k].clear()
            self.state_buffers[k].append(state[k])
            self.reward_buffers[k].append(0)  # Initial reward
        
        for t in range(T):
            # Choose action
            action = self.choose_action(state, explore_prob)
            
            # Simulate step
            next_state, cost = self.simulate_step(state, action)
            total_cost += cost
            
            # For each class, store experience
            for k in range(self.K):
                # Check if demand occurred for this class
                demand_event = (random.random() < self.classes[k]['lambda'])
                
                # Calculate reward for this class
                reward = self.calculate_class_reward(k, state[k], k == action, demand_event)
                
                # Store experience
                self.state_buffers[k].append(next_state[k])
                self.reward_buffers[k].append(reward)
                
                # Perform n-step update if we have enough experience
                if len(self.state_buffers[k]) > self.n_steps + 1:
                    tau = t - self.n_steps
                    if tau >= 0:
                        self.n_step_update(k, tau, t+1)
            
            # Update state
            state = next_state
        
        # Update remaining states at end of episode
        for k in range(self.K):
            for tau in range(max(0, T - self.n_steps), T):
                self.n_step_update(k, tau, T)
        
        return total_cost / T
    
    def learn(self, n_episodes=1000, initial_explore=0.3, final_explore=0.01):
        """Learn over multiple episodes"""
        costs = []
        explore_rate = initial_explore
        
        for episode in range(n_episodes):
            # Decay exploration rate
            explore_rate = final_explore + (initial_explore - final_explore) * np.exp(-episode / (n_episodes / 3))
            
            # Learn from episode
            avg_cost = self.learn_episode(explore_prob=explore_rate)
            costs.append(avg_cost)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Avg Cost: {avg_cost:.4f}, Explore: {explore_rate:.4f}")
        
        return costs
    
    def get_policy(self):
        """Get the learned threshold policy"""
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

# Tree Backup algorithm for n-step learning (off-policy)
class TreeBackupWhittleLearner:
    def __init__(self, classes, n_steps=3, alpha=0.05, gamma=0.95):
        self.classes = classes
        self.K = len(classes)
        self.n_steps = n_steps
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize Whittle indices
        self.whittle_indices = []
        for k in range(self.K):
            max_inv = int(8 * max(classes[k]['lambda'], classes[k]['mu'])) + 10
            self.whittle_indices.append(np.zeros(max_inv + 1))
        
        # Behavior policy parameters (ε-greedy)
        self.epsilon = 0.1
        
        # Experience buffers
        self.state_buffers = [deque(maxlen=n_steps+1) for _ in range(self.K)]
        self.reward_buffers = [deque(maxlen=n_steps+1) for _ in range(self.K)]
        self.action_probs = [deque(maxlen=n_steps+1) for _ in range(self.K)]
    
    def behavior_policy(self, state):
        """ε-greedy behavior policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.K-1)
        
        indices = [self.get_whittle_index(k, state[k]) for k in range(self.K)]
        return np.argmax(indices)
    
    def target_policy(self, state):
        """Greedy target policy (for Tree Backup)"""
        indices = [self.get_whittle_index(k, state[k]) for k in range(self.K)]
        return np.argmax(indices)
    
    def get_whittle_index(self, k, state):
        if state < len(self.whittle_indices[k]):
            return self.whittle_indices[k][state]
        return self.whittle_indices[k][-1]
    
    def tree_backup_update(self, k, tau, T):
        """Tree Backup algorithm update for class k"""
        if tau < 0 or tau + self.n_steps >= T:
            return
        
        # Calculate Tree Backup return
        G = self.reward_buffers[k][tau + 1]
        for n in range(1, self.n_steps):
            if tau + n + 1 >= T:
                break
            
            # Calculate the expectation under target policy
            expected_value = 0
            current_state = self.state_buffers[k][tau + n]
            
            # For Whittle indices, we need to estimate the value of following policy
            # This is simplified for the Whittle index setting
            G += (self.gamma ** n) * self.reward_buffers[k][tau + n + 1]
        
        # Add bootstrapped value
        if tau + self.n_steps < T:
            next_state = self.state_buffers[k][tau + self.n_steps]
            G += (self.gamma ** self.n_steps) * self.get_whittle_index(k, next_state)
        
        # Update Whittle index
        current_state = self.state_buffers[k][tau]
        current_value = self.get_whittle_index(k, current_state)
        td_error = G - current_value
        
        if current_state < len(self.whittle_indices[k]):
            self.whittle_indices[k][current_state] += self.alpha * td_error
    
    def learn_episode(self, T=1000):
        """Learn from one episode using Tree Backup"""
        state = [random.randint(0, 8) for _ in range(self.K)]
        total_cost = 0
        
        # Initialize buffers
        for k in range(self.K):
            self.state_buffers[k].clear()
            self.reward_buffers[k].clear()
            self.action_probs[k].clear()
            self.state_buffers[k].append(state[k])
            self.reward_buffers[k].append(0)
            self.action_probs[k].append(1.0)  # Initial action probability
        
        for t in range(T):
            # Choose action using behavior policy
            action = self.behavior_policy(state)
            
            # Simulate step
            next_state, cost = self.simulate_step(state, action)
            total_cost += cost
            
            # Store experience for each class
            for k in range(self.K):
                demand_event = (random.random() < self.classes[k]['lambda'])
                reward = self.calculate_class_reward(k, state[k], k == action, demand_event)
                
                # Calculate action probability for behavior policy
                action_prob = self.epsilon/self.K + (1-self.epsilon) * int(k == np.argmax(
                    [self.get_whittle_index(i, state[i]) for i in range(self.K)]))
                
                self.state_buffers[k].append(next_state[k])
                self.reward_buffers[k].append(reward)
                self.action_probs[k].append(action_prob)
                
                # Perform Tree Backup update
                if len(self.state_buffers[k]) > self.n_steps + 1:
                    tau = t - self.n_steps
                    if tau >= 0:
                        self.tree_backup_update(k, tau, t+1)
            
            state = next_state
        
        return total_cost / T

# Comparative analysis
def compare_n_step_methods(classes, n_episodes=1000):
    """Compare different n-step TD methods"""
    results = {}
    
    # Test different n values
    for n in [1, 3, 5, 10]:
        print(f"Learning with {n}-step TD...")
        learner = NStepTDWhittleLearner(classes, n_steps=n, alpha=0.05)
        costs = learner.learn(n_episodes=n_episodes//2)
        results[f"{n}-step TD"] = costs
        
        # Get final policy
        policy = learner.get_policy()
        print(f"{n}-step TD thresholds: {policy}")
    
    # Compare with Monte Carlo (n=∞)
    print("Learning with Monte Carlo approach...")
    mc_learner = NStepTDWhittleLearner(classes, n_steps=1000, alpha=0.01)  # Large n approximates MC
    mc_costs = mc_learner.learn(n_episodes=n_episodes//2)
    results["Monte Carlo"] = mc_costs
    
    # Plot results
    plt.figure(figsize=(12, 8))
    for method, costs in results.items():
        # Smooth the curve for better visualization
        smoothed_costs = np.convolve(costs, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_costs, label=method)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Cost (smoothed)')
    plt.title('Comparison of n-step TD Methods for Whittle Index Learning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results

# Example usage
if __name__ == "__main__":
    # Define the classes
    classes = [
        {'lambda': 0.8, 'mu': 1.2, 'h': 0.5, 'D': 10.0},
        {'lambda': 0.6, 'mu': 1.0, 'h': 0.4, 'D': 8.0},
        {'lambda': 1.0, 'mu': 1.5, 'h': 0.7, 'D': 12.0}
    ]
    
    print("Comparing n-step TD methods for Whittle index learning...")
    results = compare_n_step_methods(classes, n_episodes=800)
    
    # Test a specific n-step learner in detail
    print("\nRunning detailed 3-step TD learning...")
    learner_3step = NStepTDWhittleLearner(classes, n_steps=3, alpha=0.05)
    costs_3step = learner_3step.learn(n_episodes=400)
    
    # Plot learned Whittle indices
    plt.figure(figsize=(12, 6))
    for k in range(len(classes)):
        indices = learner_3step.whittle_indices[k][:15]  # First 15 states
        plt.plot(indices, 'o-', label=f'Class {k+1}')
    
    plt.xlabel('Inventory Level')
    plt.ylabel('Whittle Index')
    plt.title('Learned Whittle Indices with 3-step TD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Show the learned threshold policy
    policy = learner_3step.get_policy()
    print(f"\nLearned threshold policy:")
    for k, threshold in policy.items():
        print(f"Class {k+1}: Serve when inventory < {threshold}")
    
    # Analyze performance
    final_cost = costs_3step[-1] if costs_3step else float('inf')
    print(f"\nFinal average cost: {final_cost:.4f}")
    
    # Compare with theoretical optimal (if known)
    # Note: In practice, we'd compare with known optimal or other baselines
    print("Note: Compare with other policies (Whittle, myopic, etc.) for performance evaluation")
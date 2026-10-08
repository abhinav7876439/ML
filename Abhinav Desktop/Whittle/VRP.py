import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

class VRPEnv:
    def __init__(self, num_customers=10, vehicle_capacity=20):
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.reset()

    def reset(self):
        self.depot = np.array([0, 0])
        self.customers = np.random.rand(self.num_customers, 2) * 10
        self.demands = np.random.randint(1, 10, size=self.num_customers)
        self.visited = np.zeros(self.num_customers, dtype=bool)
        self.vehicle_load = self.vehicle_capacity
        self.current_location = self.depot
        return self.get_state()  

    def get_state(self):
        """Returns the current environment state as a NumPy array"""
        return np.concatenate((self.current_location, [self.vehicle_load], self.visited.astype(float)))

    def step(self, action):
        """Executes an action and returns the new state, reward, and done flag."""
        if action < 0 or action >= self.num_customers or self.visited[action]:
            return self.get_state(), -10, False  # Penalty for invalid move
        
        self.visited[action] = True
        self.current_location = self.customers[action]
        self.vehicle_load -= self.demands[action]
        
        reward = -np.linalg.norm(self.current_location - self.depot)  
        done = self.visited.all()
        
        return self.get_state(), reward, done

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """Takes a state tensor and returns action probabilities."""
        x = x.unsqueeze(0).unsqueeze(0)  # Ensure proper batch and sequence dimensions
        lstm_out, _ = self.lstm(x)
        attn_weights = self.softmax(self.attn(lstm_out[:, -1, :]))  
        return attn_weights.squeeze(0)

def train_vrp():
    env = VRPEnv()
    policy = PolicyNetwork(input_dim=env.num_customers + 3, hidden_dim=128, output_dim=env.num_customers)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    num_episodes = 1000
    gamma = 0.99

    for episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        log_probs = []
        rewards = []
        done = False

        while not done:
            action_probs = policy(state)
            action_probs = torch.clamp(action_probs, min=1e-5)  # Avoid numerical instability
            
            dist = Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            state_np, reward, done = env.step(action.item())
            state = torch.tensor(state_np, dtype=torch.float32)
            rewards.append(reward)

        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        loss = -sum(log_prob * R for log_prob, R in zip(log_probs, returns))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")

# Test VRP Environment
env = VRPEnv()
print("Initial State:", env.get_state())  

if __name__ == "__main__":
    train_vrp()

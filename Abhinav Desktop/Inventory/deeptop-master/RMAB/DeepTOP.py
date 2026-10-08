

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()


class DeepTOP_RMAB(object):
    # nb_arms: number of arms
    # state_dims: a list of state dimensions, one for each arm
    # action_dims: a list of action space dimensions, one for each arm
    # hidden: a list of number of neurons in each hidden layer
    def __init__(self, nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args):
        self.nb_arms = nb_arms
        self.budget = budget
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.actors = []
        self.actor_optims = []
        self.critics = []
        self.critic_targets = []
        self.critic_optims = []
        self.memories = []
        self.random_processes = []
        self.s_t = []
        self.a_t = []
        # Create Actor and Critic Networks, one for each arm
        for arm in range(nb_arms):
            self.actors.append(Actor(self.state_dims[arm], 1, hidden))  # input is state, output is threshold
            self.actor_optims.append(Adam(self.actors[arm].parameters(), lr=args.prate))

            self.critics.append(
                Critic(self.state_dims[arm] + 1, 1, hidden))  # input is state and lambda, output is Q value
            self.critic_targets.append(Critic(self.state_dims[arm] + 1, 1, hidden))
            self.critic_optims.append(Adam(self.critics[arm].parameters(), lr=args.rate))
            
            hard_update(self.critic_targets[arm], self.critics[arm])

            # Create replay buffer
            self.memories.append(SequentialMemory(limit=args.rmsize, window_length=args.window_length))
            self.random_processes.append(
                OrnsteinUhlenbeckProcess(size=action_dims[arm], theta=args.ou_theta, mu=args.ou_mu,
                                         sigma=args.ou_sigma))
            self.s_t.append(None)  # Most recent state
            self.a_t.append(None)  # Most recent action

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon


        self.epsilon = 1.0
        self.is_training = True

        if self.device == torch.device('cuda'): 
            self.cuda()

    def update_policy(self):
        for arm in range(self.nb_arms):
            # Sample batch
            state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memories[arm].sample_and_split(self.batch_size)

            price_batch = np.random.uniform(-1., 1., size=self.batch_size).reshape(self.batch_size,1)
            next_action_batch = []

            net_reward_batch = reward_batch - price_batch * action_batch

            # convert all batches to tensors
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            terminal_batch = torch.FloatTensor(terminal_batch).to(self.device)
            price_batch = torch.FloatTensor(price_batch).to(self.device)
            net_reward_batch = torch.FloatTensor(net_reward_batch).to(self.device)

            with torch.no_grad():
                critic_plus = self.critic_targets[arm]([next_state_batch, 
                                                    price_batch, 
                                                    to_tensor(np.ones((self.batch_size, 1), dtype=int)).to(self.device)]).cpu()
                critic_minus = self.critic_targets[arm]([next_state_batch,
                                                    price_batch,
                                                    to_tensor(np.zeros((self.batch_size, 1), dtype=int)).to(self.device)]).cpu()
            
                next_action_batch = torch.FloatTensor(torch.clamp(torch.sign(critic_plus - critic_minus), min=0.0)).to(self.device)

                # Prepare for the target q batch
                next_q_values = self.critic_targets[arm]([next_state_batch, price_batch, next_action_batch])

                target_q_batch = net_reward_batch + self.discount * next_q_values

            # Critic update
            self.critics[arm].zero_grad()

            q_batch = self.critics[arm]([state_batch, price_batch, action_batch])

            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            self.critic_optims[arm].step()

            # Actor update
            self.actors[arm].zero_grad()

            
            q_diff_batch = self.critics[arm]([state_batch, self.actors[arm](state_batch),
                                              to_tensor(np.ones((self.batch_size, 1), dtype=int)).to(self.device)]) - \
                           self.critics[arm]([state_batch, self.actors[arm](state_batch),
                                              to_tensor(np.zeros((self.batch_size, 1), dtype=int)).to(self.device)])
          
            q_diff_batch = q_diff_batch.detach().cpu().numpy()


            policy_loss = -to_tensor(q_diff_batch).to(self.device) * self.actors[arm](state_batch)
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optims[arm].step()

            # Target update
            soft_update(self.critic_targets[arm], self.critics[arm], self.tau)

    def eval(self):
        for arm in range(self.nb_arms):
            self.actors[arm].eval()
            self.critics[arm].eval()
            self.critic_targets[arm].eval()

    def cuda(self):
        torch.cuda.set_device(1) # specify which gpu to train on
        for arm in range(self.nb_arms):
            self.actors[arm].cuda()
            self.critics[arm].cuda()
            self.critic_targets[arm].cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            for arm in range(self.nb_arms):
                self.memories[arm].append(self.s_t[arm], self.a_t[arm], r_t[arm], done[arm])
                self.s_t[arm] = s_t1[arm]

    def random_action(self):
        indices = []
        for arm in range(self.nb_arms):
            indices.append(np.random.uniform(-1., 1.))
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(-2)  # Create an additional item to handle the case when budget = nb_arms
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        return actions

    def select_action(self, s_t, decay_epsilon=True):
        indices = []
        for arm in range(self.nb_arms):
            indices.append(self.actors[arm].forward(torch.FloatTensor(self.s_t[arm]).to(self.device)).cpu().detach().numpy()[0])
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(
            sort_indices[self.nb_arms - 1] - 2)  # Create an additional item to handle the case when budget = nb_arms
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        if decay_epsilon:
            self.epsilon -= self.depsilon
        return actions

    def reset(self, obs):
        self.s_t = obs
        for arm in range(self.nb_arms):
            self.random_processes[arm].reset_states()


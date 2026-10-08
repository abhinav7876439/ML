'''
algorithm implemented form:
https://github.com/sfujim/TD3
'''


import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()


class TD3(object):

    def __init__(self, nb_states, nb_actions, hidden, args):


        self.nb_states = nb_states
        self.nb_actions = nb_actions

        self.policy_freq = 2 # default value from reference code. 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Create Actor and Critic Network
        self.actor = Actor(self.nb_states + 1, self.nb_actions, hidden)
        self.actor_target = Actor(self.nb_states + 1, self.nb_actions, hidden)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)
        

        self.critic_one = Critic(self.nb_states + 1, self.nb_actions, hidden)
        self.critic_target_one = Critic(self.nb_states + 1, self.nb_actions, hidden)
  
        self.critic_two = Critic(self.nb_states + 1, self.nb_actions, hidden)
        self.critic_target_two = Critic(self.nb_states + 1, self.nb_actions, hidden)

        self.critic_optim  = Adam(list(self.critic_one.parameters()) + list(self.critic_two.parameters()), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target_one, self.critic_one)
        hard_update(self.critic_target_two, self.critic_two)

        self.total_it = 0 
                

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        
        if self.device == torch.device('cuda'): 
            self.cuda()


    def update_policy(self):
        self.total_it += 1 

        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        vector_batch = []  # The batch of vector state
        scalar_batch = []  # The batch of scalar state
        next_vector_batch = []
        next_scalar_batch = []
        for i in range(self.batch_size):
            vector_batch.append(list(state_batch[i]))
            next_vector_batch.append(list(next_state_batch[i]))
            scalar_batch.append([vector_batch[i].pop(0)])
            next_scalar_batch.append([next_vector_batch[i].pop(0)])
        
        vector_batch = torch.FloatTensor(np.array(vector_batch)).to(self.device)
        next_vector_batch = torch.FloatTensor(np.array(next_vector_batch)).to(self.device)
        scalar_batch = torch.FloatTensor(np.array(scalar_batch)).to(self.device)
        next_scalar_batch = torch.FloatTensor(np.array(next_scalar_batch)).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        state_batch = torch.FloatTensor(state_batch).to(self.device)

        with torch.no_grad():
            
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5) # 0.2 is policy noise. clamp noise between -0.5, 0.5
            
            next_action_batch = (self.actor_target(next_state_batch) + noise).clamp(0., 1.0)
            

            target_q_one = self.critic_target_one([
                next_vector_batch,
                next_scalar_batch,
                next_action_batch])

            target_q_two = self.critic_target_two([
                next_vector_batch,
                next_scalar_batch,
                next_action_batch])
            
            target_q_batch = torch.min(target_q_one, target_q_two)

            target_q_batch = reward_batch + self.discount*to_tensor(terminal_batch.astype(np.float))*target_q_batch


        # Critic update 

        Q_one = self.critic_one([
        vector_batch,
        scalar_batch,
        action_batch])

        Q_two = self.critic_two([
        vector_batch,
        scalar_batch,
        action_batch])

        
        value_loss = F.mse_loss(Q_one, target_q_batch) + F.mse_loss(Q_two, target_q_batch)

        self.critic_optim.zero_grad() 
        value_loss.backward() 
        self.critic_optim.step()


        if self.total_it % self.policy_freq == 0:
            # Actor update
            self.actor_optim.zero_grad()

            policy_loss = -self.critic_one([
            vector_batch, 
            scalar_batch, 
            self.actor(state_batch)
            ])

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()


            # Target update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target_one, self.critic_one, self.tau)
            soft_update(self.critic_target_two, self.critic_two, self.tau)


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic_one.eval()
        self.critic_target_one.eval()
        self.critic_two.eval()
        self.critic_target_two.eval()

    def cuda(self):
        torch.cuda.set_device(0) # specify which gpu to train on 
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic_one.cuda()
        self.critic_target_one.cuda()
        self.critic_two.cuda()
        self.critic_target_two.cuda()


    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1., self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):

        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)

        action += self.is_training*np.random.normal(0., 1.*0.1) # std is max action * exploration noise of 0.1
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon 

        self.a_t = action 

        return action 


    def reset(self, obs):
        self.s_t = obs


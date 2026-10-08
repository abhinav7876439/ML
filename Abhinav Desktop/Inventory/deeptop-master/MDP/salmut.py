

import random

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()



class SALMUT(object):

    def __init__(self, state_dim, action_dim, args, m, B, customer_classes):

        self.num_scalar_state = m+B+1 
        self.num_vector_state = customer_classes 
        self.scalar_state_counter = np.zeros(self.num_scalar_state)  
        self.value_function = np.zeros(self.num_scalar_state) 

        self.thresholds = np.zeros(self.num_vector_state) 
        self.fixed_state = int(np.floor(self.num_scalar_state/2)) 
        
        self.myRandomPRNG = np.random.RandomState(args.seed)

        self.W = m+B 
 
        self.s_t = None
        self.a_t = None

        self.iterator = 0 

    def _h_s(self, scalar_state):
        return 0.1*(scalar_state)**2

    def _a_n(self, n):
        denom = (np.floor(n/100) + 2)**0.6
        return (1 / denom)

    def _b_n(self, n):
        return (10/n)
    

    def _update_value_func(self, vector, next_vector, scalar, next_scalar):

        # value function update with a(n)
        self.scalar_state_counter[scalar] += 1
        
        first_val = (1 - self._a_n(self.scalar_state_counter[scalar]))*self.value_function[scalar]
        second_val = (self.reward + self.value_function[next_scalar] - self.value_function[self.fixed_state])

        self.value_function[scalar] = first_val + self._a_n(self.scalar_state_counter[scalar])*second_val


    def _update_threshold_values(self, vector, next_vector, scalar, next_scalar):
    
        # threshold update with b(n)

        beta = self.myRandomPRNG.choice([0,1])
        alpha = self.myRandomPRNG.choice([0,1])
        
        value = torch.autograd.Variable(torch.Tensor([scalar - self.thresholds[vector] - 0.5]), requires_grad=True)
        y = torch.sigmoid(value)
        y.backward()
        gradient = value.grad.numpy()[0]


        threshold_first_val = self.thresholds[vector]
        threshold_second_val = ((-1)**beta)*(-beta * self._h_s(scalar) + (1-beta)*(self.reward))

        threshold_third_val = ((-1)**alpha)*self.value_function[next_scalar]

        threshold_fourth_val = self._b_n(self.iterator) * gradient


        self.thresholds[vector] = threshold_first_val + threshold_fourth_val*(threshold_second_val + threshold_third_val)
        
        
        if vector == 0: # we subtract 1 for correct indexing
            self.thresholds[vector] = max(0, min(self.W, self.thresholds[vector]))
        else:
            self.thresholds[vector] = max(0, max(self.thresholds[vector], self.thresholds[vector-1]))

        
        
        for i in range(vector+1, self.num_vector_state):
            self.thresholds[i] = max(self.thresholds[i], self.thresholds[i-1])
        

    def update_policy(self):
        self.iterator += 1 

        vector=self.s_t.copy()
        scalar=vector.pop(0)
        next_vector=self.s_t1.copy()
        next_scalar=next_vector.pop(0)
        

        self._update_value_func(vector=vector[0]-1, next_vector=next_vector[0]-1,\
            scalar=scalar, next_scalar=next_scalar)
        self._update_threshold_values(vector=vector[0]-1, next_vector=next_vector[0]-1,\
            scalar=scalar, next_scalar=next_scalar)


    def select_action(self, s_t):
        vector = s_t.copy()
        scalar = vector.pop(0)

        threshold = self.thresholds[vector[0]-1] #vector[0] - 1 for correct indexing since indexing starts from 0.

        if threshold > scalar:
            action = [1]
        else:
            action = [0]
        
        self.a_t = action 

        return action 


    def observe(self, r_t, s_t, s_t1, done):
        if self.is_training:
            
            self.s_t = s_t
            self.s_t1 = s_t1
            self.reward = r_t


    def random_action(self):
        action = [random.randint(0, 1)]
        self.a_t = action
        return action


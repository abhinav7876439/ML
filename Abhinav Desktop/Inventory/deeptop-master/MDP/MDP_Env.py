
import gym
import math
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
from gym import spaces
from random_process import OrnsteinUhlenbeckProcess


class chargingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment for charging EVs 
    '''

    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, seed, min_charge, max_charge, min_deadline, max_deadline, theta, mu, sigma, dt, x0):
        super(chargingEnv, self).__init__()
        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_deadline = min_deadline
        self.max_deadline = max_deadline
        self.ou_process = OrnsteinUhlenbeckProcess(theta, mu, sigma, dt, x0)

        self.x = self.ou_process.sample()
        self.charge = self.myRandomPRNG.randint(self.min_charge, self.max_charge)
        self.deadline = self.myRandomPRNG.randint(self.min_deadline, self.max_deadline)

        self.action_space = spaces.Discrete(2)



    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''
        if self.charge <= 0:
            action = 0
        reward = 0
        if action == 1:
            reward = 1 - self.x
            self.deadline = self.deadline - 1
            self.charge = self.charge - 1
        elif action == 0:
            reward = 0
            self.deadline = self.deadline - 1

        if self.deadline <= 0:  # A new vehicle arrives
            reward = reward - 0.2 * self.charge * self.charge
            self.charge = self.myRandomPRNG.randint(self.min_charge, self.max_charge)
            self.deadline = self.myRandomPRNG.randint(self.min_deadline, self.max_deadline)

        self.x = self.ou_process.sample()
        nextState = [self.x, self.charge, self.deadline]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.ou_process.reset_states()
        self.x = self.ou_process.sample()
        self.charge = self.myRandomPRNG.randint(self.min_charge, self.max_charge)
        self.deadline = self.myRandomPRNG.randint(self.min_deadline, self.max_deadline)
        initialState = [self.x, self.charge, self.deadline]

        return initialState

class inventoryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment for inventory management
    parameters are: 
    cap: capacity of the warehouse
    order_size: size of an order
    demand_list: a list of seasonal demand
    selling_price: selling price
    '''

    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, seed, cap, order_size, demand_list, selling_price):
        super(inventoryEnv, self).__init__()
        self.seed = seed
        self.myRandomPRNG = np.random.RandomState(self.seed)
        self.cap = cap
        self.order_size = order_size
        self.demand_list = demand_list
        self.selling_price = selling_price

        self.t = 0
        self.inventory = self.order_size
        self.arriving = 0  # 1 if there is an order arriving at the end of this slot

        self.action_space = spaces.Discrete(2)



    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''

        # First, determine demand in this slot and calculate reward
        this_demand = self.myRandomPRNG.poisson(self.demand_list[self.t])
        reward = self.selling_price * min(this_demand, self.inventory)
        self.inventory = self.inventory - min(this_demand, self.inventory)

        # Next, incur holding cost, update order arrival and current time
        reward = reward - self.inventory
        if action > 0:
            self.inventory = min(self.inventory + self.order_size, self.cap)
        self.t = (self.t + 1) % len(self.demand_list)

        # Finally, generate state
        nextState = [self.inventory, self.t]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.t = 0
        self.inventory = self.order_size
        self.arriving = 0  # 1 if there is an order arriving at the end of this slot

        initialState = [self.inventory, self.t]

        return initialState


class MakeToStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, seed, customer_classes, m, B, mu):
        super(MakeToStockEnv, self).__init__()
        self.seed = seed 
        self.customer_classes = customer_classes
        self.m = m 
        self.B = B 
        self.mu = mu 
        self.myRandomPRNG = np.random.RandomState(self.seed)
        
        self.k_choices = np.arange(0, self.customer_classes+1)

        self.R = np.linspace(200, 10, num=self.customer_classes)

    def _h_s(self, scalar_state):
        return 0.1*(scalar_state)**2


    def _calRewardAndState(self, action):

        if action == 0:
            reward = -1*self._h_s(self.scalar_state)
            self.scalar_state = self.scalar_state
        elif action == 1:

            if self.scalar_state == (self.m + self.B):
                reward = -1*self._h_s(self.scalar_state)
                self.scalar_state = self.scalar_state

            else:
                reward = self.R[self.vector_state-1] - self._h_s(self.scalar_state) #self.vector_state - 1 for correct indexing
                self.scalar_state = min(self.m + self.B, self.scalar_state + 1)
        else:
            print(f'wrong action value. exiting...')
            exit(2)
    
        k = 0
        while (k <= 1): 
            w = min(self.m, self.scalar_state)
            probabilites = np.append( (self.mu*w)/(self.mu*w + self.customer_classes), np.repeat(1 / (self.mu*w + self.customer_classes), self.customer_classes))
            
            k = self.myRandomPRNG.choice(self.k_choices, p=probabilites)
            
            if k == 0:
                self.scalar_state = max(0, self.scalar_state - 1)

        self.vector_state = k 

        nextState = [self.scalar_state, self.vector_state]

        return nextState, reward 


    def step(self, action):

        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        done = False
        info = {}

        return nextState, reward, done, info 


    def reset(self):
        self.scalar_state = self.myRandomPRNG.randint(0, self.m+self.B+1)
        self.vector_state = self.myRandomPRNG.randint(1, self.customer_classes+1)

        initialState = [self.scalar_state, self.vector_state]

        return initialState


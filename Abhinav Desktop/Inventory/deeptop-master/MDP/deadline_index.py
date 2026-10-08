
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

class Deadline_Index(object):
    # state_dim: the dimension of the vector state. IMPORTANT: not including the scalar state
    # action_dim: the dimension of actions. This should be 1
    # hidden: a list of number of neurons in each hidden layer
    def __init__(self, state_dim, action_dim, hidden, args):
        self.is_training = True
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.myRandomPRNG = random.Random(args.seed)

    def update_policy(self):
        # Do nothing
        return

    def eval(self):
        # Do nothing
        return

    def cuda(self):
        # Do nothing
        return

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.s_t = s_t1

    def random_action(self):
        action = [self.myRandomPRNG.randint(0, 1)]
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        vector = s_t.copy()
        scalar = vector.pop(0)
        if vector[0] < vector[1]:  # charge < deadline
            if scalar < 1.0:
                action = [1]
            else:
                action = [0]
        else:
            if scalar < 1.0 + 0.2 * (2 * vector[0] - 1):
                action = [1]
            else: action = [0]
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs

    def seed(self, s):
        return


import os
import torch
import random
import argparse
from copy import deepcopy
import itertools
import numpy as np
import operator

import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from MDP_Env import MakeToStockEnv
from salmut import SALMUT




def initializeEnv():
    global env, m, B, customer_classes

    env = MakeToStockEnv(seed=5723,customer_classes=customer_classes,m=m,B=B,mu=4)


def resetEnvs():
    global state, env
    state = env.reset()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SALMUT for the make-to-stock environment')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=458472, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')


    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    m = 50
    B = 50 
    customer_classes = 50

    env = None
    state = None
    state_dim = 1
    action_dim = 1
    initializeEnv()
    #initialize agent

    agent = SALMUT(state_dim, action_dim, args, m=m, B=B, customer_classes=customer_classes)

    resetEnvs()
  
    cumulative_reward = 0

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    iteration = 0
    num_step = 0

    for t in range(260001):
        if t % 13000 == 0:
            iteration = iteration + 1
            num_step = 0
            print(f'iteration {iteration}')
            agent = SALMUT(state_dim, action_dim, args, m=m, B=B, customer_classes=customer_classes)

            resetEnvs()

        agent.is_training = True
        num_step = num_step + 1

        if num_step <= args.warmup:
            action = agent.random_action()
        elif random.uniform(0, 1.0) < 0.05:
            action = agent.random_action()
        else:
            action = agent.select_action(state)

        # env response with next_state, reward, terminate_info
        next_state, reward, done, info = env.step(action[0])
        next_state = deepcopy(next_state)

        # agent observes and update policy
        agent.observe(r_t=reward, s_t=state, s_t1=next_state, done=done)
        if num_step > args.warmup:
            cumulative_reward = cumulative_reward + reward
            
            agent.update_policy()
            

            if( (num_step-args.warmup)%100 == 0 ):
                print(f'{cumulative_reward/100}')
                cumulative_reward = 0
                

        
        
        state = deepcopy(next_state)

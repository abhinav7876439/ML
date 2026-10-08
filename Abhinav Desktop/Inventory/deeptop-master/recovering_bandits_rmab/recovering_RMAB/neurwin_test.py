

import numpy as np

import torch
import torch.nn as nn

from model import Actor



class NeurWIN_Wrapper(object):
    # nb_arms: number of arms
    # state_dims: a list of state dimensions, one for each arm
    # action_dims: a list of action space dimensions, one for each arm
    # hidden: a list of number of neurons in each hidden layer
    def __init__(self, nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args):
        self.nb_arms = nb_arms
        self.budget = budget
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.hidden = hidden
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.actors = []
        self.s_t = []
        self.a_t = []
        # Create Actor and Critic Networks, one for each arm
        for arm in range(nb_arms):
            self.s_t.append(None)  # Most recent state
            self.a_t.append(None)  # Most recent action

        if self.device == torch.device('cuda'): 
            self.cuda()

    def get_agents(self, directory, episode):
        
        agents = []
        
        for x in range(self.nb_arms):
            agent = Actor(self.state_dims[x], 1, self.hidden)
            agent.load_state_dict(torch.load(directory+(f'arm_{x}_trainedNumEpisodes_{episode}/trained_model.pt')))
            agent.eval()
            agents.append(agent)
       
        self.actors = agents
        
    def cuda(self):
        torch.cuda.set_device(1) # specify which gpu to train on
        for arm in range(self.nb_arms):
            self.actors[arm].cuda()


    def random_action(self):
        indices = []
        for arm in range(self.nb_arms):
            indices.append(np.random.uniform(-10., 10.))
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(-2)  # Create an additional item to handle the case when budget = nb_arms
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
               
            else:
                actions.append(0)
                
        return actions

    def select_action(self, s_t):
        indices = []
        for arm in range(self.nb_arms):
            indices.append(self.actors[arm].forward(torch.FloatTensor(s_t[arm]).to(self.device)).cpu().detach().numpy()[0])
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(
            sort_indices[self.nb_arms - 1] - 2)  # Create an additional item to handle the case when budget = nb_arms
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                
            else:
                actions.append(0)

        return actions


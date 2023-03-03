# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2023 10:34 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, actor_lr, input_dims, n_actions, name):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.pi = nn.Linear(64, n_actions)
        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = T.tanh(self.pi(x))

        return action


class CriticNetwork(nn.Module):
    def __init__(self, critic_lr, input_dims, n_agents, n_actions, name):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims+(n_agents*n_actions), 64)
        self.fc2 = nn.Linear(64, 64)
        self.q = nn.Linear(64, 1)  # output of critic network is always 1.
        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

# -*- coding: utf-8 -*-
"""
@Time    : 1/4/2024 2:26 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import math
from torch.distributions import Normal
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Single_ActorNetwork(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(Single_ActorNetwork, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())

    def forward(self, cur_state):
        own_obs = self.own_fc(cur_state[0])
        own_grid = self.own_grid(cur_state[1])
        merge_obs_grid = torch.cat((own_obs, own_grid), dim=1)
        merge_feature = self.merge_feature(merge_obs_grid)
        out_action = self.act_out(merge_feature)
        return out_action


class Single_CriticNetwork(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions):
        super(Single_CriticNetwork, self).__init__()
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64+64, 256), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(256, 1))

    def forward(self, single_state, single_action):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction = self.SA_fc(obsWaction)
        own_grid = self.SA_grid(single_state[1])
        merge_obs_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        merge_feature = self.merge_fc_grid(merge_obs_grid)
        q = self.out_feature_q(merge_feature)
        return q

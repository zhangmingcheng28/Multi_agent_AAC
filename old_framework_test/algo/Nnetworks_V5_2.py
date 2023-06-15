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
import torch
import numpy as np
import math


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.K = nn.Linear(input_dim, input_dim)
        self.Q = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, feature_dim = x.shape

        # Apply linear transformations
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))
        attention_weights = nn.functional.softmax(scores, dim=-1)

        # Apply attention to values
        values = torch.matmul(attention_weights, V)

        return values


class ActorNetwork(nn.Module):
    def __init__(self, actor_lr, actor_obs, n_actions, max_nei_num,  name):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork, self).__init__()

        self.n_heads = 3
        self.single_head_dim = int((64+64+64) / self.n_heads)

        # self.own_fc = nn.Sequential(nn.Linear(actor_obs[0], 64), nn.ReLU())
        self.own_fc = nn.Sequential(nn.Linear(actor_obs[0], 512), nn.ReLU())

        # perform a self-attention for own obs_grids, actor_obs[1], assume actor_obs = [6, 6, 6]

        self.grid_K = nn.Linear(actor_obs[1], 64)
        self.grid_Q = nn.Linear(actor_obs[1], 64)
        self.grid_V = nn.Linear(actor_obs[1], 64)

        self.own_grid = nn.Sequential(nn.Linear(actor_obs[1], 64), nn.ReLU())

        self.surr_drone = nn.Sequential(nn.Linear(max_nei_num * actor_obs[2], 64), nn.ReLU(),
                                        nn.Linear(64, 64), nn.ReLU())
        # self-attention for 2D grids that has arbitrary length after flattened, with 2 head.

        # use attention here
        self.combine_att_xe = nn.Sequential(nn.Linear(64+64+64, 128), nn.ReLU())
        # NOTE: For the com_k,q,v here, they are used for "single head" attention calculation, so we only use
        # dimension of the single_head_dim, in the linear transformation.

        self.com_k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.com_q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.com_v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)

        self.multi_att_out = nn.Linear(self.n_heads * self.single_head_dim, 128)

        self.action_out = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())

        self.action_out_V3 = nn.Sequential(nn.Linear(64+64+64, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        self.action_out_V4 = nn.Sequential(nn.Linear(64+64, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        self.action_out_V5 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, n_actions), nn.Tanh())
        self.action_out_V5_1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, n_actions))

        # attention for surr_drones
        # the number here is 64 because we need to align with the output neural number of the "surr_done" NN
        self.k = nn.Linear(64, 64, bias=False)
        self.q = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)

        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # own_e = self.own_fc(state[0])
        own_e = self.own_fc(state)

        # when padding is used for drone's own grid
        # env_e = self.own_grid(state[1])

        # intru_e = self.surr_drone(state[2].view(state[0].shape[0], -1))

        # concat = torch.cat((own_e, env_e, intru_e), dim=1)
        # concat = torch.cat((own_e, intru_e), dim=1)

        # action_out = self.action_out_V4(concat)
        action_out = self.action_out_V5(own_e)
        # action_out_1 = self.action_out_V5_1(own_e)

        return action_out


class CriticNetwork(nn.Module):
    def __init__(self, critic_lr, critic_obs, n_agents, n_actions, name):
        super(CriticNetwork, self).__init__()
        # in critic network we should use multi-head attention mechanism to help to capture more complex relationship
        # between different inputs, in the context of this paper, the input consists of many drone's states as well as
        # their actions. This is two group of inputs, therefore my hypothesis is that using multi-head attention is
        # better here.

        # critic_obs[0] is sum of all agent's own observed states
        # critic_obs[1] is sum of all agent's observed grid maps
        # critic_obs[3] is sum of all agent's action taken

        self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs[0]*n_agents, 1024), nn.ReLU())  # may be here can be replaced with another attention mechanism
        self.sum_env_fc = nn.Sequential(nn.Linear(critic_obs[1]*n_agents, 256), nn.ReLU())
        # for surrounding agents' encoding, for each agent, we there are n-neighbours, each neighbour is represented by
        # a vector of length = 6. Before we put into an experience replay, we pad it up to max_num_neigh * 6 array.
        # so, one agent will have an array of max_num_neigh * 6, after flatten, then for one batch, there are a total of
        # n_agents exist in the airspace, therefore, the final dimension will be max_num_neigh * 6 * max_num_neigh.
        self.sum_sur_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents*n_agents, 256), nn.ReLU())
        # critic attention for overall sur_neighbours with overall own_state
        self.sum_k = nn.Linear(256, 256, bias=False)
        self.sum_q = nn.Linear(256, 256, bias=False)
        self.sum_v = nn.Linear(256, 256, bias=False)

        self.n_heads = 3
        self.single_head_dim = int((256+256+256) / self.n_heads)
        self.com_k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.com_q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.com_v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.multi_att_out = nn.Sequential(nn.Linear(self.n_heads * self.single_head_dim + n_agents * n_actions, 128),
                                           nn.ReLU())

        # self.combine_env_fc = nn.Sequential(nn.Linear(256+256+256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
        #                                     nn.Linear(128, 64), nn.ReLU())
        self.combine_env_fc = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                            nn.Linear(128, 64), nn.ReLU())

        self.combine_all = nn.Sequential(nn.Linear(1024+n_agents * n_actions, 512), nn.ReLU(),
                                         nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

        self.sum_agents_action_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents, 256), nn.ReLU())

        self.multi_attention = nn.MultiheadAttention(embed_dim=256+256, num_heads=2)  # 1st input is the sum of neurons from actions and combined states encoding

        # the input of this judgement layer is 256+256 because this is right after the multi-head attention layer
        # the output dimension of the multi-head attention is default to be the dimension of the "embed_dim"
        self.judgement_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, actor_obs):  # state[0] is sum of all agent's own observed states, state[1] is is sum of all agent's observed grid maps
        # NOTE: for critic network, we must include all individual drone's action (which is actor_obs),
        # as dimension: (batch X actor's combined action)
        # NOTE: here, this detach() is a must, in order to avoid "inplace operation during backpropagation"

        # actor_obs_detached = actor_obs.detach()
        # combine_obs = actor_obs_detached.view(actor_obs_detached.shape[0], -1)

        # combine_obs = torch.randn(64, 1, 10)
        sum_own_e = self.sum_own_fc(state).squeeze(1)
        # sum_own_e = self.sum_own_fc(state[0]).squeeze(1)
        #sum_env_e = self.sum_env_fc(state[1])
        # sum_sur_nei = self.sum_sur_fc(state[2])

        # env_concat = torch.cat((sum_own_e, sum_env_e, sum_sur_nei), dim=2).squeeze(1)
        # env_concat = torch.cat((sum_own_e, sum_sur_nei), dim=2).squeeze(1)
        # env_encode = self.combine_env_fc(env_concat)
        # entire_comb = torch.cat((env_encode, combine_obs), dim=1)
        entire_comb = torch.cat((sum_own_e, actor_obs), dim=1)
        q = self.combine_all(entire_comb)
        return q

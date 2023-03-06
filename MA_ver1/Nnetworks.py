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


class ActorNetwork(nn.Module):
    def __init__(self, actor_lr, actor_obs, n_actions, name):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_obs[0], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Flatten(),
                                      nn.Linear(actor_obs[1], 64), nn.ReLU())
        self.surr_drone = nn.Sequential(nn.Linear(actor_obs[2], 64), nn.ReLU(),
                                        nn.Linear(64, 64), nn.ReLU())
        self.combine_fc = nn.Sequential(nn.Linear(64+64+64, 128), nn.ReLU(),
                                        nn.Linear(128, 128), nn.ReLU(),
                                        nn.Linear(128, n_actions), nn.Tanh())
        # attention
        self.k = nn.Linear(64, 64, bias=False)  # the number here is 64 because we need to align with the output neural number of the "surr_done" NN
        self.q = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)

        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        own_e = self.own_fc(state[0])
        env_e = self.own_grid(state[1])
        intru_e = self.surr_drone(state[2])

        # mask attention embedding
        q = self.q(own_e)
        k = self.k(intru_e)
        v = self.v(intru_e)
        mask = state[2].mean(axis=2, keepdim=True).bool()
        score = torch.bmm(k, q.unsqueeze(axis=2))
        score_mask = score.clone()  # clone操作很必要
        score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度

        alpha = F.softmax(score_mask / np.sqrt(k.size(-1)), dim=1)
        alpha_mask = alpha.clone()
        alpha_mask[~mask] = 0
        v_att = torch.sum(v * alpha_mask, axis=1)

        concat = torch.cat((own_e, env_e, v_att),dim=1)
        action_out = self.combine_fc(concat)
        return action_out


class CriticNetwork(nn.Module):
    def __init__(self, critic_lr, critic_obs, n_agents, n_actions, name):
        super(CriticNetwork, self).__init__()
        # in critic network we should use multi-head attention mechanism to help to capture more complex relationship
        # between different inputs, in the context of this paper, the input consists of many drone's states as well as
        # there actions. This is two group of inputs, therefore my hypothesis is that using multi-head attention is
        # better here.

        # critic_obs[0] is sum of all agent's own observed states
        # critic_obs[1] is sum of all agent's observed grid maps
        # critic_obs[3] is sum of all agent's action taken


        self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs[0]*n_agents, 256), nn.ReLU())
        self.sum_env_fc = nn.Sequential(nn.Flatten(), nn.Linear(critic_obs[1]*n_agents, 256), nn.ReLU())
        self.combine_own_env_fc = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU())

        self.sum_agents_action_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents, 256), nn.ReLU())

        self.multi_attention = nn.MultiheadAttention(embed_dim=256+256, num_heads=2)  # 1st input is the sum of neurons from actions and combined states encoding

        # the input of this judgement layer is 256+256 because this is right after the multi-head attention layer
        # the output dimension of the multi-head attention is default to be the dimension of the "embed_dim"
        self.judgement_fc = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU(),
                                          nn.Linear(256, 1))

        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):  # state[0] is sum of all agent's own observed states, state[1] is is sum of all agent's observed grid maps
        sum_own_e = self.sum_own_fc(state[0])
        sum_env_e = self.sum_env_fc(state[1])
        combine_state_e = self.combine_own_env_fc([sum_own_e, sum_env_e])
        sum_action_e = self.sum_agents_action_fc(state[2])
        multi_dim_out = self.multi_attention([combine_state_e, sum_action_e])
        q = self.judgement_fc(multi_dim_out)

        return q

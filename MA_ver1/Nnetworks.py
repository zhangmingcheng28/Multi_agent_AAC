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
    def __init__(self, actor_lr, actor_obs, n_actions, name):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork, self).__init__()

        self.n_heads = 3
        self.single_head_dim = int((64+64+64) / self.n_heads)

        self.own_fc = nn.Sequential(nn.Linear(actor_obs[0], 64), nn.ReLU())

        # perform a self-attention for own obs_grids, actor_obs[1], assume actor_obs = [6, 6, 6]

        self.grid_K = nn.Linear(actor_obs[1], 64)
        self.grid_Q = nn.Linear(actor_obs[1], 64)
        self.grid_V = nn.Linear(actor_obs[1], 64)

        self.own_grid = nn.Sequential(nn.Linear(actor_obs[1], 64), nn.ReLU())

        self.surr_drone = nn.Sequential(nn.Linear(actor_obs[2], 64), nn.ReLU(),
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
        own_e = self.own_fc(state[0])

        # when padding is used for drone's own grid
        env_e = self.own_grid(state[1])

        # # apply one-head self-attention
        # grid_K = self.grid_K(state[1])
        # grid_Q = self.grid_Q(state[1])
        # grid_V = self.grid_V(state[1])
        # grid_feature_dim = len(state[1])
        # # Compute attention scores for grids
        # scores = torch.matmul(grid_Q, grid_K.transpose(-2, -1)) / torch.sqrt(torch.tensor(grid_feature_dim, dtype=torch.float32))
        # attention_weights = nn.functional.softmax(scores, dim=-1)
        #
        # # Apply attention to values for grids
        # env_e = torch.matmul(attention_weights, grid_V)

        intru_e = self.surr_drone(state[2]).unsqueeze(0)

        # mask attention embedding for surrounding drones
        q = self.q(own_e)
        k = self.k(intru_e)
        v = self.v(intru_e)
        mask = intru_e.mean(axis=2, keepdim=True).bool()  # this line requires the input vector to have a batch size
                                                           # at left most dimension
        score = torch.bmm(k, q.unsqueeze(axis=2))
        score_mask = score.clone()  # clone操作很必要
        score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度

        alpha = F.softmax(score_mask / np.sqrt(k.size(-1)), dim=1)
        alpha_mask = alpha.clone()
        alpha_mask[~mask] = 0
        v_att = torch.sum(v * alpha_mask, axis=1)  # 1x64

        # 1 more layer of attention (not consider self attention, as input and output sequence are not the same)
        # I have 3 inputs (embeded): own_e, env_e, and v_att

        concat = torch.cat((own_e, env_e, v_att), dim=1)

        raw_k = concat
        raw_q = concat
        raw_v = concat

        batch_size, inputDim = concat.shape
        seq_length = 1
        raw_k = raw_k.view(batch_size, seq_length, self.n_heads,
                       self.single_head_dim)  # batch_size x sequence_length x n_heads x single_head_dim
        raw_q = raw_q.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        raw_v = raw_v.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        # linear transform
        comQ = self.com_q(raw_k)
        comK = self.com_k(raw_q)
        comV = self.com_v(raw_v)
        comQ = comQ.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        comK = comK.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        comV = comV.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = comK.transpose(-1, -2)  # (batch_size, n_heads, single_head_dim, seq_len)
        product = torch.matmul(comQ, k_adjusted)  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10)
        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)
        # applying softmax
        scores = F.softmax(product, dim=-1)
        # mutiply with value matrix
        scores = torch.matmul(scores, comV)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)
        # concatenated output
        concat_multiAtt = scores.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                          self.single_head_dim * self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        multiAtt_out = self.multi_att_out(concat_multiAtt)
        action_out = self.action_out(multiAtt_out)  # (32,10,512) -> (32,10,512)

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


        self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs[0]*n_agents, 256), nn.ReLU())  # may be here can be replaced with another attention mechanism
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

    def forward(self, state, actor_obs):  # state[0] is sum of all agent's own observed states, state[1] is is sum of all agent's observed grid maps
        # preprocess of the input data "state", may be shifted to other places
        pre_own_e = torch.tensor(state[0], dtype=torch.float).view(-1, state[0].size).to(self.device)
        sum_own_e = self.sum_own_fc(pre_own_e)
        # preprocess: padding of env_state (for all element in the states)
        for env_state in enumerate(state[1]):
            tobePad_gridObs = list(np.zeros(actor_obs[1] - len(env_state[1]), dtype=int))
            padded_gridObs = env_state[1] + tobePad_gridObs
            state[1][env_state[0]] = padded_gridObs
        pre_env_e = torch.tensor(np.array(state[1]), dtype=torch.float).view(-1, np.array(state[1]).size).to(self.device)
        sum_env_e = self.sum_env_fc(pre_env_e)
        # preprocess for surr neighbours
        afterPro_surr_neigh = []
        for single_sur_nei in enumerate(state[2]):
            if len(single_sur_nei[1]) == 0:
                padding_surNeigh = np.zeros((1,32))
                afterPro_surr_neigh.append(padding_surNeigh)
            else:  # use attention to convert the nx6 to 1x32
                # perform attention here
                pass
        sur_nei = torch.tensor(np.array(afterPro_surr_neigh), dtype=torch.float).view(-1, np.array(afterPro_surr_neigh).size).to(self.device)






        combine_state_e = self.combine_own_env_fc([sum_own_e, sum_env_e])
        sum_action_e = self.sum_agents_action_fc(state[2])
        multi_dim_out = self.multi_attention([combine_state_e, sum_action_e])
        q = self.judgement_fc(multi_dim_out)

        return q

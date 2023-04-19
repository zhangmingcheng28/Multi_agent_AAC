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

        intru_e = self.surr_drone(state[2])

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
        self.multi_att_out = nn.Linear(self.n_heads * self.single_head_dim, 128)

        self.combine_own_env_fc = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU())

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
        sum_own_e = self.sum_own_fc(state[0])
        sum_env_e = self.sum_env_fc(state[1])
        sum_sur_nei = self.sum_sur_fc(state[2])

        # mask attention embedding
        sum_query = self.sum_q(sum_own_e)
        sum_key = self.sum_k(sum_sur_nei)
        sum_value = self.sum_v(sum_sur_nei)
        mask = state[2].mean(axis=2, keepdim=True).bool()
        score = torch.bmm(sum_key, sum_query.transpose(1, 2))
        score_mask = score.clone()  # clone操作很必要
        score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度

        alpha = F.softmax(score_mask / np.sqrt(sum_key.size(-1)), dim=1)
        alpha_mask = alpha.clone()
        alpha_mask[~mask] = 0
        v_att = torch.sum(sum_value * alpha_mask, axis=1)

        critic_concat = torch.cat((sum_own_e.squeeze(dim=1), sum_env_e.squeeze(dim=1), v_att), dim=1)
        # perform self attention on concatenated
        raw_k = critic_concat
        raw_q = critic_concat
        raw_v = critic_concat
        batch_size, inputDim = critic_concat.shape
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

        q = self.judgement_fc(multiAtt_out)

        return q

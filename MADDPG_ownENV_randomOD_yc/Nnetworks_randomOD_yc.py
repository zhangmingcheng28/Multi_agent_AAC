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
from torch.distributions import Normal
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork, self).__init__()

        self.rnn_hidden_dim = actor_hidden_state_size
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())

        self.own_se_out = nn.Sequential(nn.Linear(64+64, 64+64), nn.ReLU(),
                                           nn.Linear(64+64, n_actions), nn.Tanh())

    def forward(self, obs_state, grid_state, history_info):
        own_s = self.own_fc(obs_state)

        own_e = self.own_grid(grid_state)
        SE_feature = torch.cat((own_s, own_e), dim=1)
        action_out = self.own_se_out(SE_feature)

        return action_out


class GRU_actor(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRU_actor, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # gru layer
        self.gru = nn.GRU(actor_dim[0], actor_hidden_state_size, 1, batch_first=True)

        self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_info):
        own_e = self.own_fc(cur_state)
        gru_out, hn = self.gru(history_info)  # hn = last column (or the most recent one) of the output hidden state
        hn_owne = torch.cat((own_e, hn.squeeze(0)),dim=1)
        action_out = self.own_fc_outlay(hn_owne)
        return action_out, hn


class GRUCELL_actor(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRUCELL_actor, self).__init__()
        self.rnn_hidden_dim = actor_hidden_state_size
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        self.own_fc_outlay = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_hidden_state):
        own_e = self.own_fc(cur_state)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.gru_cell(own_e, h_in)
        action_out = self.own_fc_outlay(h)
        return action_out, h


class Stocha_actor(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(Stocha_actor, self).__init__()
        init_w = 3e-3
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 256), nn.ReLU())
        self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU())
        self.log_std_min = -20
        self.log_std_max = 2
        self.mean_linear = nn.Linear(256, n_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(256, n_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        own_e = self.own_fc(state[0])
        own_hidden = self.own_fc_lay2(own_e)
        mean = self.mean_linear(own_hidden)
        log_std = self.log_std_linear(own_hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        normal = Normal(0, 1)  # this is the re-parameterization trick
        z = normal.sample(mean.shape)
        action = torch.tanh(mean + std * z.to(device))
        return action


class CriticNetwork(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions):
        super(CriticNetwork, self).__init__()

        self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0])*n_agents, 256), nn.ReLU())

        self.sum_combine_grid = nn.Sequential(nn.Linear((critic_obs[1])*n_agents, 256), nn.ReLU())

        self.combine_all = nn.Sequential(nn.Linear(256+256+(n_actions * n_agents), 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, sum_obs_state, sum_grid_state, actor_actions):  # state[0] is sum of all agent's own observed states, state[1] is is sum of all agent's observed grid maps

        combine_obs_state = self.sum_combine_fc(sum_obs_state)
        combine_grid_state = self.sum_combine_grid(sum_grid_state)

        combine_SEA = torch.cat((combine_obs_state, combine_grid_state, actor_actions), dim=1)

        q = self.combine_all(combine_SEA)

        return q


class CriticNetwork_woGru(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, history_horizon_step):
        super(CriticNetwork_woGru, self).__init__()
        self.combine_state_fc = nn.Sequential(nn.Linear((critic_obs[0]) * n_agents, 256), nn.ReLU()) # extract combine state information
        self.combine_hn_fc = nn.Sequential(nn.Linear(history_horizon_step * (critic_obs[0]) * n_agents, 512), nn.ReLU()) # extract history state information
        self.sum_inputs = nn.Sequential(nn.Linear(256+512+(n_actions * n_agents), 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))  # obtain Q value

    def forward(self, state, actor_actions, history_info):
        combine_state = self.combine_state_fc(state)
        combine_hn = self.combine_hn_fc(history_info)
        combine_S_A_hn = torch.cat((combine_state, combine_hn, actor_actions), dim=1)
        q = self.sum_inputs(combine_S_A_hn)
        return q


class CriticNetwork_wGru(nn.Module):  #
    def __init__(self, critic_obs, n_agents, n_actions, combine_history):
        super(CriticNetwork_wGru, self).__init__()
        self.combine_state_fc = nn.Sequential(nn.Linear((critic_obs[0]) * n_agents, 256), nn.ReLU()) # extract combine state information
        # gru layer
        # self.gru = nn.GRU((critic_obs[0]) * n_agents, 256, 1, batch_first=True)
        self.gru = nn.GRU(critic_obs[0], 256, 1, batch_first=True)

        self.sum_inputs = nn.Sequential(nn.Linear(256+256+(n_actions * n_agents), 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))  # obtain Q value

    def forward(self, state, actor_actions, history_info):
        combine_state = self.combine_state_fc(state)
        combine_out, combine_hn = self.gru(history_info)
        combine_S_A_hn = torch.cat((combine_state, combine_hn.squeeze(0), actor_actions), dim=1)
        q = self.sum_inputs(combine_S_A_hn)
        return q


class critic_single_obs_wGRU(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_wGRU, self).__init__()
        self.rnn_hidden_dim = hidden_state_size
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.gru_cell = nn.GRUCell(64, hidden_state_size)
        self.own_fc_outlay = nn.Linear(64, 1)

    def forward(self, single_state, single_action, history_hidden_state):
        SA_combine = torch.cat((single_state, single_action), dim=1)
        SA_feature = self.SA_fc(SA_combine)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.gru_cell(SA_feature, h_in)
        q = self.own_fc_outlay(h)
        return q, h


class CriticNetwork_0724(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions):
        super(CriticNetwork_0724, self).__init__()

        # in critic network we should use multi-head attention mechanism to help to capture more complex relationship
        # between different inputs, in the context of this paper, the input consists of many drone's states as well as
        # their actions. This is two group of inputs, therefore my hypothesis is that using multi-head attention is
        # better here.

        # critic_obs[0] is sum of all agent's own observed states
        # critic_obs[1] is sum of all agent's observed grid maps
        # critic_obs[3] is sum of all agent's action taken

        # self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs*n_agents, 1024), nn.ReLU())  # may be here can be replaced with another attention mechanism
        self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs[0]*n_agents, 256), nn.ReLU())  # may be here can be replaced with another attention mechanism
        self.sum_grid_fc = nn.Sequential(nn.Linear(critic_obs[1]*n_agents, 128), nn.ReLU())

        # self.single_own_fc = nn.Sequential(nn.Linear(critic_obs[0], 128), nn.ReLU())  # may be here can be replaced with another attention mechanism
        self.single_own_fc = nn.Sequential(nn.Linear(critic_obs[0], 256), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.single_grid_fc = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        self.single_grid_fc = nn.Sequential(nn.Linear(critic_obs[1], 256), nn.ReLU())
        self.single_surr = nn.Sequential(nn.Linear(critic_obs[2], 128), nn.ReLU())

        # for surrounding agents' encoding, for each agent, we there are n-neighbours, each neighbour is represented by
        # a vector of length = 6. Before we put into an experience replay, we pad it up to max_num_neigh * 6 array.
        # so, one agent will have an array of max_num_neigh * 6, after flatten, then for one batch, there are a total of
        # n_agents exist in the airspace, therefore, the final dimension will be max_num_neigh * 6 * max_num_neigh.
        # self.sum_sur_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents*n_agents, 256), nn.ReLU())

        # critic attention for overall sur_neighbours with overall own_state
        self.single_k = nn.Linear(128, 128, bias=False)
        self.single_q = nn.Linear(128, 128, bias=False)
        self.single_v = nn.Linear(128, 128, bias=False)
        #
        # self.n_heads = 3
        # self.single_head_dim = int((256+256+256) / self.n_heads)
        # self.com_k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.multi_att_out = nn.Sequential(nn.Linear(self.n_heads * self.single_head_dim + n_agents * n_actions, 128),
        #                                    nn.ReLU())
        #
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+256+256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
        #                                     nn.Linear(128, 64), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+128, 128), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear((n_agents*128)+(n_agents*128), 64), nn.ReLU())
        self.combine_env_fc = nn.Sequential(nn.Linear((n_agents*256)+(n_agents*256), 1028), nn.ReLU(), nn.Linear(1028, 64), nn.ReLU())

        self.combine_all = nn.Sequential(nn.Linear(64+n_agents * n_actions, 64), nn.ReLU(), nn.Linear(64, 1))

        # self.sum_agents_action_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents, 256), nn.ReLU())
        #
        # self.multi_attention = nn.MultiheadAttention(embed_dim=256+256, num_heads=2)  # 1st input is the sum of neurons from actions and combined states encoding
        #
        # # the input of this judgement layer is 256+256 because this is right after the multi-head attention layer
        # # the output dimension of the multi-head attention is default to be the dimension of the "embed_dim"
        # self.judgement_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # self.name = name

        # self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #
        # self.to(self.device)

    def forward(self, state, actor_actions):  # state[0] is sum of all agent's own observed states, state[1] is is sum of all agent's observed grid maps
        # pre-process, compute attention for every agent based on their surrounding agents
        attention_all_agent = []
        grid_all_agent = []
        own_all_agent = []
        for one_agent_batch_own, one_agent_batch_grid, one_agent_batch_surr in zip(*state):  # automatically loop over 5 times
            single_grid_out = self.single_grid_fc(one_agent_batch_grid)
            single_own_out = self.single_own_fc(one_agent_batch_own)

            # single_surr_out = self.single_surr(one_agent_batch_surr)
            # single_q = self.single_q(single_own_out)
            # single_k = self.single_k(single_surr_out)
            # single_v = self.single_v(single_surr_out)
            # mask = one_agent_batch_surr.mean(axis=2, keepdim=True).bool()
            # score = torch.bmm(single_k, single_q.unsqueeze(axis=2))
            # score_mask = score.clone()  # clone操作很必要
            # score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度
            # alpha = F.softmax(score_mask / np.sqrt(single_k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
            # alpha_mask = alpha.clone()
            # alpha_mask[~mask] = 0
            # v_att = torch.sum(single_v * alpha_mask, axis=1)
            # attention_all_agent.append(v_att)

            grid_all_agent.append(single_grid_out)
            own_all_agent.append(single_own_out)


        # sum_att = torch.stack(attention_all_agent).transpose(0, 1)
        # sum_att = sum_att.reshape(sum_att.shape[0], -1)

        sum_grid = torch.stack(grid_all_agent).transpose(0, 1)
        sum_grid = sum_grid.reshape(sum_grid.shape[0], -1)

        sum_own = torch.stack(own_all_agent).transpose(0, 1)
        sum_own = sum_own.reshape(sum_own.shape[0], -1)

        # sum_own = self.sum_own_fc(state[0])
        # sum_grid = self.sum_grid_fc(state[1])

        # env_concat = torch.cat((sum_att, sum_grid), dim=1)
        env_concat = torch.cat((sum_own, sum_grid), dim=1)

        env_encode = self.combine_env_fc(env_concat)
        entire_comb = torch.cat((env_encode, actor_actions), dim=1)
        # entire_comb = torch.cat((sum_own_e, actor_actions), dim=1)
        q = self.combine_all(entire_comb)
        return q

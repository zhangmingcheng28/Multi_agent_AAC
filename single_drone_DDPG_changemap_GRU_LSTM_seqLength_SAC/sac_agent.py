# -*- coding: utf-8 -*-
"""
@Time    : 1/4/2025 3:00 pm
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

# from Nnetworks_MADDPGv3 import CriticNetwork_0724, ActorNetwork
from Nnetworks_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength_sac import *
import torch
from copy import deepcopy
from torch.optim import Adam
from memory_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength_sac import ReplayMemory, Experience
# from random_process_MADDPGv3_randomOD import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import torch.nn as nn
import time
import numpy as np
import torch as T
from utils_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength_sac import device
from Utilities_own_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength_sac import *
import csv



class SAC:
    def __init__(self, actor_dim, critic_dim, dim_act, actor_hidden_state_size, gru_history_length, n_agents, args, cr_lr, ac_lr, gamma, tau, full_observable_critic_flag, use_GRU_flag, use_attention_flag, attention_only, use_LSTM_flag, stacking, feature_matching):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics1 = []
        self.critics2 = []
        self.values = []
        self.n_agents = n_agents
        self.n_actor_dim = actor_dim
        self.n_critic_dim = critic_dim
        self.n_actions = dim_act
        self.action_range = 8

        self.actors = [sac_ActorNetwork_TwoPortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
        self.values = [sac_value_single_TwoPortion(critic_dim) for _ in range(n_agents)]  # use deterministic policy
        self.critics1 = [
            sac_critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for
            _ in range(n_agents)]
        self.critics2 = [
            sac_critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for
            _ in range(n_agents)]

        self.values_target = deepcopy(self.values)

        self.memory = ReplayMemory(args.memory_length, gru_history_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        # self.episodes_before_train = args.episode_before_train

        # self.GAMMA = 0.95  # original
        # self.tau = 0.01  # original

        self.GAMMA = gamma
        self.tau = tau

        self.var = [1.0 for i in range(n_agents)]
        # self.var = [0.5 for i in range(n_agents)]

        # original, critic learning rate is 10 times larger compared to actor
        # self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        # self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr, weight_decay=1e-4) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr, weight_decay=1e-4) for x in self.actors]

        self.critic_optimizer1 = [Adam(x.parameters(), lr=cr_lr, weight_decay=1e-5) for x in self.critics1]
        self.critic_optimizer2 = [Adam(x.parameters(), lr=cr_lr, weight_decay=1e-5) for x in self.critics2]
        self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr, weight_decay=1e-5) for x in self.actors]
        self.values_optimizer = [Adam(x.parameters(), lr=cr_lr, weight_decay=1e-5) for x in self.values]

        # self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics1:
                x.cuda()
            for x in self.critics2:
                x.cuda()
            for x in self.values:
                x.cuda()
            for x in self.values_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0


    def choose_action(self, OU_noise, state, cur_total_step, cur_episode, step, total_training_steps, noise_start_level, actor_hiddens, lstm_hist, gru_hist, use_LSTM_flag, stacking, feature_matching, noisy=True, use_GRU_flag=False):
        # ------------- MADDPG_test_181123_10_10_54 version noise -------------------
        obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        noise_value = np.zeros(2)

        actions = torch.zeros(self.n_agents, self.n_actions)
        if use_GRU_flag:
            act_hn = torch.zeros(self.n_agents, self.actors[0].rnn_hidden_dim)
        elif use_LSTM_flag:
            act_hn = torch.zeros(self.n_agents, self.actors[0].rnn_hidden_dim)
        else:
            act_hn = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # this for loop used to decrease noise level for all agents before taking any action
        # gru_history_input = torch.FloatTensor(gru_history_input).to(device)  # batch x seq_length x no_agent x feature_length
        gru_history_input = torch.FloatTensor(actor_hiddens).unsqueeze(0).to(device)  # batch x no_agent x feature_length
        for i in range(self.n_agents):
            self.var[i] = self.get_custom_linear_scaling_factor(cur_episode, total_training_steps, noise_start_level)  # self.var[i] will decrease as the episode increase
            # self.var[i] = self.linear_decay(episode, eps_end, noise_start_level)  # self.var[i] will decrease as the episode increase

        for i in range(self.n_agents):
            sb = obs[i]
            sb_grid = obs_grid[i]

            if use_GRU_flag:
                if feature_matching:
                    act, gru_hist, _ = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)],
                                                   gru_hist)
                else:
                    act, gru_hist = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)], gru_hist)
            elif use_LSTM_flag:
                act, lstm_hist = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)], lstm_hist)
            elif stacking:
                act, gru_hist, lstm_hist = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)], gru_hist, lstm_hist)
            else:
                if feature_matching:
                    act, _ = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)], use_random=False)
                else:
                    # act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)])
                    mean, log_std  = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)])
                    std = log_std.exp()
                    normal = Normal(0, 1)
                    z = normal.sample(mean.shape).to(device)
                    act = self.action_range * torch.tanh(mean + std * z)

            actions[i, :] = act
            if use_GRU_flag:
                pass
            elif use_LSTM_flag:  # don't change act_hn in this loop
                pass
            else:
                act_hn[i, :] = torch.zeros(1, self.n_actions)
            # self.actors[i].train()
        self.steps_done += 1
        # ------------- end of MADDPG_test_181123_10_10_54 version noise -------------------
        return actions.data.cpu().numpy(), noise_value, lstm_hist, gru_hist, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()



    def update(self, batch_size, reward_scale, gamma=0.99, soft_tau=1e-2):
        alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

        state, action, reward, next_state, done = self.memory.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = soft_q_net1(state, action)  # actor
        predicted_q_value2 = soft_q_net2(state, action)
        predicted_value = value_net(state)
        new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

        reward = reward_scale * (reward - reward.mean(dim=0)) / reward.std(dim=0)  # normalize with batch mean and std

        # Training Q Function
        target_value = target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value  # if done==1, only reward
        q_value_loss1 = soft_q_criterion1(predicted_q_value1,
                                          target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        soft_q_optimizer1.step()
        soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        soft_q_optimizer2.step()

        # Training Value Function
        predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - alpha * log_prob  # for stochastic training, it equals to expectation over action
        value_loss = value_criterion(predicted_value, target_value_func.detach())

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Training Policy Function
        ''' implementation 1 '''
        policy_loss = (alpha * log_prob - predicted_new_q_value).mean()
        ''' implementation 2 '''
        # policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean()  # Openai Spinning Up implementation
        ''' implementation 3 '''
        # policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() # max Advantage instead of Q to prevent the Q-value drifted high

        ''' implementation 4 '''  # version of github/higgsfield
        # log_prob_target=predicted_new_q_value - predicted_value
        # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        # mean_lambda=1e-3
        # std_lambda=1e-3
        # mean_loss = mean_lambda * mean.pow(2).mean()
        # std_loss = std_lambda * log_std.pow(2).mean()
        # policy_loss += mean_loss + std_loss

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # print('value_loss: ', value_loss)
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()


    def update_myown(self, i_episode, total_step_count, UPDATE_EVERY, single_eps_critic_cal_record, action, wandb=None, full_observable_critic_flag=False, use_GRU_flag=False):

        self.train_num = i_episode

        if len(self.memory) <= self.batch_size:
        # if True:
            return None, None, single_eps_critic_cal_record

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        value_criterion = nn.MSELoss()
        soft_q_criterion1 = nn.MSELoss()
        soft_q_criterion2 = nn.MSELoss()
        alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

        c1_loss = []
        c2_loss = []
        value_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        done_batch = torch.stack(batch.dones).type(FloatTensor)
        if use_GRU_flag:
            agents_next_hidden_state = torch.stack(batch.next_hidden).type(FloatTensor)
            agents_cur_hidden_state = torch.stack(batch.cur_hidden).type(FloatTensor)
        # stack tensors only once
        stacked_elem_0 = torch.stack([elem[0] for elem in batch.states]).to(device)
        stacked_elem_1 = torch.stack([elem[1] for elem in batch.states]).to(device)
        if full_observable_critic_flag == True:
            stacked_elem_0_combine = stacked_elem_0.view(self.batch_size, -1)  # own_state only
            stacked_elem_1_combine = stacked_elem_1.view(self.batch_size, -1)  # own_state only

        # use the stacked tensors
        # current_state in the form of list of length of agents in the environments, then, batchNo X individual Feature length
        # cur_state_list1 = [stacked_elem_0[:, i, :] for i in range(self.n_agents)]

        # for next state
        next_stacked_elem_0 = torch.stack([elem[0] for elem in batch.next_states]).to(device)
        next_stacked_elem_1 = torch.stack([elem[1] for elem in batch.next_states]).to(device)
        if full_observable_critic_flag == True:
            next_stacked_elem_0_combine = next_stacked_elem_0.view(self.batch_size, -1)
            next_stacked_elem_1_combine = next_stacked_elem_1.view(self.batch_size, -1)

        # for done
        dones_stacked = torch.stack([three_agent_dones for three_agent_dones in batch.dones]).to(device)

        for agent in range(self.n_agents):
            # whole_ownState = stacked_elem_0_combine  # own_state only

            # non_final_next_states_actorin = [next_stacked_elem_0]  # 2 portion available
            non_final_next_states_actorin = [next_stacked_elem_0, next_stacked_elem_1]  # 2 portion available

            # configured for target Q

            whole_curren_action = action_batch.view(self.batch_size, -1)

            predicted_q_value1 = self.critics1[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], action_batch[:, agent, :])
            predicted_q_value2 = self.critics2[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], action_batch[:, agent, :])
            predicted_value = self.values[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]])
            new_action, log_prob, z, mean, log_std = self.actors[agent].evaluate([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], self.action_range)

            # with/wo, reward scaling
            # reward_scale = 10
            # reward_batch = reward_scale * (reward_batch - reward_batch.mean(dim=0)) / reward_batch.std(
            #     dim=0)  # normalize with batch mean and std

            target_value = self.values_target[agent]([next_stacked_elem_0[:, agent, :], next_stacked_elem_1[:, agent, :]])
            target_q_value = reward_batch + (1 - done_batch) * self.GAMMA * target_value  # if done==1, only reward
            q_value_loss1 = soft_q_criterion1(predicted_q_value1,
                                              target_q_value.detach())  # detach: no gradients for the variable
            q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

            # training Q function
            self.critic_optimizer1[agent].zero_grad()
            q_value_loss1.backward()
            self.critic_optimizer1[agent].step()

            self.critic_optimizer2[agent].zero_grad()
            q_value_loss2.backward()
            self.critic_optimizer2[agent].step()

            # Training Value Function
            predicted_new_q_value = torch.min(self.critics1[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], new_action),
                                              self.critics2[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], new_action))
            target_value_func = predicted_new_q_value - alpha * log_prob  # for stochastic training, it equals to expectation over action
            value_loss = value_criterion(predicted_value, target_value_func.detach())

            self.values_optimizer[agent].zero_grad()
            value_loss.backward()
            self.values_optimizer[agent].step()

            # Training Policy Function
            ''' implementation 1 '''
            policy_loss = (alpha * log_prob - predicted_new_q_value).mean()
            ''' implementation 2 '''
            # policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean()  # Openai Spinning Up implementation
            ''' implementation 3 '''
            # policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() # max Advantage instead of Q to prevent the Q-value drifted high

            ''' implementation 4 '''  # version of github/higgsfield
            # log_prob_target=predicted_new_q_value - predicted_value
            # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
            # mean_lambda=1e-3
            # std_lambda=1e-3
            # mean_loss = mean_lambda * mean.pow(2).mean()
            # std_loss = std_lambda * log_std.pow(2).mean()
            # policy_loss += mean_loss + std_loss

            self.actor_optimizer[agent].zero_grad()
            policy_loss.backward()
            self.actor_optimizer[agent].step()

            if i_episode % UPDATE_EVERY == 0:  # perform a soft update at each step of an episode.
                # Soft update the target value net
                for target_param, param in zip(self.values_target[agent].parameters(), self.values[agent].parameters()):
                    target_param.data.copy_(  # copy data value into target parameters
                        target_param.data * (1.0 - self.tau) + param.data * self.tau
                    )
            return predicted_new_q_value.mean()

    def get_custom_linear_scaling_factor(self, episode, eps_end, start_scale=1, end_scale=0.05):
        # Calculate the slope of the linear decrease only up to eps_end
        if episode <= eps_end:
            slope = (end_scale - start_scale) / (eps_end - 1)
            current_scale = start_scale + slope * (episode - 1)
        else:
            current_scale = end_scale
        return current_scale
# -*- coding: utf-8 -*-
"""
@Time    : 2/4/2025 4:11 pm
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
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.autograd import Variable
import os
import torch.nn as nn
import time
import numpy as np
import torch as T
from utils_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength_sac import device
from Utilities_own_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength_sac import *
import csv

action_std_init = 0.6
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(84000)  # action_std decay frequency (in num timesteps), original fraction
# action_std_decay_freq = int(500)  # action_std decay frequency (in num timesteps)
has_continuous_action_space = True


class ActorCritic(nn.Module):
    def __init__(self, state_dim, critic_dim, action_dim, has_continuous_action_space, action_std_init, n_agents):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self._n_agents = n_agents
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = [ppo_ActorNetwork_TwoPortion(state_dim, action_dim) for _ in range(n_agents)]
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = [
            ppo_critic_single_TwoPortion(critic_dim, action_dim) for
            _ in range(n_agents)]

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, ith_agent):

        if self.has_continuous_action_space:
            action_mean = self.actor[ith_agent](state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        # action = torch.tanh(dist.sample())
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, ith_agent, action_range):

        if self.has_continuous_action_space:
            action_mean = self.actor[ith_agent](state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic[ith_agent](state, action)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, actor_dim, critic_dim, dim_act, actor_hidden_state_size, gru_history_length, n_agents, args,
                 cr_lr, ac_lr, gamma, tau, full_observable_critic_flag, use_GRU_flag, use_attention_flag,
                 attention_only, use_LSTM_flag, stacking, feature_matching):
        self.args = args
        self.mode = args.mode
        self.n_agents = n_agents
        self.n_actor_dim = actor_dim
        self.n_critic_dim = critic_dim
        self.n_actions = dim_act
        self.action_range = 8
        self.MseLoss = nn.MSELoss()
        self.eps_clip = 0.2
        self.K_epochs = 80
        self.memory = ReplayMemory(args.memory_length, gru_history_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        self.action_std = action_std_init

        self.policy = ActorCritic(actor_dim, critic_dim, dim_act, has_continuous_action_space, action_std_init, n_agents).to(device)
        self.policy_old = ActorCritic(actor_dim, critic_dim, dim_act, has_continuous_action_space, action_std_init, n_agents).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [{'params': module.parameters(), 'lr': ac_lr} for module in self.policy.actor] +
            [{'params': module.parameters(), 'lr': cr_lr} for module in self.policy.critic]
        )

        self.GAMMA = gamma
        self.tau = tau

        self.var = [1.0 for i in range(n_agents)]

        if self.use_cuda:
            self.policy.actor = [module.cuda() for module in self.policy.actor]
            self.policy.critic = [module.cuda() for module in self.policy.critic]

            self.policy_old.actor = [module.cuda() for module in self.policy_old.actor]
            self.policy_old.critic = [module.cuda() for module in self.policy_old.critic]

        self.steps_done = 0
        self.episode_done = 0

    def choose_action(self, OU_noise, state, cur_total_step, cur_episode, step, total_training_steps, noise_start_level, actor_hiddens, lstm_hist, gru_hist, use_LSTM_flag, stacking, feature_matching, noisy=True, use_GRU_flag=False):
        obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        noise_value = np.zeros(2)

        actions = torch.zeros(self.n_agents, self.n_actions)
        actions_logprob = torch.zeros(self.n_agents, 1)
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
                    action, action_logprob = self.policy_old.act([sb.unsqueeze(0), sb_grid.unsqueeze(0)], i)
                    # act = self.action_range * action
                    act = action
                    # with/wo action range for logprob
                    # act_logprob = self.action_range * action_logprob
                    act_logprob = action_logprob

            actions[i, :] = act
            actions_logprob[i, :] = act_logprob
            if use_GRU_flag:
                pass
            elif use_LSTM_flag:  # don't change act_hn in this loop
                pass
            else:
                act_hn[i, :] = torch.zeros(1, self.n_actions)
            # self.actors[i].train()
        self.steps_done += 1
        # ------------- end of MADDPG_test_181123_10_10_54 version noise -------------------
        return actions.data.cpu().numpy(), actions_logprob.data.cpu().numpy(), noise_value, lstm_hist, gru_hist, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()

    def get_custom_linear_scaling_factor(self, episode, eps_end, start_scale=1, end_scale=0.05):
        # Calculate the slope of the linear decrease only up to eps_end
        if episode <= eps_end:
            slope = (end_scale - start_scale) / (eps_end - 1)
            current_scale = start_scale + slope * (episode - 1)
        else:
            current_scale = end_scale
        return current_scale

    def update_myown(self, i_episode, total_step_count, UPDATE_EVERY, single_eps_critic_cal_record, action,
                     wandb=None, full_observable_critic_flag=False, use_GRU_flag=False):

        self.train_num = i_episode

        # if continuous action space; then decay action std of ouput action distribution
        # if self.policy.has_continuous_action_space and total_step_count % action_std_decay_freq == 0:
        self.decay_action_std(action_std_decay_rate, min_action_std, i_episode, total_step_count)

        if len(self.memory) <= self.batch_size:
            return None, None, None, None, None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        old_act_logprob_batch = torch.stack(batch.act_logprob).type(FloatTensor)

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
            # Monte Carlo estimate of returns

            # Squeeze the tensors to have shape (512,) for easier iteration
            rewards_squeeze_tensor = reward_batch.squeeze(1)
            terminals_squeeze_tensor = dones_stacked.squeeze(1)

            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rewards_squeeze_tensor), reversed(terminals_squeeze_tensor)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.GAMMA * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Optionally, convert the list of rewards back to a tensor with shape (512, 1)
            rewards_discounted = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            # rewards_discounted = (rewards_discounted - rewards_discounted.mean()) / (rewards_discounted.std() + 1e-7)
            # Optimize policy for K epochs
            for _ in range(self.K_epochs):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate([stacked_elem_0[:, agent, :],
                                                                             stacked_elem_1[:, agent, :]],
                                                                            action_batch[:, agent, :], agent, self.action_range)

                # Calculate average entropy for the minibatch
                avg_entropy = dist_entropy.mean().item()

                # match state_values tensor dimensions with rewards tensor
                # state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_act_logprob_batch[:, agent, :].squeeze(1).detach())
                avg_ratios = ratios.mean().item()
                # Finding Surrogate Loss
                advantages = rewards_discounted - state_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize advantage
                adv_mean = advantages.mean()
                adv_variance = advantages.var()  # or .std() for standard deviation
                print("Advantage Mean:", adv_mean.item())
                print("Advantage Variance:", adv_variance.item())

                surr1 = ratios.unsqueeze(1) * advantages

                # and eps_clip is your clipping parameter
                lower_bound = 1 - self.eps_clip
                upper_bound = 1 + self.eps_clip
                # Create a mask for samples that are outside the clipping bounds
                clipped_mask = (ratios < lower_bound) | (ratios > upper_bound)
                n_clipped = clipped_mask.sum().item()
                fraction_clipped = n_clipped / ratios.numel()
                print(f"Fraction of samples clipped: {fraction_clipped:.2%}")

                surr2 = torch.clamp(ratios.unsqueeze(1), 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                dist_entropy = dist_entropy.unsqueeze(1)
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_discounted) - 0.01 * dist_entropy
                # loss = -torch.min(surr1, surr2) - 0.5 * self.MseLoss(state_values, rewards_discounted) + 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                for name, param in self.policy.actor[agent].named_parameters():
                    if param.grad is not None:
                        if name == 'mean_linear.0.weight':
                            actor_last_layer_weight = param.grad.norm()
                        if name == 'mean_linear.0.bias':
                            actor_last_layer_bias = param.grad.norm()
                for name, param in self.policy.critic[agent].named_parameters():
                    if param.grad is not None:
                        if name == 'out_feature_q.0.weight':
                            critic_last_layer_weight = param.grad.norm()
                        if name == 'out_feature_q.0.bias':
                            critic_last_layer_bias = param.grad.norm()
                self.optimizer.step()

            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

            # clear buffer for PPO
            self.memory.clear()

        return loss, actor_last_layer_weight, actor_last_layer_bias, critic_last_layer_weight, critic_last_layer_bias, avg_entropy, avg_ratios

    def decay_action_std(self, action_std_decay_rate, min_action_std, episode, total_step):
        # print("--------------------------------------------------------------------------------------------")
        if has_continuous_action_space:
            if total_step % 84000 == 0:
                self.action_std = self.action_std - action_std_decay_rate
            # self.action_std = self.get_custom_linear_scaling_factor(episode, 5000, start_scale=0.6, end_scale=0.1)
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                # print("current eps is : {} setting actor output action_std to min_action_std : {}".format(episode, self.action_std))
            else:
                pass
                # print("current eps is : {} setting actor output action_std to min_action_std : {}".format(episode, self.action_std))
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        # print("--------------------------------------------------------------------------------------------")

    def set_action_std(self, new_action_std):

        if has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def save_model(self, episode, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(self.n_agents):
            torch.save(self.policy_old.actor[i].state_dict(), file_path + '/' +'episode_'+str(episode)+'_'+'agent_'+ str(i) + 'actor_net.pth')

    def load_model(self, file_path):
        if self.args.model_episode:
            print("load model!")
            for path_idx, path in enumerate(file_path):
                self.policy_old.actor[path_idx].load_state_dict(torch.load(path))
                self.policy.actor[path_idx].load_state_dict(torch.load(path))
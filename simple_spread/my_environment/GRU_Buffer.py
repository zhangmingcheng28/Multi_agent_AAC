#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MADDPG_GRU
@File ：GRU_Buffer.py
@Author ：
@Date : 2023/12/18
"""
import numpy as np
import torch


class GRU_Buffer:
    """
    replay buffer for each agent;
    使用rnn的时候, 会传递hidden特征, 保存整个episode;
    """

    def __init__(self, capacity ,obs_dim, act_dim, hidden_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_obs = np.zeros((capacity, obs_dim))
        self.hiddens = np.zeros((capacity, hidden_dim))
        self.next_hiddens = np.zeros((capacity, hidden_dim))
        self.done = np.zeros((capacity, 1), dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device


    def add(self, obs, action, hidden, reward, next_obs,next_hidden, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.hiddens[self._index] = hidden
        self.next_hiddens[self._index] = next_hidden
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        next_hiddens = self.next_hiddens[indices]
        hiddens = self.hiddens[indices]
        done = self.done[indices]

        # obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        # action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        # reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        # next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        # hiddens = torch.from_numpy(hiddens).float().to(self.device)
        # done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size

        return obs, action, hiddens, reward, next_obs, next_hiddens , done

    def __len__(self):
        return self._size

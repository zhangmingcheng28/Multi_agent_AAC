# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2023 10:44 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np


class MultiAgentReplayBuffer:  # centralized training, so this replay memeory only stores information from critics
    def __init__(self, max_size, critic_dims, actor_dim, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.n_agents = n_agents
        self.actor_dims = actor_dim
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

    def store_transition(self, cur_obs, cur_state, cur_action, cur_reward, next_obs, next_state, done):
        pass

    def sample_buffer(self):
        pass

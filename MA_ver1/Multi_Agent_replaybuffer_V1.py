# -*- coding: utf-8 -*-
"""
@Time    : 4/3/2023 7:48 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from collections import deque
import random
import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, actor_dims, critic_dims, n_agents, n_actions, batch_size):

        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = []
        self.new_state_memory = []
        self.reward_memory = []
        self.terminal_memory = []

        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        self.memory = deque(maxlen=self.mem_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        one_batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        raw_cur_state, action, reward, raw_next_state, done = zip(*one_batch)

        cur_state = []
        # transform every batch into another arrangement
        agent_batch_cur_state = []
        for batch_element in raw_cur_state:
            for agent_idx in range(self.n_agents):
                




        return cur_state, action, reward, next_state, done








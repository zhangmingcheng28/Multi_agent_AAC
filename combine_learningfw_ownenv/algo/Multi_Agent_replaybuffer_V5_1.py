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
from Utilities_V5_1 import padding_list


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

    def sample(self, batch_size, maxIntruNum, intruFeature, max_grid_obs_dim):
        # pick a random batch from the experience replay
        one_batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        raw_cur_state, raw_action, raw_reward, raw_next_state, raw_done = zip(*one_batch)

        # then perform a small preprocess to these states so that they can be easily input into to NN
        cur_state = self.experience_transform_state_v2(raw_cur_state, maxIntruNum, intruFeature, max_grid_obs_dim)
        next_state = self.experience_transform_state_v2(raw_next_state, maxIntruNum, intruFeature, max_grid_obs_dim)
        action = self.experience_transform_action(raw_action)
        done = self.experience_transform_done(raw_done)
        reward = self.experience_transform_reward(raw_reward)
        return cur_state, action, reward, next_state, done  # num_agent X (batch_num x num_features)

    def experience_transform_state(self, input_exp, maxIntruNum, intruFeature, max_grid_obs_dim):
        batched_exp = []
        for agent_idx in range(self.n_agents):
            neigh_coding = np.zeros((maxIntruNum, intruFeature))
            # below 3 are used for individual agent's state or state_ so we need to initialized them for every agent.
            own_state_batch = []
            obs_batch = []
            sur_nei_batch = []
            for batch_idx, batch_val in enumerate(input_exp):
                own_state_batch.append(batch_val[agent_idx][0])
                # padding surrounding grids
                padded_obs_grid = padding_list(max_grid_obs_dim, batch_val[agent_idx][1])
                obs_batch.append(padded_obs_grid)
                # preprocess neighbor information
                if len(batch_val[agent_idx][2]) == 0:  # no neighbour found
                    sur_nei_batch.append(neigh_coding)
                else:
                    for nei_idx, nei_feature in batch_val[agent_idx][2].items():
                        neigh_coding[nei_idx, :] = nei_feature
                    sur_nei_batch.append(neigh_coding)
            batched_exp.append([np.array(own_state_batch, dtype=np.float32), np.array(obs_batch, dtype=np.float32),
                                np.array(sur_nei_batch, dtype=np.float32)])
        return batched_exp

    def experience_transform_state_v2(self, input_exp, maxIntruNum, intruFeature, max_grid_obs_dim):
        batched_exp = []
        for agent_idx in range(self.n_agents):
            neigh_coding = np.zeros((maxIntruNum, intruFeature))
            # below 3 are used for individual agent's state or state_ so we need to initialized them for every agent.
            own_state_batch = []
            obs_batch = []
            sur_nei_batch = []
            for batch_idx, batch_val in enumerate(input_exp):
                own_state_batch.append(batch_val[agent_idx])

            batched_exp.append(np.array(own_state_batch, dtype=np.float32))
        return batched_exp

    def experience_transform_action(self, input_exp):
        batched_exp = []
        for agent_idx in range(self.n_agents):
            # for action or drone or reward
            oneAgent_other_batch = []
            for batch_idx, batch_val in enumerate(input_exp):
                oneAgent_other_batch.append(batch_val[agent_idx])
            batched_exp.append(np.array(oneAgent_other_batch, dtype=np.float32))
        return batched_exp

    def experience_transform_done(self, input_exp):
        batched_exp = []
        for agent_idx in range(self.n_agents):
            # for action or drone or reward
            oneAgent_other_batch = []
            for batch_idx, batch_val in enumerate(input_exp):
                oneAgent_other_batch.append(batch_val[agent_idx])
            batched_exp.append(np.array(oneAgent_other_batch).reshape(-1, 1))
        return batched_exp

    def experience_transform_reward(self, input_exp):
        batched_exp = []
        for agent_idx in range(self.n_agents):
            # for action or drone or reward
            oneAgent_other_batch = []
            for batch_idx, batch_val in enumerate(input_exp):
                oneAgent_other_batch.append(batch_val[agent_idx])
            batched_exp.append(np.array(oneAgent_other_batch, dtype=np.float32).reshape(-1, 1))
        return batched_exp








